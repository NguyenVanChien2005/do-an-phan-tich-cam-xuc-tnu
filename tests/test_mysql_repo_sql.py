"""Kiểm thử tầng SQL tham số hóa, khoảng ngày nửa mở, phân trang tại DB và chống ghi lặp của MySQLFeedbackRepository."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError

from tnu_sentiment.constants import DataOrigin, LabelPolicy, ProcessingStatus, ReviewStatus
from tnu_sentiment.domain.models import FilterCriteria
from tnu_sentiment.repositories.base import NewFeedbackPayload, compute_half_open_date_bounds
from tnu_sentiment.repositories.mysql_repo import MySQLFeedbackRepository


def _create_sqlite_compatible_schema(engine) -> None:
    stmts = [
        """
        CREATE TABLE units (
            unit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_code TEXT NOT NULL UNIQUE,
            unit_name TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """,
        """
        CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_code TEXT NOT NULL UNIQUE,
            source_name TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """,
        """
        CREATE TABLE topics (
            topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_code TEXT NOT NULL UNIQUE,
            topic_name TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """,
        """
        CREATE TABLE feedbacks (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            idempotency_key TEXT UNIQUE,
            source_record_id TEXT,
            content_hash TEXT NOT NULL,
            author_hash TEXT,
            data_origin TEXT NOT NULL,
            origin_note TEXT,
            batch_id TEXT,
            unit_code TEXT NOT NULL DEFAULT 'UNSPECIFIED',
            source_code TEXT NOT NULL DEFAULT 'UNSPECIFIED',
            masked_content TEXT NOT NULL,
            processed_content TEXT,
            pii_types_detected TEXT,
            was_truncated INTEGER NOT NULL DEFAULT 0,
            is_multi_aspect INTEGER NOT NULL DEFAULT 0,
            is_duplicate_candidate INTEGER NOT NULL DEFAULT 0,
            processing_status TEXT NOT NULL DEFAULT 'PENDING',
            processing_error TEXT,
            predicted_sentiment_label INTEGER,
            predicted_sentiment_text TEXT,
            confidence_score REAL,
            prob_negative REAL,
            prob_neutral REAL,
            prob_positive REAL,
            predicted_topic_code TEXT NOT NULL DEFAULT 'UNCLASSIFIED',
            topic_assignment_source TEXT NOT NULL DEFAULT 'MANUAL',
            model_version TEXT,
            pipeline_version TEXT,
            inference_time_ms REAL,
            llm_suggested_sentiment INTEGER,
            llm_suggested_topic_code TEXT,
            llm_note TEXT,
            review_status TEXT NOT NULL DEFAULT 'UNREVIEWED',
            review_reason TEXT,
            reviewed_sentiment_label INTEGER,
            reviewed_topic_code TEXT,
            reviewed_by TEXT,
            reviewed_at TIMESTAMP,
            review_note TEXT,
            record_version INTEGER NOT NULL DEFAULT 1,
            feedback_date TIMESTAMP NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (source_code, source_record_id),
            FOREIGN KEY (unit_code) REFERENCES units (unit_code),
            FOREIGN KEY (source_code) REFERENCES sources (source_code),
            FOREIGN KEY (predicted_topic_code) REFERENCES topics (topic_code)
        )
        """,
    ]

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    with engine.begin() as conn:
        for s in stmts:
            conn.execute(text(s))
        conn.execute(
            text("INSERT INTO units (unit_code, unit_name) VALUES ('ICTU', 'Trường ĐH CNTT&TT')")
        )
        conn.execute(
            text("INSERT INTO sources (source_code, source_name) VALUES ('SURVEY_PORTAL', 'Khảo sát')")
        )
        conn.execute(
            text("INSERT INTO topics (topic_code, topic_name) VALUES ('LECTURER', 'Giảng viên')")
        )


def test_half_open_date_bounds_and_parameterized_sql_execution() -> None:
    start_dt, end_excl = compute_half_open_date_bounds(
        date(2026, 10, 1), date(2026, 10, 31)
    )
    assert start_dt == datetime(2026, 10, 1, 0, 0, 0)
    assert end_excl == datetime(2026, 11, 1, 0, 0, 0)

    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_sqlite_compatible_schema(engine)
    repo = MySQLFeedbackRepository(engine=engine, safe_conn_summary="sqlite-test")

    payload = NewFeedbackPayload(
        data_origin=DataOrigin.TNU_REAL.value,
        origin_note="Khảo sát thực tế",
        unit_code="ICTU",
        source_code="SURVEY_PORTAL",
        masked_content="Thầy dạy dễ hiểu, sđt 0912345678 cần che.",
        processed_content="thầy dạy dễ hiểu",
        pii_types_detected=None,
        was_truncated=False,
        is_multi_aspect=False,
        processing_status=ProcessingStatus.DONE.value,
        processing_error=None,
        predicted_sentiment_label=0,
        predicted_sentiment_text="Negative",
        confidence_score=0.77,
        prob_negative=0.77,
        prob_neutral=0.13,
        prob_positive=0.10,
        predicted_topic_code="LECTURER",
        topic_assignment_source="MANUAL",
        model_version="PhoBERT-v1",
        pipeline_version="v1",
        inference_time_ms=30.0,
        review_status=ReviewStatus.NEEDS_REVIEW.value,
        review_reason="Kiểm thử",
        feedback_date=datetime(2026, 10, 31, 23, 59, 0),  # Sát biên cuối ngày 31/10!
        idempotency_key="idem-sql-001",
        source_record_id="SRC-SQL-001",
    )

    r1 = repo.save_feedback(payload)
    assert r1.saved is True
    # Lần lưu thứ hai cùng idempotency_key phải bị chặn!
    r2 = repo.save_feedback(payload)
    assert r2.saved is False
    assert r2.was_idempotent_duplicate is True

    flt = FilterCriteria(
        data_origins=[DataOrigin.TNU_REAL.value],
        date_from=date(2026, 10, 1),
        date_to=date(2026, 10, 31),
        unit_codes=["ICTU"],
        label_policy=LabelPolicy.VERIFIED_FIRST,
    )
    paged = repo.list_paginated(flt, page=1, page_size=15, only_negative_for_review=True)
    assert paged.total_count == 1
    # SĐT chưa che trong payload vẫn được MySQLFeedbackRepository tự động che trước khi lưu/đọc!
    assert "0912345678" not in paged.items[0].masked_content
    assert "<PHONE>" in paged.items[0].masked_content


def test_mysql_repo_distinguishes_not_null_and_fk_from_idempotency_key() -> None:
    """Bước 1.1: Kiểm thử lỗi NOT NULL, FOREIGN KEY và lỗi trùng idempotency_key thực sự."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_sqlite_compatible_schema(engine)
    repo = MySQLFeedbackRepository(engine=engine, safe_conn_summary="sqlite-constraint-test")

    base_kwargs = dict(
        origin_note="Test ràng buộc",
        masked_content="Phản hồi kiểm thử ràng buộc CSDL.",
        processed_content="phản_hồi kiểm_thử",
        pii_types_detected=None,
        was_truncated=False,
        is_multi_aspect=False,
        processing_status=ProcessingStatus.DONE.value,
        processing_error=None,
        predicted_sentiment_label=2,
        predicted_sentiment_text="Positive",
        confidence_score=0.92,
        prob_negative=0.03,
        prob_neutral=0.05,
        prob_positive=0.92,
        predicted_topic_code="LECTURER",
        topic_assignment_source="MANUAL",
        model_version="PhoBERT-v2",
        pipeline_version="v1",
        inference_time_ms=12.0,
        review_status=ReviewStatus.UNREVIEWED.value,
        review_reason=None,
        feedback_date=datetime(2026, 10, 20, 9, 0, 0),
    )

    # 1. Lỗi NOT NULL (data_origin=None) -> saved=False, was_idempotent_duplicate=False
    not_null_payload = NewFeedbackPayload(
        data_origin=None,  # type: ignore[arg-type]
        unit_code="ICTU",
        source_code="SURVEY_PORTAL",
        idempotency_key="idem-notnull-001",
        **base_kwargs,
    )
    res_not_null = repo.save_feedback(not_null_payload)
    assert res_not_null.saved is False
    assert res_not_null.was_idempotent_duplicate is False
    assert res_not_null.was_source_duplicate is False

    # 2. Lỗi FOREIGN KEY (unit_code không tồn tại trong bảng units) -> saved=False, was_idempotent_duplicate=False
    fk_payload = NewFeedbackPayload(
        data_origin=DataOrigin.TNU_REAL.value,
        unit_code="UNKNOWN_UNIT_FK",
        source_code="SURVEY_PORTAL",
        idempotency_key="idem-fk-001",
        **base_kwargs,
    )
    res_fk = repo.save_feedback(fk_payload)
    assert res_fk.saved is False
    assert res_fk.was_idempotent_duplicate is False
    assert res_fk.was_source_duplicate is False

    # 3. Lỗi trùng idempotency_key thực sự (kể cả khi nhảy vào nhánh IntegrityError code 1062 / UNIQUE)
    valid_payload = NewFeedbackPayload(
        data_origin=DataOrigin.TNU_REAL.value,
        unit_code="ICTU",
        source_code="SURVEY_PORTAL",
        idempotency_key="idem-real-dup-001",
        **base_kwargs,
    )
    res_ok = repo.save_feedback(valid_payload)
    assert res_ok.saved is True

    res_dup = repo.save_feedback(valid_payload)
    assert res_dup.saved is False
    assert res_dup.was_idempotent_duplicate is True

    # Mô phỏng MySQL IntegrityError (1062, "Duplicate entry 'idem-1062' for key 'feedbacks.idempotency_key'")
    class _FakeMySQLDupOrig(Exception):
        def __init__(self) -> None:
            super().__init__(1062, "Duplicate entry 'idem-1062' for key 'feedbacks.idempotency_key'")
            self.args = (1062, "Duplicate entry 'idem-1062' for key 'feedbacks.idempotency_key'")

    class _RaceEngine:
        def begin(self):
            raise IntegrityError("INSERT INTO feedbacks (idempotency_key) ...", {}, _FakeMySQLDupOrig())

    race_repo = MySQLFeedbackRepository(engine=_RaceEngine(), safe_conn_summary="mysql-1062")  # type: ignore[arg-type]
    res_1062 = race_repo.save_feedback(valid_payload)
    assert res_1062.saved is False
    assert res_1062.was_idempotent_duplicate is True
