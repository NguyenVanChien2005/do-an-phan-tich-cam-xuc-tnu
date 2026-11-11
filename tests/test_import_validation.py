"""Kiểm thử Import CSV/XLSX: sai cột, sai ngày, chặn công thức Excel, trùng bản ghi nguồn, lỗi DB và tổng đối soát."""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl

from tnu_sentiment.config import load_app_config
from tnu_sentiment.domain.models import ComponentHealth, SaveRecordResult, SentimentPrediction
from tnu_sentiment.models.sentiment_adapter import BaseSentimentAdapter
from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor
from tnu_sentiment.repositories.base import NewFeedbackPayload
from tnu_sentiment.repositories.demo_repo import DemoFeedbackRepository
from tnu_sentiment.services.import_service import ImportService, parse_feedback_datetime


class _StubAdapter(BaseSentimentAdapter):
    @property
    def is_ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return "stub-v1"

    def health(self) -> ComponentHealth:
        return ComponentHealth(True, "Sẵn sàng", "stub")

    def predict(self, processed_text: str) -> SentimentPrediction:
        return SentimentPrediction(
            label_id=0,
            label_code="Negative",
            label_vi="Tiêu cực",
            confidence=0.88,
            probabilities={0: 0.88, 1: 0.06, 2: 0.06},
            model_version="stub-v1",
            was_truncated=False,
            token_count=8,
            max_length=256,
            inference_time_ms=5.0,
            measurement_scope="stub",
        )


class _FailingDBRepo(DemoFeedbackRepository):
    def save_feedback(self, payload: NewFeedbackPayload) -> SaveRecordResult:
        if "gây lỗi db" in payload.masked_content.lower():
            raise RuntimeError("Simulated MySQL write failure")
        return super().save_feedback(payload)


def test_import_missing_columns_reported_clearly(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = DemoFeedbackRepository(tmp_path / "empty.json")
    svc = ImportService(cfg, _StubAdapter(), VietnamesePreprocessor())

    bad_csv = "noi_dung,don_vi\nXin chào,ICTU\n".encode("utf-8")
    headers, rows = svc.read_tabular_bytes("bad.csv", bad_csv)
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert len(schema_errs) == 1
    assert "Thiếu cột bắt buộc" in schema_errs[0]
    assert previews == []


def test_import_summary_counts_are_disjoint_and_sum_to_total_rows(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = _FailingDBRepo(tmp_path / "empty.json")
    svc = ImportService(cfg, _StubAdapter(), VietnamesePreprocessor())

    # 5 dòng kiểm thử đủ các nhánh:
    # 1. Hợp lệ + chứa PII -> che PII & lưu thành công (saved=1)
    # 2. Trùng source_record_id với dòng 1 -> bỏ qua do trùng (duplicate_skipped=1)
    # 3. Sai ngày tháng -> lỗi đầu vào (validation_error=1)
    # 4. Sai mã đơn vị -> lỗi đầu vào (validation_error=2)
    # 5. Lỗi ghi DB mô phỏng -> lỗi ghi dữ liệu (db_error=1)
    csv_content = (
        "sentence,unit_code,source_name,feedback_date,source_record_id\n"
        "\"Em sđt 0987654321 phản ánh máy chiếu hỏng.\",ICTU,QA_BOX,2026-10-20 08:00:00,SRC-100\n"
        "\"Bản ghi trùng mã nguồn.\",ICTU,QA_BOX,2026-10-20 09:00:00,SRC-100\n"
        "\"Phản hồi sai ngày tháng.\",TNUT,QA_BOX,2026-99-99,SRC-101\n"
        "\"Phản hồi sai mã đơn vị.\",DON_VI_LA,QA_BOX,2026-10-21,SRC-102\n"
        "\"Phản hồi gây lỗi db khi ghi.\",ICTU,QA_BOX,2026-10-22,SRC-103\n"
    ).encode("utf-8")

    headers, rows = svc.read_tabular_bytes("batch.csv", csv_content)
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert schema_errs == []
    assert len(previews) == 5
    # PII đã được che ngay từ bước preview!
    assert "0987654321" not in previews[0].masked_sentence
    assert "<PHONE>" in previews[0].masked_sentence

    summary = svc.execute_import(
        filename="batch.csv",
        declared_origin="DEMO",
        origin_note="Test import",
        previews=previews,
        repository=repo,
    )
    assert summary.total_rows_read == 5
    assert summary.saved_count == 1
    assert summary.duplicate_skipped_count == 1
    assert summary.validation_error_count == 2
    assert summary.inference_error_count == 0
    assert summary.db_error_count == 1
    assert summary.unprocessed_count == 0
    assert summary.accounted_rows == summary.total_rows_read


def test_xlsx_formula_cells_are_blocked(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = DemoFeedbackRepository(tmp_path / "empty.json")
    svc = ImportService(cfg, _StubAdapter(), VietnamesePreprocessor())

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["sentence", "unit_code", "source_name", "feedback_date"])
    ws.append(["=HYPERLINK(\"http://evil.example\",\"click\")", "ICTU", "QA_BOX", "2026-10-20"])
    buf = io.BytesIO()
    wb.save(buf)
    wb.close()

    headers, rows = svc.read_tabular_bytes("formula.xlsx", buf.getvalue())
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert schema_errs == []
    assert len(previews) == 1
    assert previews[0].is_valid is False
    assert any("công thức" in e.lower() for e in previews[0].errors)


def test_parse_feedback_datetime_converts_utc_to_vietnam_timezone() -> None:
    """Bước 1.3: Kiểm tra chuyển đổi múi giờ sang Asia/Ho_Chi_Minh trước khi bỏ tzinfo."""
    from datetime import datetime

    dt_utc = parse_feedback_datetime("2026-11-09T20:30:00+00:00")
    assert dt_utc == datetime(2026, 11, 10, 3, 30, 0)
    assert dt_utc.tzinfo is None

    dt_z = parse_feedback_datetime("2026-11-09T20:30:00Z")
    assert dt_z == datetime(2026, 11, 10, 3, 30, 0)

    dt_naive = parse_feedback_datetime("2026-11-10 08:15:00")
    assert dt_naive == datetime(2026, 11, 10, 8, 15, 0)


def test_crawler_contract_and_idempotent_pipeline_end_to_end(tmp_path: Path) -> None:
    """Bước 4.1 & 4.2: Kiểm thử crawler.py xuất khớp 100% hợp đồng Batch Import (FACEBOOK, YOUTUBE, NEWS_COMMENT,
    source_record_id, source_url, post_id, parent_comment_id, collected_at, topic_code='UNCLASSIFIED', idempotency_key)
    và chạy lại quy trình Lấy -> Chuẩn hóa -> Phân tích -> Lưu -> Hiển thị nhiều lần không sinh bản ghi trùng.
    """
    from crawler import CSV_COLUMNS, comments_to_csv_bytes, normalize_crawled_comment
    from tnu_sentiment.domain.models import FilterCriteria

    assert set(
        [
            "sentence",
            "unit_code",
            "source_name",
            "feedback_date",
            "source_record_id",
            "source_url",
            "post_id",
            "parent_comment_id",
            "collected_at",
            "topic_code",
            "idempotency_key",
        ]
    ).issubset(set(CSV_COLUMNS))

    raw_crawled_items = [
        {
            "message": "Em mã sv DTC202600999 thấy phòng máy C5 trường ICTU mới nâng cấp rất mượt!",
            "unit_code": "ICTU",
            "platform": "facebook",
            "created_time": "2026-11-09T20:30:00+00:00",
            "id": "FB_CMT_001",
            "post_id": "FB_POST_100",
            "source": "https://facebook.com/ictu/posts/100",
            "parent": {"id": "FB_PARENT_000"},
            "collected_at": "2026-11-09T21:00:00Z",
        },
        {
            "content": "Video giới thiệu chương trình đào tạo của TNUT rất chi tiết và dễ hiểu.",
            "unit_code": "TNUT",
            "source_name": "YOUTUBE",
            "created_at": "2026-11-10 09:00:00",
            "comment_id": "YT_CMT_002",
            "post_id": "YT_VID_200",
            "source_url": "https://youtube.com/watch?v=200",
            "collected_at": "2026-11-10 09:05:00",
        },
        {
            # Bản ghi không có source_record_id -> idempotency_key tính từ (content_hash, unit_code, feedback_date)
            "sentence": "Bản tin học vụ học kỳ này thông báo lịch đăng ký tín chỉ rõ ràng.",
            "unit_code": "TUEBA",
            "source_name": "NEWS_COMMENT",
            "feedback_date": "2026-11-10 10:30:00",
            "source_url": "https://tnu.edu.vn/news/300",
        },
    ]

    normalized_first = normalize_crawled_comment(raw_crawled_items[0])
    assert "<STUDENT_ID>" in normalized_first["sentence"]
    assert normalized_first["source_name"] == "FACEBOOK"
    assert normalized_first["topic_code"] == "UNCLASSIFIED"
    assert normalized_first["feedback_date"] == "2026-11-10 03:30:00"
    assert normalized_first["idempotency_key"].startswith("idem-src-")

    csv_bytes = comments_to_csv_bytes(raw_crawled_items)

    cfg = load_app_config()
    repo = DemoFeedbackRepository(tmp_path / "empty.json")
    svc = ImportService(cfg, _StubAdapter(), VietnamesePreprocessor())

    # Lần chạy 1: Lấy -> Chuẩn hóa -> Phân tích -> Lưu -> Hiển thị
    headers, rows = svc.read_tabular_bytes("crawled_batch.csv", csv_bytes)
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert schema_errs == []
    assert len(previews) == 3
    assert all(p.is_valid for p in previews)

    sum1 = svc.execute_import(
        filename="crawled_batch.csv",
        declared_origin="TNU_REAL",
        origin_note="Thu thập từ Crawler",
        previews=previews,
        repository=repo,
    )
    assert sum1.saved_count == 3
    assert sum1.duplicate_skipped_count == 0

    flt = FilterCriteria(select_all_origins=True)
    paged1 = repo.list_paginated(flt, page=1, page_size=15)
    assert paged1.total_count == 3
    by_unit = {r.unit_code: r for r in paged1.items}
    assert by_unit["ICTU"].source_code == "FACEBOOK"
    assert by_unit["ICTU"].post_id == "FB_POST_100"
    assert by_unit["ICTU"].parent_comment_id == "FB_PARENT_000"
    assert by_unit["ICTU"].source_url == "https://facebook.com/ictu/posts/100"
    assert by_unit["ICTU"].predicted_topic_code == "UNCLASSIFIED"

    # Lần chạy 2 trên cùng dữ liệu Crawler -> 0 bản ghi lưu mới, toàn bộ 3 dòng bị chặn trùng
    sum2 = svc.execute_import(
        filename="crawled_batch.csv",
        declared_origin="TNU_REAL",
        origin_note="Chạy lại lần 2",
        previews=previews,
        repository=repo,
    )
    assert sum2.saved_count == 0
    assert sum2.duplicate_skipped_count == 3
    assert repo.list_paginated(flt, page=1, page_size=15).total_count == 3


def test_batched_inference_batch_size_32_and_import_batches_history(tmp_path: Path) -> None:
    """Bước 5: Kiểm thử Batch Import gom lô suy luận theo batch_size = 32 và lưu lịch sử vào import_batches."""
    from typing import List

    batch_call_sizes: List[int] = []

    class _BatchSpyAdapter(_StubAdapter):
        def predict_batch(
            self, processed_texts: List[str], batch_size: int = 32
        ) -> List[SentimentPrediction]:
            safe_b = max(1, int(batch_size))
            out: List[SentimentPrediction] = []
            for start in range(0, len(processed_texts), safe_b):
                chunk = processed_texts[start : start + safe_b]
                batch_call_sizes.append(len(chunk))
                out.extend(self.predict(t) for t in chunk)
            return out

    cfg = load_app_config()
    repo = DemoFeedbackRepository(tmp_path / "empty.json")
    svc = ImportService(cfg, _BatchSpyAdapter(), VietnamesePreprocessor())

    lines = ["sentence,unit_code,source_name,feedback_date,source_record_id"]
    for i in range(1, 71):
        lines.append(
            f"\"Phản hồi kiểm thử lô số {i} về chất lượng phòng máy.\",ICTU,SURVEY_PORTAL,2026-11-10 08:00:00,BATCH-REC-{i:03d}"
        )
    csv_bytes = "\n".join(lines).encode("utf-8")

    headers, rows = svc.read_tabular_bytes("batch_70.csv", csv_bytes)
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert schema_errs == []
    assert len(previews) == 70

    summary = svc.execute_import(
        filename="batch_70.csv",
        declared_origin="TNU_REAL",
        origin_note="Kiểm thử batched inference 70 dòng",
        previews=previews,
        repository=repo,
        batch_size=32,
    )
    assert summary.saved_count == 70
    # 70 dòng chia theo batch_size=32 -> 3 lô có kích thước [32, 32, 6]
    assert batch_call_sizes == [32, 32, 6]

    batches_history = repo.list_import_batches()
    assert len(batches_history) == 1
    assert batches_history[0]["batch_id"] == summary.batch_id
    assert batches_history[0]["filename"] == "batch_70.csv"
    assert batches_history[0]["total_rows"] == 70
    assert batches_history[0]["saved_count"] == 70
