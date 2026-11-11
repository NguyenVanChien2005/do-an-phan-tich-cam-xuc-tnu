"""Repository MySQL sử dụng SQLAlchemy Core + truy vấn tham số hóa, phân trang tại DB và Transaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from tnu_sentiment.constants import (
    INITIAL_SOURCES,
    INITIAL_TOPICS,
    INITIAL_UNITS,
    LabelPolicy,
    ReviewStatus,
    SENTIMENT_ID_TO_CODE,
    USER_STATUS_LABELS,
)
from tnu_sentiment.domain.models import (
    CatalogItem,
    ComponentHealth,
    FeedbackRecord,
    FilterCriteria,
    ImportExecutionSummary,
    PaginatedRecords,
    ReviewLogEntry,
    ReviewUpdateRequest,
    SaveRecordResult,
)
from tnu_sentiment.nlp.pii_masker import compute_content_hash, mask_pii
from tnu_sentiment.repositories.base import (
    FeedbackRepository,
    NewFeedbackPayload,
    compute_half_open_date_bounds,
)


class MySQLFeedbackRepository(FeedbackRepository):
    """Truy cập CSDL MySQL thực tế.
    - Nhận Engine dùng chung từ cache_resource nhưng luôn mở kết nối/giao dịch ngắn hạn riêng cho từng thao tác.
    - Toàn bộ bộ lọc và phân trang thực thi tại MySQL bằng tham số hóa (:param).
    - Không tự động xóa hay DROP bảng dữ liệu thật.
    """

    def __init__(
        self,
        engine: Optional[Engine],
        safe_conn_summary: str,
        tz_name: str = "Asia/Ho_Chi_Minh",
        init_error: Optional[str] = None,
    ) -> None:
        self._engine = engine
        self._safe_summary = safe_conn_summary
        self._tz = ZoneInfo(tz_name)
        self._init_error = init_error
        self._import_batches_cache: List[Dict[str, object]] = []

    @property
    def repository_kind(self) -> str:
        return "MYSQL"

    def _now_local(self) -> datetime:
        return datetime.now(self._tz).replace(tzinfo=None)

    def health(self) -> ComponentHealth:
        if self._engine is None:
            return ComponentHealth(
                ready=False,
                user_badge=USER_STATUS_LABELS["db_disconnected"],
                technical_detail=self._init_error or f"Chưa kết nối được MySQL ({self._safe_summary}).",
            )
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                tbl_check = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema = DATABASE() AND table_name = 'feedbacks'"
                    )
                ).scalar_one()
                if int(tbl_check) == 0:
                    return ComponentHealth(
                        ready=False,
                        user_badge=USER_STATUS_LABELS["db_disconnected"],
                        technical_detail=(
                            f"Đã kết nối MySQL ({self._safe_summary}) nhưng chưa tìm thấy bảng 'feedbacks'. "
                            "Vui lòng chạy tệp sql/schema_init.sql để khởi tạo cấu trúc bảng an toàn."
                        ),
                    )
            return ComponentHealth(
                ready=True,
                user_badge=USER_STATUS_LABELS["db_connected"],
                technical_detail=f"Kết nối MySQL hoạt động bình thường ({self._safe_summary}).",
            )
        except Exception as exc:
            return ComponentHealth(
                ready=False,
                user_badge=USER_STATUS_LABELS["db_disconnected"],
                technical_detail=f"Lỗi kết nối MySQL ({self._safe_summary}): {exc}",
            )

    def list_units(self) -> List[CatalogItem]:
        if self._engine is None:
            return [CatalogItem(code=c, name=n) for c, n in INITIAL_UNITS]
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT unit_code, unit_name FROM units "
                        "WHERE is_active = 1 ORDER BY unit_id ASC"
                    )
                ).mappings().all()
            if rows:
                return [CatalogItem(code=str(r["unit_code"]), name=str(r["unit_name"])) for r in rows]
        except SQLAlchemyError:
            pass
        return [CatalogItem(code=c, name=n) for c, n in INITIAL_UNITS]

    def list_sources(self) -> List[CatalogItem]:
        if self._engine is None:
            return [CatalogItem(code=c, name=n) for c, n in INITIAL_SOURCES]
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT source_code, source_name FROM sources "
                        "WHERE is_active = 1 ORDER BY source_id ASC"
                    )
                ).mappings().all()
            if rows:
                return [
                    CatalogItem(code=str(r["source_code"]), name=str(r["source_name"]))
                    for r in rows
                ]
        except SQLAlchemyError:
            pass
        return [CatalogItem(code=c, name=n) for c, n in INITIAL_SOURCES]

    def list_topics(self) -> List[CatalogItem]:
        if self._engine is None:
            return [CatalogItem(code=c, name=n) for c, n, _ in INITIAL_TOPICS]
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT topic_code, topic_name FROM topics "
                        "WHERE is_active = 1 ORDER BY topic_id ASC"
                    )
                ).mappings().all()
            if rows:
                return [
                    CatalogItem(code=str(r["topic_code"]), name=str(r["topic_name"]))
                    for r in rows
                ]
        except SQLAlchemyError:
            pass
        return [CatalogItem(code=c, name=n) for c, n, _ in INITIAL_TOPICS]

    @staticmethod
    def _effective_sentiment_sql(policy: LabelPolicy) -> str:
        if policy == LabelPolicy.VERIFIED_ONLY:
            return (
                "CASE WHEN f.processing_status = 'DONE' AND f.review_status = 'VERIFIED' "
                "THEN f.reviewed_sentiment_label ELSE NULL END"
            )
        return (
            "CASE WHEN f.processing_status != 'DONE' THEN NULL "
            "WHEN f.review_status = 'VERIFIED' AND f.reviewed_sentiment_label IS NOT NULL "
            "THEN f.reviewed_sentiment_label ELSE f.predicted_sentiment_label END"
        )

    @staticmethod
    def _effective_topic_sql(policy: LabelPolicy) -> str:
        if policy == LabelPolicy.VERIFIED_ONLY:
            return (
                "CASE WHEN f.review_status = 'VERIFIED' AND f.reviewed_topic_code IS NOT NULL "
                "THEN f.reviewed_topic_code ELSE 'UNCLASSIFIED' END"
            )
        return (
            "CASE WHEN f.review_status = 'VERIFIED' AND f.reviewed_topic_code IS NOT NULL "
            "THEN f.reviewed_topic_code ELSE f.predicted_topic_code END"
        )

    def _build_where_clause(
        self,
        filters: FilterCriteria,
        only_negative_for_review: bool = False,
    ) -> Tuple[str, Dict[str, Any], List[Any]]:
        clauses: List[str] = ["1=1"]
        params: Dict[str, Any] = {}
        expanding_binds: List[Any] = []

        if not filters.select_all_origins and filters.data_origins:
            clauses.append("f.data_origin IN :origins")
            params["origins"] = list(filters.data_origins)
            expanding_binds.append(bindparam("origins", expanding=True))

        start_dt, end_excl = compute_half_open_date_bounds(
            filters.date_from, filters.date_to
        )
        if start_dt is not None:
            clauses.append("f.feedback_date >= :start_dt")
            params["start_dt"] = start_dt
        if end_excl is not None:
            clauses.append("f.feedback_date < :end_excl")
            params["end_excl"] = end_excl

        if filters.unit_codes:
            clauses.append("f.unit_code IN :unit_codes")
            params["unit_codes"] = list(filters.unit_codes)
            expanding_binds.append(bindparam("unit_codes", expanding=True))

        if filters.source_codes:
            clauses.append("f.source_code IN :source_codes")
            params["source_codes"] = list(filters.source_codes)
            expanding_binds.append(bindparam("source_codes", expanding=True))

        if filters.topic_codes:
            eff_topic = self._effective_topic_sql(filters.label_policy)
            clauses.append(f"({eff_topic}) IN :topic_codes")
            params["topic_codes"] = list(filters.topic_codes)
            expanding_binds.append(bindparam("topic_codes", expanding=True))

        if filters.processing_statuses:
            clauses.append("f.processing_status IN :proc_statuses")
            params["proc_statuses"] = list(filters.processing_statuses)
            expanding_binds.append(bindparam("proc_statuses", expanding=True))

        if filters.review_statuses:
            clauses.append("f.review_status IN :rev_statuses")
            params["rev_statuses"] = list(filters.review_statuses)
            expanding_binds.append(bindparam("rev_statuses", expanding=True))

        eff_sent = self._effective_sentiment_sql(filters.label_policy)
        if filters.sentiment_filter:
            clauses.append(f"({eff_sent}) IN :sent_labels")
            params["sent_labels"] = [int(x) for x in filters.sentiment_filter]
            expanding_binds.append(bindparam("sent_labels", expanding=True))

        if only_negative_for_review:
            clauses.append(f"({eff_sent}) = 0")

        if filters.search_query.strip():
            clauses.append("f.masked_content LIKE :search_like")
            escaped = (
                filters.search_query.strip()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            params["search_like"] = f"%{escaped}%"

        return " AND ".join(clauses), params, expanding_binds

    @staticmethod
    def _row_to_record(row: Any) -> FeedbackRecord:
        safe_masked, _ = mask_pii(str(row["masked_content"] or ""))
        return FeedbackRecord(
            feedback_id=int(row["feedback_id"]),
            data_origin=str(row["data_origin"]),
            origin_note=row["origin_note"],
            unit_code=str(row["unit_code"]),
            unit_name=str(row["unit_name"] or row["unit_code"]),
            source_code=str(row["source_code"]),
            source_name=str(row["source_name"] or row["source_code"]),
            masked_content=safe_masked,
            processed_content=row.get("processed_content"),
            pii_types_detected=row.get("pii_types_detected"),
            was_truncated=bool(row.get("was_truncated", 0)),
            is_multi_aspect=bool(row.get("is_multi_aspect", 0)),
            is_duplicate_candidate=bool(row.get("is_duplicate_candidate", 0)),
            processing_status=str(row["processing_status"]),
            processing_error=row.get("processing_error"),
            predicted_sentiment_label=(
                int(row["predicted_sentiment_label"])
                if row["predicted_sentiment_label"] is not None
                else None
            ),
            predicted_sentiment_text=row.get("predicted_sentiment_text"),
            confidence_score=(
                float(row["confidence_score"])
                if row["confidence_score"] is not None
                else None
            ),
            prob_negative=(
                float(row["prob_negative"]) if row.get("prob_negative") is not None else None
            ),
            prob_neutral=(
                float(row["prob_neutral"]) if row.get("prob_neutral") is not None else None
            ),
            prob_positive=(
                float(row["prob_positive"]) if row.get("prob_positive") is not None else None
            ),
            predicted_topic_code=str(row["predicted_topic_code"]),
            predicted_topic_name=str(
                row.get("predicted_topic_name") or row["predicted_topic_code"]
            ),
            topic_assignment_source=str(row.get("topic_assignment_source") or "MANUAL"),
            model_version=row.get("model_version"),
            pipeline_version=row.get("pipeline_version"),
            inference_time_ms=(
                float(row["inference_time_ms"])
                if row.get("inference_time_ms") is not None
                else None
            ),
            review_status=str(row["review_status"]),
            review_reason=row.get("review_reason"),
            reviewed_sentiment_label=(
                int(row["reviewed_sentiment_label"])
                if row["reviewed_sentiment_label"] is not None
                else None
            ),
            reviewed_topic_code=row.get("reviewed_topic_code"),
            reviewed_topic_name=row.get("reviewed_topic_name"),
            reviewed_by=row.get("reviewed_by"),
            reviewed_at=row.get("reviewed_at"),
            review_note=row.get("review_note"),
            record_version=int(row.get("record_version", 1)),
            feedback_date=row["feedback_date"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            idempotency_key=row.get("idempotency_key"),
            source_record_id=row.get("source_record_id"),
            source_url=row.get("source_url"),
            post_id=row.get("post_id"),
            parent_comment_id=row.get("parent_comment_id"),
            collected_at=row.get("collected_at"),
            content_hash=str(row.get("content_hash") or ""),
        )

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("Chưa kết nối cơ sở dữ liệu MySQL.")
        return self._engine

    def list_matching_for_stats(self, filters: FilterCriteria) -> List[FeedbackRecord]:
        """Chỉ truy vấn các cột siêu dữ liệu nhẹ đã lọc tại DB để tính thống kê (không tải cột văn bản dài)."""
        eng = self._require_engine()
        where_sql, params, binds = self._build_where_clause(filters)
        sql = text(
            f"""
            SELECT
                f.feedback_id, f.data_origin, NULL AS origin_note,
                f.unit_code, u.unit_name,
                f.source_code, s.source_name,
                '' AS masked_content,
                f.processing_status, f.predicted_sentiment_label, f.predicted_sentiment_text,
                f.confidence_score, f.predicted_topic_code, tp.topic_name AS predicted_topic_name,
                f.topic_assignment_source, f.model_version,
                f.review_status, f.reviewed_sentiment_label,
                f.reviewed_topic_code, tr.topic_name AS reviewed_topic_name,
                f.record_version, f.feedback_date, f.created_at, f.updated_at
            FROM feedbacks f
            LEFT JOIN units u ON f.unit_code = u.unit_code
            LEFT JOIN sources s ON f.source_code = s.source_code
            LEFT JOIN topics tp ON f.predicted_topic_code = tp.topic_code
            LEFT JOIN topics tr ON f.reviewed_topic_code = tr.topic_code
            WHERE {where_sql}
            """
        )
        if binds:
            sql = sql.bindparams(*binds)
        with eng.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [self._row_to_record(r) for r in rows]

    def list_paginated(
        self,
        filters: FilterCriteria,
        page: int = 1,
        page_size: int = 15,
        sort_by: str = "newest",
        only_negative_for_review: bool = False,
    ) -> PaginatedRecords:
        eng = self._require_engine()
        where_sql, params, binds = self._build_where_clause(
            filters, only_negative_for_review=only_negative_for_review
        )

        order_map = {
            "newest": "f.feedback_date DESC, f.feedback_id DESC",
            "oldest": "f.feedback_date ASC, f.feedback_id ASC",
            "confidence_asc": "CASE WHEN f.confidence_score IS NULL THEN 1 ELSE 0 END ASC, f.confidence_score ASC, f.feedback_date DESC",
            "confidence_desc": "CASE WHEN f.confidence_score IS NULL THEN 1 ELSE 0 END ASC, f.confidence_score DESC, f.feedback_date DESC",
        }
        order_sql = order_map.get(sort_by, order_map["newest"])

        count_stmt = text(f"SELECT COUNT(*) FROM feedbacks f WHERE {where_sql}")
        if binds:
            count_stmt = count_stmt.bindparams(*binds)

        with eng.connect() as conn:
            total = int(conn.execute(count_stmt, params).scalar_one())
            safe_size = max(1, int(page_size))
            max_page = max(1, (total + safe_size - 1) // safe_size)
            safe_page = min(max(1, int(page)), max_page)
            offset = (safe_page - 1) * safe_size

            page_params = dict(params)
            page_params["limit_n"] = safe_size
            page_params["offset_n"] = offset

            data_stmt = text(
                f"""
                SELECT
                    f.*,
                    u.unit_name,
                    s.source_name,
                    tp.topic_name AS predicted_topic_name,
                    tr.topic_name AS reviewed_topic_name
                FROM feedbacks f
                LEFT JOIN units u ON f.unit_code = u.unit_code
                LEFT JOIN sources s ON f.source_code = s.source_code
                LEFT JOIN topics tp ON f.predicted_topic_code = tp.topic_code
                LEFT JOIN topics tr ON f.reviewed_topic_code = tr.topic_code
                WHERE {where_sql}
                ORDER BY {order_sql}
                LIMIT :limit_n OFFSET :offset_n
                """
            )
            if binds:
                data_stmt = data_stmt.bindparams(*binds)
            rows = conn.execute(data_stmt, page_params).mappings().all()

        return PaginatedRecords(
            items=[self._row_to_record(r) for r in rows],
            total_count=total,
            page=safe_page,
            page_size=safe_size,
        )

    def get_by_id(self, feedback_id: int) -> Optional[FeedbackRecord]:
        eng = self._require_engine()
        stmt = text(
            """
            SELECT
                f.*,
                u.unit_name,
                s.source_name,
                tp.topic_name AS predicted_topic_name,
                tr.topic_name AS reviewed_topic_name
            FROM feedbacks f
            LEFT JOIN units u ON f.unit_code = u.unit_code
            LEFT JOIN sources s ON f.source_code = s.source_code
            LEFT JOIN topics tp ON f.predicted_topic_code = tp.topic_code
            LEFT JOIN topics tr ON f.reviewed_topic_code = tr.topic_code
            WHERE f.feedback_id = :fid
            """
        )
        with eng.connect() as conn:
            row = conn.execute(stmt, {"fid": int(feedback_id)}).mappings().first()
        return self._row_to_record(row) if row else None

    def check_source_record_exists(
        self, source_code: str, source_record_id: str
    ) -> bool:
        if not source_record_id:
            return False
        eng = self._require_engine()
        stmt = text(
            "SELECT feedback_id FROM feedbacks "
            "WHERE source_code = :scode AND source_record_id = :srid LIMIT 1"
        )
        with eng.connect() as conn:
            hit = conn.execute(
                stmt, {"scode": source_code, "srid": source_record_id}
            ).first()
        return hit is not None

    def save_feedback(self, payload: NewFeedbackPayload) -> SaveRecordResult:
        eng = self._require_engine()
        safe_masked, detected_pii = mask_pii(payload.masked_content)
        c_hash = payload.content_hash or compute_content_hash(safe_masked)
        sent_text = (
            SENTIMENT_ID_TO_CODE.get(payload.predicted_sentiment_label)
            if payload.predicted_sentiment_label is not None
            else payload.predicted_sentiment_text
        )

        try:
            with eng.begin() as conn:
                if payload.idempotency_key:
                    idem_hit = conn.execute(
                        text(
                            "SELECT feedback_id FROM feedbacks WHERE idempotency_key = :ikey LIMIT 1"
                        ),
                        {"ikey": payload.idempotency_key},
                    ).first()
                    if idem_hit:
                        return SaveRecordResult(
                            saved=False,
                            was_idempotent_duplicate=True,
                            was_source_duplicate=False,
                            is_content_duplicate_candidate=False,
                            feedback_id=int(idem_hit[0]),
                            message=f"Thao tác đã được lưu trước đó (ID #{idem_hit[0]}), không tạo bản ghi lặp.",
                        )

                if payload.source_record_id:
                    src_hit = conn.execute(
                        text(
                            "SELECT feedback_id FROM feedbacks "
                            "WHERE source_code = :scode AND source_record_id = :srid LIMIT 1"
                        ),
                        {"scode": payload.source_code, "srid": payload.source_record_id},
                    ).first()
                    if src_hit:
                        return SaveRecordResult(
                            saved=False,
                            was_idempotent_duplicate=False,
                            was_source_duplicate=True,
                            is_content_duplicate_candidate=False,
                            feedback_id=int(src_hit[0]),
                            message=(
                                f"Bỏ qua bản ghi nguồn trùng source_record_id='{payload.source_record_id}' "
                                f"(ID #{src_hit[0]})."
                            ),
                        )

                hash_hit = conn.execute(
                    text("SELECT feedback_id FROM feedbacks WHERE content_hash = :ch LIMIT 1"),
                    {"ch": c_hash},
                ).first()
                dup_candidate = hash_hit is not None

                insert_stmt = text(
                    """
                    INSERT INTO feedbacks (
                        idempotency_key, source_record_id, content_hash, author_hash,
                        data_origin, origin_note, batch_id,
                        unit_code, source_code,
                        masked_content, processed_content, pii_types_detected,
                        was_truncated, is_multi_aspect, is_duplicate_candidate,
                        processing_status, processing_error,
                        predicted_sentiment_label, predicted_sentiment_text,
                        confidence_score, prob_negative, prob_neutral, prob_positive,
                        predicted_topic_code, topic_assignment_source,
                        model_version, pipeline_version, inference_time_ms,
                        llm_suggested_sentiment, llm_suggested_topic_code, llm_note,
                        review_status, review_reason, record_version, feedback_date
                    ) VALUES (
                        :idempotency_key, :source_record_id, :content_hash, :author_hash,
                        :data_origin, :origin_note, :batch_id,
                        :unit_code, :source_code,
                        :masked_content, :processed_content, :pii_types_detected,
                        :was_truncated, :is_multi_aspect, :is_duplicate_candidate,
                        :processing_status, :processing_error,
                        :predicted_sentiment_label, :predicted_sentiment_text,
                        :confidence_score, :prob_negative, :prob_neutral, :prob_positive,
                        :predicted_topic_code, :topic_assignment_source,
                        :model_version, :pipeline_version, :inference_time_ms,
                        :llm_suggested_sentiment, :llm_suggested_topic_code, :llm_note,
                        :review_status, :review_reason, 1, :feedback_date
                    )
                    """
                )
                res = conn.execute(
                    insert_stmt,
                    {
                        "idempotency_key": payload.idempotency_key,
                        "source_record_id": payload.source_record_id,
                        "content_hash": c_hash,
                        "author_hash": payload.author_hash,
                        "data_origin": payload.data_origin,
                        "origin_note": payload.origin_note,
                        "batch_id": payload.batch_id,
                        "unit_code": payload.unit_code,
                        "source_code": payload.source_code,
                        "masked_content": safe_masked,
                        "processed_content": payload.processed_content,
                        "pii_types_detected": (
                            ",".join(detected_pii)
                            if detected_pii
                            else payload.pii_types_detected
                        ),
                        "was_truncated": int(bool(payload.was_truncated)),
                        "is_multi_aspect": int(bool(payload.is_multi_aspect)),
                        "is_duplicate_candidate": int(dup_candidate),
                        "processing_status": payload.processing_status,
                        "processing_error": payload.processing_error,
                        "predicted_sentiment_label": payload.predicted_sentiment_label,
                        "predicted_sentiment_text": sent_text,
                        "confidence_score": payload.confidence_score,
                        "prob_negative": payload.prob_negative,
                        "prob_neutral": payload.prob_neutral,
                        "prob_positive": payload.prob_positive,
                        "predicted_topic_code": payload.predicted_topic_code,
                        "topic_assignment_source": payload.topic_assignment_source,
                        "model_version": payload.model_version,
                        "pipeline_version": payload.pipeline_version,
                        "inference_time_ms": payload.inference_time_ms,
                        "llm_suggested_sentiment": payload.llm_suggested_sentiment,
                        "llm_suggested_topic_code": payload.llm_suggested_topic_code,
                        "llm_note": payload.llm_note,
                        "review_status": payload.review_status,
                        "review_reason": payload.review_reason,
                        "feedback_date": payload.feedback_date,
                    },
                )
                new_id = int(res.lastrowid)
                return SaveRecordResult(
                    saved=True,
                    was_idempotent_duplicate=False,
                    was_source_duplicate=False,
                    is_content_duplicate_candidate=dup_candidate,
                    feedback_id=new_id,
                    message=f"Đã lưu bản ghi #{new_id} vào MySQL thành công.",
                )
        except IntegrityError as exc:
            orig = getattr(exc, "orig", None)
            orig_msg = str(orig if orig is not None else "").lower()
            orig_args = getattr(orig, "args", ()) or ()
            is_mysql_dup_code = len(orig_args) >= 1 and orig_args[0] == 1062
            is_unique_violation = (
                is_mysql_dup_code
                or "unique" in orig_msg
                or "duplicate entry" in orig_msg
            )

            if is_unique_violation and (
                "idempotency_key" in orig_msg or "content_hash" in orig_msg
            ):
                return SaveRecordResult(
                    saved=False,
                    was_idempotent_duplicate=True,
                    was_source_duplicate=False,
                    is_content_duplicate_candidate=False,
                    feedback_id=None,
                    message="Thao tác lưu trùng idempotency_key đã bị ràng buộc duy nhất chặn lại.",
                )
            if is_unique_violation and (
                "uq_source_record" in orig_msg or "source_record_id" in orig_msg
            ):
                return SaveRecordResult(
                    saved=False,
                    was_idempotent_duplicate=False,
                    was_source_duplicate=True,
                    is_content_duplicate_candidate=False,
                    feedback_id=None,
                    message=f"Trùng khóa nghiệp vụ (source_code, source_record_id='{payload.source_record_id}').",
                )
            return SaveRecordResult(
                saved=False,
                was_idempotent_duplicate=False,
                was_source_duplicate=False,
                is_content_duplicate_candidate=False,
                feedback_id=None,
                message=f"Lỗi ràng buộc CSDL (NOT NULL / FOREIGN KEY): {orig if orig is not None else exc}",
            )
        except SQLAlchemyError as exc:
            return SaveRecordResult(
                saved=False,
                was_idempotent_duplicate=False,
                was_source_duplicate=False,
                is_content_duplicate_candidate=False,
                feedback_id=None,
                message=f"Lỗi kết nối / thực thi CSDL: {exc}",
            )

    def apply_review_updates(
        self, updates: List[ReviewUpdateRequest]
    ) -> Tuple[int, List[str]]:
        eng = self._require_engine()
        applied = 0
        errors: List[str] = []
        now_dt = self._now_local()

        for upd in updates:
            reviewer = upd.reviewed_by.strip()
            if not reviewer:
                errors.append(f"Bản ghi ID #{upd.feedback_id}: thiếu tên người duyệt.")
                continue
            if (
                upd.new_review_status == ReviewStatus.VERIFIED.value
                and upd.reviewed_sentiment_label not in (0, 1, 2)
            ):
                errors.append(
                    f"Bản ghi ID #{upd.feedback_id}: cần chọn nhãn cảm xúc được duyệt (0/1/2) khi xác nhận VERIFIED."
                )
                continue

            safe_note, _ = mask_pii(upd.review_note or "")
            try:
                with eng.begin() as conn:
                    curr = conn.execute(
                        text(
                            "SELECT review_status, reviewed_sentiment_label, reviewed_topic_code, record_version "
                            "FROM feedbacks WHERE feedback_id = :fid FOR UPDATE"
                        ),
                        {"fid": int(upd.feedback_id)},
                    ).mappings().first()

                    if curr is None:
                        errors.append(f"Không tìm thấy bản ghi ID #{upd.feedback_id}.")
                        continue

                    curr_ver = int(curr["record_version"])
                    if curr_ver != int(upd.expected_version):
                        errors.append(
                            f"Xung đột cập nhật đồng thời tại ID #{upd.feedback_id} "
                            f"(CSDL v{curr_ver} != v{upd.expected_version})."
                        )
                        continue

                    new_ver = curr_ver + 1
                    # Tuyệt đối KHÔNG chạm vào các cột predicted_*
                    upd_res = conn.execute(
                        text(
                            """
                            UPDATE feedbacks
                            SET review_status = :rstatus,
                                reviewed_sentiment_label = :rsent,
                                reviewed_topic_code = :rtopic,
                                reviewed_by = :rby,
                                reviewed_at = :rat,
                                review_note = :rnote,
                                record_version = :nver
                            WHERE feedback_id = :fid AND record_version = :ever
                            """
                        ),
                        {
                            "rstatus": upd.new_review_status,
                            "rsent": upd.reviewed_sentiment_label,
                            "rtopic": upd.reviewed_topic_code,
                            "rby": reviewer,
                            "rat": now_dt,
                            "rnote": safe_note or None,
                            "nver": new_ver,
                            "fid": int(upd.feedback_id),
                            "ever": curr_ver,
                        },
                    )
                    if upd_res.rowcount != 1:
                        errors.append(f"Không thể cập nhật ID #{upd.feedback_id} do thay đổi phiên bản.")
                        continue

                    conn.execute(
                        text(
                            """
                            INSERT INTO feedback_review_logs (
                                feedback_id,
                                previous_review_status, new_review_status,
                                previous_sentiment_label, new_sentiment_label,
                                previous_topic_code, new_topic_code,
                                reviewed_by, is_authenticated_user, review_note,
                                record_version_before, record_version_after, changed_at
                            ) VALUES (
                                :fid,
                                :prev_status, :new_status,
                                :prev_sent, :new_sent,
                                :prev_topic, :new_topic,
                                :rby, 0, :rnote,
                                :vbefore, :vafter, :cat
                            )
                            """
                        ),
                        {
                            "fid": int(upd.feedback_id),
                            "prev_status": curr["review_status"],
                            "new_status": upd.new_review_status,
                            "prev_sent": curr["reviewed_sentiment_label"],
                            "new_sent": upd.reviewed_sentiment_label,
                            "prev_topic": curr["reviewed_topic_code"],
                            "new_topic": upd.reviewed_topic_code,
                            "rby": reviewer,
                            "rnote": safe_note or None,
                            "vbefore": curr_ver,
                            "vafter": new_ver,
                            "cat": now_dt,
                        },
                    )
                    applied += 1
            except SQLAlchemyError as exc:
                errors.append(f"Lỗi CSDL khi lưu kiểm duyệt ID #{upd.feedback_id}: {exc}")

        return applied, errors

    def list_review_logs(
        self, feedback_id: Optional[int] = None, limit: int = 50
    ) -> List[ReviewLogEntry]:
        eng = self._require_engine()
        if feedback_id is not None:
            stmt = text(
                "SELECT * FROM feedback_review_logs WHERE feedback_id = :fid "
                "ORDER BY changed_at DESC, log_id DESC LIMIT :lim"
            )
            params: Dict[str, Any] = {"fid": int(feedback_id), "lim": max(1, int(limit))}
        else:
            stmt = text(
                "SELECT * FROM feedback_review_logs "
                "ORDER BY changed_at DESC, log_id DESC LIMIT :lim"
            )
            params = {"lim": max(1, int(limit))}

        with eng.connect() as conn:
            rows = conn.execute(stmt, params).mappings().all()

        return [
            ReviewLogEntry(
                log_id=int(r["log_id"]),
                feedback_id=int(r["feedback_id"]),
                previous_review_status=r["previous_review_status"],
                new_review_status=str(r["new_review_status"]),
                previous_sentiment_label=(
                    int(r["previous_sentiment_label"])
                    if r["previous_sentiment_label"] is not None
                    else None
                ),
                new_sentiment_label=(
                    int(r["new_sentiment_label"])
                    if r["new_sentiment_label"] is not None
                    else None
                ),
                previous_topic_code=r["previous_topic_code"],
                new_topic_code=r["new_topic_code"],
                reviewed_by=str(r["reviewed_by"]),
                is_authenticated_user=bool(r["is_authenticated_user"]),
                review_note=r["review_note"],
                record_version_before=int(r["record_version_before"]),
                record_version_after=int(r["record_version_after"]),
                changed_at=r["changed_at"],
            )
            for r in rows
        ]

    def export_records(
        self, filters: FilterCriteria, limit: int = 5000
    ) -> List[Dict[str, object]]:
        page_res = self.list_paginated(
            filters=filters,
            page=1,
            page_size=max(1, min(int(limit), 10000)),
            sort_by="newest",
        )
        out: List[Dict[str, object]] = []
        for r in page_res.items:
            safe_text, _ = mask_pii(r.masked_content)
            out.append(
                {
                    "feedback_id": r.feedback_id,
                    "data_origin": r.data_origin,
                    "feedback_date": r.feedback_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "unit_code": r.unit_code,
                    "source_code": r.source_code,
                    "masked_content": safe_text,
                    "processing_status": r.processing_status,
                    "predicted_sentiment_label": r.predicted_sentiment_label,
                    "predicted_sentiment_vi": r.predicted_sentiment_vi,
                    "confidence_score": r.confidence_score,
                    "predicted_topic_code": r.predicted_topic_code,
                    "topic_assignment_source": r.topic_assignment_source,
                    "review_status": r.review_status,
                    "reviewed_sentiment_label": r.reviewed_sentiment_label,
                    "reviewed_topic_code": r.reviewed_topic_code,
                    "reviewed_by_unverified": r.reviewed_by,
                    "reviewed_at": (
                        r.reviewed_at.strftime("%Y-%m-%d %H:%M:%S")
                        if r.reviewed_at
                        else None
                    ),
                    "review_note": r.review_note,
                    "model_version": r.model_version,
                }
            )
        return out

    def save_import_batch(
        self, summary: ImportExecutionSummary, origin_note: Optional[str] = None
    ) -> None:
        batch_entry: Dict[str, object] = {
            "batch_id": summary.batch_id,
            "filename": summary.filename,
            "declared_origin": summary.declared_origin,
            "origin_confirmation_note": origin_note,
            "total_rows": int(summary.total_rows_read),
            "saved_count": int(summary.saved_count),
            "duplicate_skipped_count": int(summary.duplicate_skipped_count),
            "validation_error_count": int(summary.validation_error_count),
            "inference_error_count": int(summary.inference_error_count),
            "db_error_count": int(summary.db_error_count),
            "unprocessed_count": int(summary.unprocessed_count),
            "created_at": self._now_local(),
        }
        self._import_batches_cache.insert(0, batch_entry)
        if self._engine is None:
            return
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO import_batches (
                            batch_id, filename, declared_origin, origin_confirmation_note,
                            total_rows, saved_count, duplicate_skipped_count,
                            validation_error_count, inference_error_count,
                            db_error_count, unprocessed_count, created_at
                        ) VALUES (
                            :batch_id, :filename, :declared_origin, :origin_confirmation_note,
                            :total_rows, :saved_count, :duplicate_skipped_count,
                            :validation_error_count, :inference_error_count,
                            :db_error_count, :unprocessed_count, :created_at
                        )
                        """
                    ),
                    batch_entry,
                )
        except SQLAlchemyError:
            pass

    def list_import_batches(self, limit: int = 20) -> List[Dict[str, object]]:
        if self._engine is not None:
            try:
                with self._engine.connect() as conn:
                    rows = (
                        conn.execute(
                            text(
                                "SELECT * FROM import_batches "
                                "ORDER BY created_at DESC LIMIT :lim"
                            ),
                            {"lim": max(1, int(limit))},
                        )
                        .mappings()
                        .all()
                    )
                if rows:
                    return [dict(r) for r in rows]
            except SQLAlchemyError:
                pass
        return list(self._import_batches_cache[: max(1, int(limit))])
