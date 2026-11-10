"""Repository dữ liệu mô phỏng trong phiên (Demo Mode), đọc khởi tạo từ tệp fixture."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from tnu_sentiment.constants import (
    INITIAL_SOURCES,
    INITIAL_TOPICS,
    INITIAL_UNITS,
    ReviewStatus,
    SENTIMENT_ID_TO_CODE,
    USER_STATUS_LABELS,
)
from tnu_sentiment.domain.models import (
    CatalogItem,
    ComponentHealth,
    FeedbackRecord,
    FilterCriteria,
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


class DemoFeedbackRepository(FeedbackRepository):
    """Lưu trữ dữ liệu demo cô lập theo phiên người dùng (st.session_state).
    Mọi thao tác thêm mới/import/kiểm duyệt chỉ tác động tới bản sao trong phiên hiện tại.
    """

    def __init__(
        self,
        fixture_path: Path,
        tz_name: str = "Asia/Ho_Chi_Minh",
    ) -> None:
        self._fixture_path = fixture_path
        self._tz = ZoneInfo(tz_name)
        self._units: Dict[str, str] = {code: name for code, name in INITIAL_UNITS}
        self._sources: Dict[str, str] = {code: name for code, name in INITIAL_SOURCES}
        self._topics: Dict[str, str] = {code: name for code, name, _ in INITIAL_TOPICS}

        self._records: List[FeedbackRecord] = []
        self._review_logs: List[ReviewLogEntry] = []
        self._next_id: int = 1
        self._next_log_id: int = 1
        self._idempotency_index: Dict[str, int] = {}
        self._source_record_index: Dict[Tuple[str, str], int] = {}
        self._content_hashes: Dict[str, int] = {}

        self._load_fixture()

    def _now_local(self) -> datetime:
        return datetime.now(self._tz).replace(tzinfo=None)

    def _load_fixture(self) -> None:
        if not self._fixture_path.exists():
            return
        with self._fixture_path.open("r", encoding="utf-8") as fh:
            raw_items = json.load(fh)

        for item in raw_items:
            masked_text, pii_types = mask_pii(str(item.get("masked_content", "")))
            c_hash = item.get("content_hash") or compute_content_hash(masked_text)
            fb_dt = datetime.fromisoformat(str(item["feedback_date"]))
            created_dt = datetime.fromisoformat(
                str(item.get("created_at", item["feedback_date"]))
            )
            updated_dt = datetime.fromisoformat(
                str(item.get("updated_at", item["feedback_date"]))
            )
            reviewed_at_str = item.get("reviewed_at")
            reviewed_dt = (
                datetime.fromisoformat(str(reviewed_at_str))
                if reviewed_at_str
                else None
            )

            u_code = str(item.get("unit_code", "UNSPECIFIED"))
            s_code = str(item.get("source_code", "UNSPECIFIED"))
            t_code = str(item.get("predicted_topic_code", "UNCLASSIFIED"))
            rev_t_code = item.get("reviewed_topic_code")

            fid = int(item.get("feedback_id", self._next_id))
            self._next_id = max(self._next_id, fid + 1)

            rec = FeedbackRecord(
                feedback_id=fid,
                data_origin=str(item.get("data_origin", "DEMO")),
                origin_note=item.get("origin_note", "Fixture dữ liệu mô phỏng giao diện"),
                unit_code=u_code,
                unit_name=self._units.get(u_code, u_code),
                source_code=s_code,
                source_name=self._sources.get(s_code, s_code),
                masked_content=masked_text,
                processed_content=item.get("processed_content"),
                pii_types_detected=",".join(pii_types) if pii_types else item.get("pii_types_detected"),
                was_truncated=bool(item.get("was_truncated", False)),
                is_multi_aspect=bool(item.get("is_multi_aspect", False)),
                is_duplicate_candidate=c_hash in self._content_hashes,
                processing_status=str(item.get("processing_status", "DONE")),
                processing_error=item.get("processing_error"),
                predicted_sentiment_label=item.get("predicted_sentiment_label"),
                predicted_sentiment_text=item.get("predicted_sentiment_text"),
                confidence_score=item.get("confidence_score"),
                prob_negative=item.get("prob_negative"),
                prob_neutral=item.get("prob_neutral"),
                prob_positive=item.get("prob_positive"),
                predicted_topic_code=t_code,
                predicted_topic_name=self._topics.get(t_code, t_code),
                topic_assignment_source=str(item.get("topic_assignment_source", "MANUAL")),
                model_version=item.get("model_version"),
                pipeline_version=item.get("pipeline_version"),
                inference_time_ms=item.get("inference_time_ms"),
                review_status=str(item.get("review_status", "UNREVIEWED")),
                review_reason=item.get("review_reason"),
                reviewed_sentiment_label=item.get("reviewed_sentiment_label"),
                reviewed_topic_code=rev_t_code,
                reviewed_topic_name=self._topics.get(rev_t_code, rev_t_code) if rev_t_code else None,
                reviewed_by=item.get("reviewed_by"),
                reviewed_at=reviewed_dt,
                review_note=item.get("review_note"),
                record_version=int(item.get("record_version", 1)),
                feedback_date=fb_dt,
                created_at=created_dt,
                updated_at=updated_dt,
                idempotency_key=item.get("idempotency_key"),
                source_record_id=item.get("source_record_id"),
                content_hash=c_hash,
            )
            self._records.append(rec)
            self._content_hashes[c_hash] = fid
            if rec.idempotency_key:
                self._idempotency_index[rec.idempotency_key] = fid
            if rec.source_record_id:
                self._source_record_index[(rec.source_code, rec.source_record_id)] = fid

    @property
    def repository_kind(self) -> str:
        return "DEMO"

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            ready=True,
            user_badge=USER_STATUS_LABELS["demo_mode"],
            technical_detail=(
                f"Đang sử dụng bộ dữ liệu mô phỏng trong bộ nhớ phiên ({len(self._records)} bản ghi từ "
                f"{self._fixture_path.name}). Mọi thay đổi sẽ mất khi kết thúc phiên làm việc."
            ),
        )

    def list_units(self) -> List[CatalogItem]:
        return [CatalogItem(code=c, name=n) for c, n in self._units.items()]

    def list_sources(self) -> List[CatalogItem]:
        return [CatalogItem(code=c, name=n) for c, n in self._sources.items()]

    def list_topics(self) -> List[CatalogItem]:
        return [CatalogItem(code=c, name=n) for c, n in self._topics.items()]

    def _matches(self, rec: FeedbackRecord, f: FilterCriteria) -> bool:
        if not f.select_all_origins and f.data_origins:
            if rec.data_origin not in f.data_origins:
                return False

        start_dt, end_excl = compute_half_open_date_bounds(f.date_from, f.date_to)
        if start_dt is not None and rec.feedback_date < start_dt:
            return False
        if end_excl is not None and rec.feedback_date >= end_excl:
            return False

        if f.unit_codes and rec.unit_code not in f.unit_codes:
            return False
        if f.source_codes and rec.source_code not in f.source_codes:
            return False
        if f.topic_codes:
            eff_topic = rec.effective_topic_code(f.label_policy)
            if eff_topic not in f.topic_codes:
                return False
        if f.processing_statuses and rec.processing_status not in f.processing_statuses:
            return False
        if f.review_statuses and rec.review_status not in f.review_statuses:
            return False
        if f.sentiment_filter:
            eff_sent = rec.effective_sentiment_label(f.label_policy)
            if eff_sent not in f.sentiment_filter:
                return False
        if f.search_query.strip():
            q = f.search_query.strip().lower()
            if q not in rec.masked_content.lower():
                return False
        return True

    def list_matching_for_stats(self, filters: FilterCriteria) -> List[FeedbackRecord]:
        return [copy.copy(r) for r in self._records if self._matches(r, filters)]

    def list_paginated(
        self,
        filters: FilterCriteria,
        page: int = 1,
        page_size: int = 15,
        sort_by: str = "newest",
        only_negative_for_review: bool = False,
    ) -> PaginatedRecords:
        matched = [r for r in self._records if self._matches(r, filters)]
        if only_negative_for_review:
            matched = [
                r
                for r in matched
                if r.effective_sentiment_label(filters.label_policy) == 0
            ]

        if sort_by == "confidence_asc":
            matched.sort(
                key=lambda r: (
                    r.confidence_score is None,
                    r.confidence_score if r.confidence_score is not None else 999.0,
                    -r.feedback_date.timestamp(),
                )
            )
        elif sort_by == "confidence_desc":
            matched.sort(
                key=lambda r: (
                    r.confidence_score is not None,
                    r.confidence_score if r.confidence_score is not None else -1.0,
                    r.feedback_date.timestamp(),
                ),
                reverse=True,
            )
        elif sort_by == "oldest":
            matched.sort(key=lambda r: (r.feedback_date, r.feedback_id))
        else:
            matched.sort(key=lambda r: (r.feedback_date, r.feedback_id), reverse=True)

        total = len(matched)
        safe_size = max(1, page_size)
        max_page = max(1, (total + safe_size - 1) // safe_size)
        safe_page = min(max(1, page), max_page)
        start = (safe_page - 1) * safe_size
        sliced = [copy.copy(r) for r in matched[start : start + safe_size]]

        return PaginatedRecords(
            items=sliced,
            total_count=total,
            page=safe_page,
            page_size=safe_size,
        )

    def get_by_id(self, feedback_id: int) -> Optional[FeedbackRecord]:
        for r in self._records:
            if r.feedback_id == feedback_id:
                return copy.copy(r)
        return None

    def check_source_record_exists(
        self, source_code: str, source_record_id: str
    ) -> bool:
        if not source_record_id:
            return False
        return (source_code, source_record_id) in self._source_record_index

    def save_feedback(self, payload: NewFeedbackPayload) -> SaveRecordResult:
        if payload.idempotency_key and payload.idempotency_key in self._idempotency_index:
            existing_id = self._idempotency_index[payload.idempotency_key]
            return SaveRecordResult(
                saved=False,
                was_idempotent_duplicate=True,
                was_source_duplicate=False,
                is_content_duplicate_candidate=False,
                feedback_id=existing_id,
                message=f"Thao tác đã được lưu trước đó (ID #{existing_id}), hệ thống chặn ghi lặp.",
            )

        if payload.source_record_id:
            src_key = (payload.source_code, payload.source_record_id)
            if src_key in self._source_record_index:
                existing_id = self._source_record_index[src_key]
                return SaveRecordResult(
                    saved=False,
                    was_idempotent_duplicate=False,
                    was_source_duplicate=True,
                    is_content_duplicate_candidate=False,
                    feedback_id=existing_id,
                    message=(
                        f"Bỏ qua bản ghi nguồn trùng source_record_id='{payload.source_record_id}' "
                        f"(đã tồn tại ở ID #{existing_id})."
                    ),
                )

        safe_masked, detected_pii = mask_pii(payload.masked_content)
        c_hash = payload.content_hash or compute_content_hash(safe_masked)
        dup_candidate = c_hash in self._content_hashes

        now_dt = self._now_local()
        fid = self._next_id
        self._next_id += 1

        u_code = payload.unit_code if payload.unit_code in self._units else "UNSPECIFIED"
        s_code = payload.source_code if payload.source_code in self._sources else "UNSPECIFIED"
        t_code = (
            payload.predicted_topic_code
            if payload.predicted_topic_code in self._topics
            else "UNCLASSIFIED"
        )
        sent_text = (
            SENTIMENT_ID_TO_CODE.get(payload.predicted_sentiment_label)
            if payload.predicted_sentiment_label is not None
            else payload.predicted_sentiment_text
        )

        rec = FeedbackRecord(
            feedback_id=fid,
            data_origin=payload.data_origin,
            origin_note=payload.origin_note,
            unit_code=u_code,
            unit_name=self._units.get(u_code, u_code),
            source_code=s_code,
            source_name=self._sources.get(s_code, s_code),
            masked_content=safe_masked,
            processed_content=payload.processed_content,
            pii_types_detected=(
                ",".join(detected_pii) if detected_pii else payload.pii_types_detected
            ),
            was_truncated=payload.was_truncated,
            is_multi_aspect=payload.is_multi_aspect,
            is_duplicate_candidate=dup_candidate,
            processing_status=payload.processing_status,
            processing_error=payload.processing_error,
            predicted_sentiment_label=payload.predicted_sentiment_label,
            predicted_sentiment_text=sent_text,
            confidence_score=payload.confidence_score,
            prob_negative=payload.prob_negative,
            prob_neutral=payload.prob_neutral,
            prob_positive=payload.prob_positive,
            predicted_topic_code=t_code,
            predicted_topic_name=self._topics.get(t_code, t_code),
            topic_assignment_source=payload.topic_assignment_source,
            model_version=payload.model_version,
            pipeline_version=payload.pipeline_version,
            inference_time_ms=payload.inference_time_ms,
            review_status=payload.review_status,
            review_reason=payload.review_reason,
            reviewed_sentiment_label=None,
            reviewed_topic_code=None,
            reviewed_topic_name=None,
            reviewed_by=None,
            reviewed_at=None,
            review_note=None,
            record_version=1,
            feedback_date=payload.feedback_date,
            created_at=now_dt,
            updated_at=now_dt,
            idempotency_key=payload.idempotency_key,
            source_record_id=payload.source_record_id,
            content_hash=c_hash,
        )
        self._records.append(rec)
        self._content_hashes.setdefault(c_hash, fid)
        if payload.idempotency_key:
            self._idempotency_index[payload.idempotency_key] = fid
        if payload.source_record_id:
            self._source_record_index[(s_code, payload.source_record_id)] = fid

        extra_dup = (
            " (Lưu ý: nội dung trùng khớp với một phản hồi trước đó — đã gắn cờ ứng viên trùng nội dung)"
            if dup_candidate
            else ""
        )
        return SaveRecordResult(
            saved=True,
            was_idempotent_duplicate=False,
            was_source_duplicate=False,
            is_content_duplicate_candidate=dup_candidate,
            feedback_id=fid,
            message=f"Đã lưu tạm bản ghi #{fid} vào bộ dữ liệu Demo trong phiên{extra_dup}.",
        )

    def apply_review_updates(
        self, updates: List[ReviewUpdateRequest]
    ) -> Tuple[int, List[str]]:
        by_id = {r.feedback_id: r for r in self._records}
        applied = 0
        errors: List[str] = []
        now_dt = self._now_local()

        for upd in updates:
            rec = by_id.get(upd.feedback_id)
            if rec is None:
                errors.append(f"Không tìm thấy bản ghi ID #{upd.feedback_id}.")
                continue
            if rec.record_version != upd.expected_version:
                errors.append(
                    f"Bản ghi ID #{upd.feedback_id} đã thay đổi phiên bản "
                    f"(hiện tại v{rec.record_version} != v{upd.expected_version}). Vui lòng tải lại bảng."
                )
                continue

            reviewer = upd.reviewed_by.strip()
            if not reviewer:
                errors.append(f"Bản ghi ID #{upd.feedback_id}: cần nhập tên/mã người duyệt.")
                continue

            if upd.new_review_status == ReviewStatus.VERIFIED.value:
                if upd.reviewed_sentiment_label not in (0, 1, 2):
                    errors.append(
                        f"Bản ghi ID #{upd.feedback_id}: khi chuyển sang Đã xác nhận (VERIFIED), "
                        "cần chọn nhãn cảm xúc được duyệt hợp lệ (0/1/2)."
                    )
                    continue

            safe_note, _ = mask_pii(upd.review_note or "")
            prev_status = rec.review_status
            prev_sent = rec.reviewed_sentiment_label
            prev_topic = rec.reviewed_topic_code
            prev_ver = rec.record_version

            # KHÔNG ghi đè predicted_sentiment_label / predicted_topic_code!
            rec.review_status = upd.new_review_status
            rec.reviewed_sentiment_label = upd.reviewed_sentiment_label
            rec.reviewed_topic_code = upd.reviewed_topic_code
            rec.reviewed_topic_name = (
                self._topics.get(upd.reviewed_topic_code, upd.reviewed_topic_code)
                if upd.reviewed_topic_code
                else None
            )
            rec.reviewed_by = reviewer
            rec.reviewed_at = now_dt
            rec.review_note = safe_note or None
            rec.record_version = prev_ver + 1
            rec.updated_at = now_dt

            log_entry = ReviewLogEntry(
                log_id=self._next_log_id,
                feedback_id=rec.feedback_id,
                previous_review_status=prev_status,
                new_review_status=rec.review_status,
                previous_sentiment_label=prev_sent,
                new_sentiment_label=rec.reviewed_sentiment_label,
                previous_topic_code=prev_topic,
                new_topic_code=rec.reviewed_topic_code,
                reviewed_by=reviewer,
                is_authenticated_user=False,
                review_note=rec.review_note,
                record_version_before=prev_ver,
                record_version_after=rec.record_version,
                changed_at=now_dt,
            )
            self._next_log_id += 1
            self._review_logs.insert(0, log_entry)
            applied += 1

        return applied, errors

    def list_review_logs(
        self, feedback_id: Optional[int] = None, limit: int = 50
    ) -> List[ReviewLogEntry]:
        logs = self._review_logs
        if feedback_id is not None:
            logs = [lg for lg in logs if lg.feedback_id == feedback_id]
        return [copy.copy(lg) for lg in logs[: max(1, limit)]]

    def export_records(
        self, filters: FilterCriteria, limit: int = 5000
    ) -> List[Dict[str, object]]:
        matched = [r for r in self._records if self._matches(r, filters)]
        matched.sort(key=lambda r: (r.feedback_date, r.feedback_id), reverse=True)
        out: List[Dict[str, object]] = []
        for r in matched[: max(1, limit)]:
            # Chỉ xuất dữ liệu đã che PII, tuyệt đối KHÔNG xuất author_hash hay secret
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

