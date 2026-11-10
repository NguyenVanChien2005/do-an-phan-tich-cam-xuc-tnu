"""Dịch vụ phân tích phản hồi đơn lẻ, quản lý câu mẫu Demo và lưu chống ghi lặp."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from tnu_sentiment.config import AppConfig
from tnu_sentiment.constants import (
    AppMode,
    DataOrigin,
    ProcessingStatus,
    ReviewStatus,
    SENTIMENT_ID_TO_CODE,
    SENTIMENT_ID_TO_VI,
    TopicAssignmentSource,
)
from tnu_sentiment.domain.models import (
    AnalysisOutcome,
    SaveRecordResult,
    SentimentPrediction,
)
from tnu_sentiment.models.llm_helper import OptionalLLMHelper
from tnu_sentiment.models.sentiment_adapter import BaseSentimentAdapter
from tnu_sentiment.nlp.pii_masker import compute_author_hash, compute_content_hash
from tnu_sentiment.nlp.preprocessor import (
    SegmenterUnavailableError,
    VietnamesePreprocessor,
)
from tnu_sentiment.repositories.base import FeedbackRepository, NewFeedbackPayload


def compute_text_signature(raw_text: str) -> str:
    """Chữ ký SHA-256 của nội dung nguyên gốc để phát hiện chỉnh sửa sau khi đã phân tích."""
    return hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()


class AnalysisService:
    """Điều phối luồng Nhập liệu -> Che PII & Tiền xử lý -> Suy luận -> Đánh giá độ tự tin -> Lưu."""

    def __init__(
        self,
        config: AppConfig,
        sentiment_adapter: BaseSentimentAdapter,
        preprocessor: VietnamesePreprocessor,
        llm_helper: OptionalLLMHelper,
    ) -> None:
        self._cfg = config
        self._sentiment = sentiment_adapter
        self._preprocessor = preprocessor
        self._llm = llm_helper
        self._sample_cases = self._load_sample_cases(
            config.fixtures_dir / "demo_sample_cases.json"
        )

    @staticmethod
    def _load_sample_cases(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []

    def list_sample_cases(self) -> List[Dict[str, Any]]:
        return list(self._sample_cases)

    def find_matching_sample_case(self, raw_text: str) -> Optional[Dict[str, Any]]:
        cleaned = (raw_text or "").strip()
        for case in self._sample_cases:
            if str(case.get("text", "")).strip() == cleaned:
                return case
        return None

    @staticmethod
    def is_outcome_stale(current_raw_text: str, outcome: Optional[AnalysisOutcome]) -> bool:
        if outcome is None:
            return True
        return compute_text_signature(current_raw_text) != outcome.analyzed_text_signature

    def analyze(
        self,
        raw_text: str,
        mode: AppMode,
        unit_code: str,
        source_code: str,
        manual_topic_code: str,
        confirmed_tnu_real: bool = False,
        origin_note: Optional[str] = None,
    ) -> AnalysisOutcome:
        text_stripped = (raw_text or "").strip()
        if not text_stripped:
            raise ValueError("Vui lòng nhập nội dung phản hồi trước khi phân tích.")
        if len(text_stripped) > self._cfg.max_input_chars:
            raise ValueError(
                f"Nội dung vượt quá giới hạn cấu hình ({len(text_stripped)}/{self._cfg.max_input_chars} ký tự)."
            )

        t_start = time.perf_counter()
        idempotency_key = f"analyze-{uuid.uuid4().hex}"
        signature = compute_text_signature(raw_text)

        matched_sample = self.find_matching_sample_case(text_stripped)

        # Quy tắc trung thực về nguồn gốc dữ liệu:
        # - Chế độ Demo hoặc câu mẫu có sẵn luôn gắn DEMO (không bao giờ đưa câu mẫu vào thống kê thực tế).
        # - Chế độ Thật chỉ gắn TNU_REAL khi người nhập xác nhận rõ nguồn khảo sát thực tế ĐHTN.
        if mode == AppMode.DEMO or matched_sample is not None:
            resolved_origin = DataOrigin.DEMO.value
            resolved_note = (
                "Câu mẫu thử nghiệm giao diện"
                if matched_sample is not None
                else "Nhập thử nghiệm trong chế độ Demo"
            )
        elif confirmed_tnu_real:
            resolved_origin = DataOrigin.TNU_REAL.value
            resolved_note = (origin_note or "Phản hồi thực tế xác nhận thủ công").strip()
        else:
            resolved_origin = DataOrigin.DEMO.value
            resolved_note = "Nhập kiểm thử trực tiếp (chưa xác nhận nguồn khảo sát thực tế ĐHTN)"

        # Thực thi tiền xử lý
        try:
            prep_res = self._preprocessor.run(text_stripped)
        except SegmenterUnavailableError as seg_err:
            fallback_prep = VietnamesePreprocessor(require_segmenter=False).run(text_stripped)
            return AnalysisOutcome(
                idempotency_key=idempotency_key,
                analyzed_raw_length=len(text_stripped),
                analyzed_text_signature=signature,
                preprocess=fallback_prep,
                prediction=None,
                selected_topic_code=manual_topic_code or "UNCLASSIFIED",
                topic_source=TopicAssignmentSource.MANUAL.value,
                unit_code=unit_code or "UNSPECIFIED",
                source_code=source_code or "UNSPECIFIED",
                data_origin=resolved_origin,
                origin_note=resolved_note,
                review_status=ReviewStatus.NEEDS_REVIEW.value,
                review_reason=str(seg_err),
                total_elapsed_ms=None,
                error_message=str(seg_err),
            )

        # Suy luận cảm xúc
        prediction: Optional[SentimentPrediction] = None
        error_msg: Optional[str] = None

        if self._sentiment.is_ready:
            try:
                prediction = self._sentiment.predict(prep_res.processed_content)
            except Exception as exc:
                error_msg = f"Lỗi trong quá trình suy luận mô hình: {exc}"
        else:
            # Khi KHÔNG có checkpoint mô hình:
            # - Ở chế độ DEMO và khớp chính xác câu mẫu có kết quả minh họa cố định -> trả kết quả minh họa rõ nhãn.
            # - Mọi trường hợp văn bản mới hoặc ở chế độ REAL -> báo rõ chưa thể phân tích, tuyệt đối không tạo số giả.
            if (
                mode == AppMode.DEMO
                and matched_sample is not None
                and matched_sample.get("illustrative_result") is not None
            ):
                ill = matched_sample["illustrative_result"]
                lbl_id = int(ill["label_id"])
                probs_raw = ill["probabilities"]
                probs = {
                    0: float(probs_raw["0"]),
                    1: float(probs_raw["1"]),
                    2: float(probs_raw["2"]),
                }
                prediction = SentimentPrediction(
                    label_id=lbl_id,
                    label_code=SENTIMENT_ID_TO_CODE[lbl_id],
                    label_vi=SENTIMENT_ID_TO_VI[lbl_id],
                    confidence=float(ill["confidence"]),
                    probabilities=probs,
                    model_version="DEMO_STATIC_FIXTURE (Kết quả minh họa cố định)",
                    was_truncated=False,
                    token_count=len(prep_res.processed_content.split()),
                    max_length=self._cfg.sentiment_max_length,
                    inference_time_ms=None,
                    measurement_scope=(
                        "Kết quả minh họa cố định từ tệp mẫu Demo — không thực thi suy luận PhoBERT"
                    ),
                    is_illustrative_demo=True,
                )
            else:
                error_msg = (
                    "Chưa thể phân tích văn bản này vì chưa cấu hình checkpoint mô hình cảm xúc hợp lệ. "
                    + (
                        "Trong chế độ Demo khi thiếu mô hình, bạn chỉ có thể chọn nguyên vẹn một câu mẫu có sẵn để xem kết quả minh họa."
                        if mode == AppMode.DEMO
                        else "Hệ thống vẫn cho phép tra cứu và kiểm duyệt dữ liệu đã lưu trong CSDL."
                    )
                )

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2) if prediction and not prediction.is_illustrative_demo else None

        review_status = ReviewStatus.UNREVIEWED.value
        review_reason: Optional[str] = None
        if prediction is not None:
            if (
                prediction.confidence is not None
                and prediction.confidence < self._cfg.confidence_review_threshold
            ):
                review_status = ReviewStatus.NEEDS_REVIEW.value
                review_reason = (
                    f"Confidence ({prediction.confidence:.2%}) thấp hơn ngưỡng cấu hình thử nghiệm "
                    f"({self._cfg.confidence_review_threshold:.0%})."
                )
            elif prep_res.is_multi_aspect:
                review_status = ReviewStatus.NEEDS_REVIEW.value
                review_reason = (
                    "Phát hiện câu nhiều ý/mệnh đề tương phản — khuyến nghị kiểm duyệt thủ công."
                )

        return AnalysisOutcome(
            idempotency_key=idempotency_key,
            analyzed_raw_length=len(text_stripped),
            analyzed_text_signature=signature,
            preprocess=prep_res,
            prediction=prediction,
            selected_topic_code=manual_topic_code or "UNCLASSIFIED",
            topic_source=TopicAssignmentSource.MANUAL.value,
            unit_code=unit_code or "UNSPECIFIED",
            source_code=source_code or "UNSPECIFIED",
            data_origin=resolved_origin,
            origin_note=resolved_note,
            review_status=review_status,
            review_reason=review_reason,
            total_elapsed_ms=total_ms,
            error_message=error_msg,
        )

    def save_outcome(
        self,
        repository: FeedbackRepository,
        current_raw_text: str,
        outcome: AnalysisOutcome,
        override_topic_code: Optional[str] = None,
    ) -> SaveRecordResult:
        if self.is_outcome_stale(current_raw_text, outcome):
            raise ValueError(
                "Nội dung văn bản đã bị chỉnh sửa sau lần phân tích gần nhất. "
                "Vui lòng bấm 'Phân tích' lại trước khi lưu để không lưu nhầm kết quả cũ."
            )
        if outcome.prediction is None or outcome.error_message:
            raise ValueError("Chưa có kết quả phân tích hợp lệ để lưu.")

        now_local = datetime.now(ZoneInfo(self._cfg.timezone_name)).replace(tzinfo=None)
        pred = outcome.prediction
        topic_code = override_topic_code or outcome.selected_topic_code or "UNCLASSIFIED"
        c_hash = compute_content_hash(outcome.preprocess.masked_content)
        a_hash = compute_author_hash(
            f"{outcome.source_code}:{outcome.unit_code}", self._cfg.secret_salt
        )

        llm_sent = (
            outcome.llm_suggestion.suggested_sentiment_label
            if outcome.llm_suggestion
            else None
        )
        llm_topic = (
            outcome.llm_suggestion.suggested_topic_code
            if outcome.llm_suggestion
            else None
        )
        llm_note = (
            outcome.llm_suggestion.explanation if outcome.llm_suggestion else None
        )

        payload = NewFeedbackPayload(
            data_origin=outcome.data_origin,
            origin_note=outcome.origin_note,
            unit_code=outcome.unit_code,
            source_code=outcome.source_code,
            masked_content=outcome.preprocess.masked_content,
            processed_content=outcome.preprocess.processed_content,
            pii_types_detected=(
                ",".join(outcome.preprocess.pii_types_detected)
                if outcome.preprocess.pii_types_detected
                else None
            ),
            was_truncated=pred.was_truncated,
            is_multi_aspect=outcome.preprocess.is_multi_aspect,
            processing_status=ProcessingStatus.DONE.value,
            processing_error=None,
            predicted_sentiment_label=pred.label_id,
            predicted_sentiment_text=pred.label_code,
            confidence_score=pred.confidence,
            prob_negative=pred.probabilities.get(0),
            prob_neutral=pred.probabilities.get(1),
            prob_positive=pred.probabilities.get(2),
            predicted_topic_code=topic_code,
            topic_assignment_source=outcome.topic_source,
            model_version=pred.model_version,
            pipeline_version=outcome.preprocess.pipeline_version,
            inference_time_ms=outcome.total_elapsed_ms or pred.inference_time_ms,
            review_status=outcome.review_status,
            review_reason=outcome.review_reason,
            feedback_date=now_local,
            idempotency_key=outcome.idempotency_key,
            source_record_id=None,
            content_hash=c_hash,
            author_hash=a_hash,
            llm_suggested_sentiment=llm_sent,
            llm_suggested_topic_code=llm_topic,
            llm_note=llm_note,
        )
        return repository.save_feedback(payload)

