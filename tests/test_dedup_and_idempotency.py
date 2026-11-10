"""Kiểm thử chống ghi lặp (idempotency, source_record_id, content_hash), phát hiện sửa văn bản sau phân tích
và nguyên tắc không suy luận giả khi thiếu mô hình."""

from __future__ import annotations

from pathlib import Path
import pytest

from tnu_sentiment.config import load_app_config
from tnu_sentiment.constants import AppMode, DataOrigin
from tnu_sentiment.domain.models import SentimentPrediction
from tnu_sentiment.models.llm_helper import OptionalLLMHelper
from tnu_sentiment.models.sentiment_adapter import (
    BaseSentimentAdapter,
    UnavailableSentimentAdapter,
)
from tnu_sentiment.domain.models import ComponentHealth
from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor
from tnu_sentiment.repositories.demo_repo import DemoFeedbackRepository
from tnu_sentiment.services.analysis_service import AnalysisService


class _StubReadyAdapter(BaseSentimentAdapter):
    @property
    def is_ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return "stub-phobert-test"

    def health(self) -> ComponentHealth:
        return ComponentHealth(True, "Mô hình cảm xúc sẵn sàng", "stub")

    def predict(self, processed_text: str) -> SentimentPrediction:
        return SentimentPrediction(
            label_id=2,
            label_code="Positive",
            label_vi="Tích cực",
            confidence=0.91,
            probabilities={0: 0.04, 1: 0.05, 2: 0.91},
            model_version="stub-phobert-test",
            was_truncated=False,
            token_count=6,
            max_length=256,
            inference_time_ms=12.3,
            measurement_scope="Stub test",
        )


def test_no_fake_inference_for_arbitrary_text_when_model_missing(tmp_path: Path) -> None:
    cfg = load_app_config()
    missing_adapter = UnavailableSentimentAdapter("Thiếu checkpoint")
    svc = AnalysisService(
        config=cfg,
        sentiment_adapter=missing_adapter,
        preprocessor=VietnamesePreprocessor(require_segmenter=False),
        llm_helper=OptionalLLMHelper(cfg),
    )

    # Văn bản tùy ý mới trong chế độ DEMO khi không có model -> KHÔNG được sinh nhãn giả
    out_demo = svc.analyze(
        raw_text="Một câu phản hồi hoàn toàn mới chưa từng có trong bộ mẫu.",
        mode=AppMode.DEMO,
        unit_code="ICTU",
        source_code="QA_BOX",
        manual_topic_code="FACILITY",
    )
    assert out_demo.prediction is None
    assert out_demo.error_message is not None

    # Câu mẫu có sẵn trong chế độ DEMO khi không có model -> được phép hiển thị kết quả minh họa cố định
    sample_txt = svc.list_sample_cases()[0]["text"]
    out_sample = svc.analyze(
        raw_text=sample_txt,
        mode=AppMode.DEMO,
        unit_code="ICTU",
        source_code="SURVEY_PORTAL",
        manual_topic_code="LECTURER",
    )
    assert out_sample.prediction is not None
    assert out_sample.prediction.is_illustrative_demo is True
    assert out_sample.prediction.inference_time_ms is None


def test_stale_edit_blocks_save_and_idempotency_prevents_double_save(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = DemoFeedbackRepository(fixture_path=tmp_path / "empty.json")
    svc = AnalysisService(
        config=cfg,
        sentiment_adapter=_StubReadyAdapter(),
        preprocessor=VietnamesePreprocessor(),
        llm_helper=OptionalLLMHelper(cfg),
    )

    text_v1 = "Giảng viên hướng dẫn rất nhiệt tình."
    outcome = svc.analyze(
        raw_text=text_v1,
        mode=AppMode.REAL,
        unit_code="ICTU",
        source_code="SURVEY_PORTAL",
        manual_topic_code="LECTURER",
        confirmed_tnu_real=True,
    )
    assert outcome.data_origin == DataOrigin.TNU_REAL.value

    # Đổi nội dung sau khi phân tích -> không được lưu nhầm kết quả cũ
    text_v2 = "Giảng viên hướng dẫn rất tệ và đi trễ."
    assert svc.is_outcome_stale(text_v2, outcome) is True
    with pytest.raises(ValueError, match="chỉnh sửa"):
        svc.save_outcome(repo, text_v2, outcome)

    # Lưu đúng văn bản đã phân tích lần 1 -> thành công
    res1 = svc.save_outcome(repo, text_v1, outcome)
    assert res1.saved is True
    assert res1.feedback_id == 1

    # Bấm lưu lặp cùng một thao tác (cùng idempotency_key) -> chặn ghi lặp!
    res2 = svc.save_outcome(repo, text_v1, outcome)
    assert res2.saved is False
    assert res2.was_idempotent_duplicate is True
    assert len(repo.list_matching_for_stats(
        type("F", (), {
            "select_all_origins": True,
            "data_origins": [],
            "date_from": None,
            "date_to": None,
            "unit_codes": [],
            "source_codes": [],
            "topic_codes": [],
            "processing_statuses": [],
            "review_statuses": [],
            "sentiment_filter": [],
            "search_query": "",
            "label_policy": None,
        })()
    )) == 1


def test_sample_sentences_always_forced_to_demo_origin_even_in_real_mode() -> None:
    cfg = load_app_config()
    svc = AnalysisService(
        config=cfg,
        sentiment_adapter=_StubReadyAdapter(),
        preprocessor=VietnamesePreprocessor(),
        llm_helper=OptionalLLMHelper(cfg),
    )
    sample_text = svc.list_sample_cases()[0]["text"]
    outcome = svc.analyze(
        raw_text=sample_text,
        mode=AppMode.REAL,
        unit_code="ICTU",
        source_code="SURVEY_PORTAL",
        manual_topic_code="LECTURER",
        confirmed_tnu_real=True,  # Kể cả khi người dùng bấm tích, câu mẫu vẫn luôn là DEMO!
    )
    assert outcome.data_origin == DataOrigin.DEMO.value


def test_independent_identical_content_is_not_deleted_only_flagged(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = DemoFeedbackRepository(fixture_path=tmp_path / "empty.json")
    svc = AnalysisService(
        config=cfg,
        sentiment_adapter=_StubReadyAdapter(),
        preprocessor=VietnamesePreprocessor(),
        llm_helper=OptionalLLMHelper(cfg),
    )
    same_text = "Phòng học thoáng mát."
    out_a = svc.analyze(same_text, AppMode.DEMO, "ICTU", "SURVEY_PORTAL", "FACILITY")
    out_b = svc.analyze(same_text, AppMode.DEMO, "TNUT", "SURVEY_PORTAL", "FACILITY")

    save_a = svc.save_outcome(repo, same_text, out_a)
    save_b = svc.save_outcome(repo, same_text, out_b)

    assert save_a.saved is True
    assert save_a.is_content_duplicate_candidate is False
    assert save_b.saved is True  # Không bị tự động xóa chỉ vì trùng content_hash!
    assert save_b.is_content_duplicate_candidate is True

