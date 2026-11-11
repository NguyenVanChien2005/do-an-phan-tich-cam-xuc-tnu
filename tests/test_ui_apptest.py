"""Kiểm thử tích hợp cho src/preprocessor.py, src/llm_fallback.py, src/trainer_focal_loss.py và giao diện 4 Tab app.py."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
from streamlit.testing.v1 import AppTest

from src.llm_fallback import call_gemini_LowConfidence_router, infer_topic_by_keywords
from src.preprocessor import TNUStudentFeedbackPreprocessor
from src.trainer_focal_loss import MultiClassFocalLoss

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PATH = ROOT_DIR / "app.py"
DEFAULT_CKPT_WEIGHTS = (
    ROOT_DIR
    / "phobert_tnu_best_model-20261101T002214Z-1-001"
    / "phobert_tnu_best_model"
    / "model.safetensors"
)


def test_tnu_student_feedback_preprocessor_six_steps_and_pii() -> None:
    prep = TNUStudentFeedbackPreprocessor()

    # 1. Kiểm thử ẩn danh PII (Mã SV DTC235200079, SĐT, Email)
    raw_pii = (
        "Em là sv mã số DTC235200079, sđt 0912345678, "
        "email chien.ictu@tnu.edu.vn xin phản ánh lịch thi."
    )
    redacted = prep.redact_pii_only(raw_pii)
    assert "<STUDENT_ID>" in redacted
    assert "<PHONE>" in redacted
    assert "<EMAIL>" in redacted
    assert "DTC235200079" not in redacted
    assert "0912345678" not in redacted
    assert "chien.ictu@tnu.edu.vn" not in redacted

    # 2. Kiểm thử thứ tự thay thế Emoji/Emoticon dài trước ngắn (:))) -> tiêu_cực_mỉa_mai, :(( -> tiêu_cực_rất_buồn)
    steps = prep.preprocess_with_steps(
        "Hệ thống đk tín chỉ mượt ghê :))) mà phòng học nóng quá :(("
    )
    assert set(steps.keys()) == {
        "step1_nfc_pii",
        "step2_emoji",
        "step3_elongated",
        "step4_5_teencode_tone",
        "step6_segmented",
    }
    assert "tiêu_cực_mỉa_mai" in steps["step2_emoji"]
    assert "tiêu_cực_rất_buồn" in steps["step2_emoji"]
    assert "đăng_ký" in steps["step6_segmented"] or "đăng ký" in steps["step4_5_teencode_tone"]


def test_llm_fallback_and_focal_loss_modules() -> None:
    tid, tname, _ = infer_topic_by_keywords("Máy chiếu phòng học và wifi bị hỏng")
    assert tid == 3
    assert "Cơ sở vật chất" in tname

    router_out = call_gemini_LowConfidence_router(
        masked_text="Hệ thống đăng ký tín chỉ mượt ghê :)))",
        phobert_label=2,
        phobert_confidence=0.52,
    )
    assert router_out["review_status"] == "NEEDS_REVIEW"
    assert router_out["suggested_sentiment_label"] == 0

    loss_fn = MultiClassFocalLoss(alpha=[1.0, 3.0, 1.0], gamma=2.0, label_smoothing=0.05)
    logits = torch.tensor([[2.1, -0.5, -1.0], [-0.8, 1.5, -0.2]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss_val = loss_fn(logits, targets)
    assert float(loss_val.item()) > 0.0


def test_apptest_change_unit_after_analyze_saves_new_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bước 1.2: Streamlit AppTest tái hiện kịch bản:
    Phân tích với ICTU -> Đổi selectbox sang TNUT -> Bấm Lưu -> Xác nhận bản ghi trong DB có unit_code == 'TNUT'.
    """
    monkeypatch.setenv("MYSQL_HOST", "")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("SENTIMENT_CHECKPOINT_DIR", "")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Uncaught exception on load: {at.exception}"

    # 1. Chọn đơn vị ICTU và nhập câu phản hồi mới -> bấm Phân tích
    at.selectbox(key="t1_unit_code").set_value("ICTU")
    at.text_area(key="t1_raw_text").input(
        "Phòng thực hành mạng máy tính tại tòa nhà C5 vừa được trang bị dàn máy mới rất mượt."
    )
    at.run()

    at.button(key="t1_analyze_btn").click()
    at.run()
    assert not at.exception, f"Analyze failed: {at.exception}"
    assert at.button(key="t1_save_btn").disabled is False

    # 2. Đổi selectbox đơn vị từ ICTU sang TNUT sau khi đã phân tích (không sửa văn bản)
    at.selectbox(key="t1_unit_code").set_value("TNUT")
    at.run()
    assert not at.exception
    # Nút Lưu vẫn mở vì văn bản gốc không thay đổi
    assert at.button(key="t1_save_btn").disabled is False

    # 3. Bấm Lưu vào CSDL -> xác nhận bản ghi mới nhất lưu unit_code == "TNUT"
    at.button(key="t1_save_btn").click()
    at.run()
    assert not at.exception, f"Save failed: {at.exception}"

    df_saved = at.session_state["offline_feedbacks_df"]
    matched = df_saved[
        df_saved["raw_content"].str.contains(
            "Phòng thực hành mạng máy tính tại tòa nhà C5 vừa được trang bị dàn máy mới",
            na=False,
        )
    ]
    assert len(matched) >= 1
    assert matched.iloc[0]["unit_code"] == "TNUT"


def test_analyze_page_component_change_unit_after_analyze_saves_new_unit(
    tmp_path: Path,
) -> None:
    """Bước 1.2 (kiểm chứng trực tiếp trên AnalysisService + DemoFeedbackRepository):
    Phân tích với ICTU -> Đổi sang TNUT khi gọi save_outcome -> Bản ghi trong DB có unit_code == 'TNUT'.
    """
    from tnu_sentiment.config import load_app_config
    from tnu_sentiment.constants import AppMode
    from tnu_sentiment.domain.models import ComponentHealth, SentimentPrediction
    from tnu_sentiment.models.llm_helper import OptionalLLMHelper
    from tnu_sentiment.models.sentiment_adapter import BaseSentimentAdapter
    from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor
    from tnu_sentiment.repositories.demo_repo import DemoFeedbackRepository
    from tnu_sentiment.services.analysis_service import AnalysisService

    class _FastStubAdapter(BaseSentimentAdapter):
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
                label_id=2,
                label_code="Positive",
                label_vi="Tích cực",
                confidence=0.95,
                probabilities={0: 0.02, 1: 0.03, 2: 0.95},
                model_version="stub-v1",
                was_truncated=False,
                token_count=7,
                max_length=256,
                inference_time_ms=4.0,
                measurement_scope="stub",
            )

    cfg = load_app_config()
    repo = DemoFeedbackRepository(tmp_path / "empty.json")
    svc = AnalysisService(cfg, _FastStubAdapter(), VietnamesePreprocessor(), OptionalLLMHelper(cfg))

    txt = "Giảng viên hướng dẫn đồ án nhiệt tình và chi tiết."
    outcome = svc.analyze(
        raw_text=txt,
        mode=AppMode.REAL,
        unit_code="ICTU",
        source_code="SURVEY_PORTAL",
        manual_topic_code="LECTURER",
        confirmed_tnu_real=False,
    )
    assert outcome.unit_code == "ICTU"

    # Sau khi phân tích, người dùng đổi đơn vị sang TNUT và tích xác nhận TNU_REAL trên UI rồi bấm Lưu
    save_res = svc.save_outcome(
        repository=repo,
        current_raw_text=txt,
        outcome=outcome,
        override_unit_code="TNUT",
        override_source_code="QA_BOX",
        override_topic_code="FACILITY",
        override_confirmed_tnu_real=True,
        mode=AppMode.REAL,
    )
    assert save_res.saved is True
    saved_rec = repo.get_by_id(save_res.feedback_id or 1)
    assert saved_rec is not None
    assert saved_rec.unit_code == "TNUT"
    assert saved_rec.source_code == "QA_BOX"
    assert saved_rec.predicted_topic_code == "FACILITY"
    assert saved_rec.data_origin == "TNU_REAL"


@pytest.mark.requires_checkpoint
def test_real_phobert_checkpoint_inference_when_available() -> None:
    """Bước 2.1: Bài test cần trọng số thật model.safetensors, tự động skip nếu máy mới chưa tải checkpoint."""
    if not DEFAULT_CKPT_WEIGHTS.exists():
        pytest.skip(
            "Thiếu file trọng số model.safetensors (515MB). Chạy python scripts/download_checkpoint.py để tải."
        )
    from tnu_sentiment.models.sentiment_adapter import PhoBertSentimentAdapter

    adapter = PhoBertSentimentAdapter(DEFAULT_CKPT_WEIGHTS.parent)
    pred = adapter.predict("giảng_viên dạy rất nhiệt_tình và dễ hiểu")
    assert pred.label_id in (0, 1, 2)
    assert 0.0 <= (pred.confidence or 0.0) <= 1.0
