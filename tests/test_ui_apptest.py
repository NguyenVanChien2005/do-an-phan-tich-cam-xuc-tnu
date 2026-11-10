"""Kiểm thử tích hợp cho src/preprocessor.py, src/llm_fallback.py, src/trainer_focal_loss.py và giao diện 4 Tab app.py."""

from __future__ import annotations

from pathlib import Path
import torch

from src.llm_fallback import call_gemini_LowConfidence_router, infer_topic_by_keywords
from src.preprocessor import TNUStudentFeedbackPreprocessor
from src.trainer_focal_loss import MultiClassFocalLoss


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
