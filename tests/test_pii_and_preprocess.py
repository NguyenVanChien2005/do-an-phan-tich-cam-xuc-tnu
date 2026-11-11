"""Kiểm thử che PII, tiền xử lý tiếng Việt và bảo vệ XSS trên HTML badge."""

from __future__ import annotations

import pytest

from tnu_sentiment.nlp.pii_masker import mask_pii
from tnu_sentiment.nlp.preprocessor import (
    SegmenterUnavailableError,
    VietnamesePreprocessor,
)
from tnu_sentiment.ui.theme import render_pill_html


def test_pii_masking_and_token_preservation_through_pipeline() -> None:
    raw = (
        "Em là sv mã số DTC990012345, sđt 0912 345 678, "
        "email dtc990012345@example.edu.vn xin phản ánh lịch thi."
    )
    masked, pii_types = mask_pii(raw)
    assert "<EMAIL>" in masked
    assert "<STUDENT_ID>" in masked
    assert "<PHONE>" in masked
    assert "dtc990012345" not in masked.lower()
    assert "0912" not in masked
    assert set(pii_types) == {"EMAIL", "STUDENT_ID", "PHONE"}

    prep = VietnamesePreprocessor()
    res = prep.run(raw)
    # Bước làm sạch và tách từ pyvi tuyệt đối không được phá các token PII
    assert "<EMAIL>" in res.processed_content
    assert "<STUDENT_ID>" in res.processed_content
    assert "<PHONE>" in res.processed_content
    assert "phtok" not in res.processed_content


def test_emoji_longest_match_and_no_forced_negative_emoticon() -> None:
    prep = VietnamesePreprocessor()
    happy_text = "Buổi seminar học thuật hôm nay vui quá :)))"
    res = prep.run(happy_text)
    assert "tiêu_cực" not in res.processed_content
    assert "mỉa_mai" not in res.processed_content
    # Không bị sót dấu ngoặc đơn dư '))' do khớp nhầm ':)' trước ':)))'
    assert ")" not in res.processed_content
    assert len(res.compatibility_notes) >= 1


def test_elongated_chars_do_not_corrupt_double_letter_words() -> None:
    prep = VietnamesePreprocessor()
    res = prep.run("Thầy dạy môn oop trên moodle siêuuuu dễ hỉu 🥰")
    assert "siêu" in res.processed_content
    assert "siêuuuu" not in res.processed_content
    assert "oop" in res.processed_content
    assert "moodle" in res.processed_content
    assert "hiểu" in res.processed_content
    assert "tích_cực" in res.processed_content


def test_multi_aspect_detection() -> None:
    prep = VietnamesePreprocessor()
    res = prep.run("Giảng viên dạy rất nhiệt tình nhưng máy chiếu phòng học mờ quá.")
    assert res.is_multi_aspect is True


def test_segmenter_failure_not_silent_when_required() -> None:
    def _broken_segmenter(_txt: str) -> str:
        raise SegmenterUnavailableError("Segmenter hỏng")

    prep = VietnamesePreprocessor(segmenter_fn=_broken_segmenter, require_segmenter=True)
    with pytest.raises(SegmenterUnavailableError):
        prep.run("Giảng viên nhiệt tình.")


def test_html_pill_escapes_untrusted_text() -> None:
    malicious = '<img src=x onerror="alert(1)">'
    rendered = render_pill_html(malicious, "#FFFFFF", "#000000")
    assert "<img" not in rendered
    assert "&lt;img" in rendered


def test_rebuild_clean_splits_zero_leakage_and_preprocess_sync() -> None:
    """Bước 3.1: Kiểm tra rebuild_clean_splits chạy lại VietnamesePreprocessor.preprocess(),
    xóa bỏ hoàn toàn rò rỉ train/val (0 overlap) kể cả biến thể cùng template gốc, giữ nguyên test 3.166 dòng.
    """
    from scripts.rebuild_clean_splits import compute_template_group_key, rebuild_clean_splits

    # Kiểm tra gom nhóm biến thể từ cùng 1 template miền ĐHTN (chỉ đổi mã trường ICTU vs TNUT vs DTC)
    k1 = compute_template_group_key(
        "Giảng viên trường ictu dạy thực hành nhiệt tình 👍.",
        "giảng_viên trường đại_học công_nghệ_thông_tin và truyền_thông dạy thực_hành nhiệt_tình tích_cực_tốt .",
    )
    k2 = compute_template_group_key(
        "Giảng viên trường tnut dạy thực hành nhiệt tình 👍.",
        "giảng_viên trường đại_học kỹ_thuật công_nghiệp dạy thực_hành nhiệt_tình tích_cực_tốt .",
    )
    assert k1 == k2

    report = rebuild_clean_splits(write_files=False)
    assert report["preprocessing_drift_fixed"]["train"] == 2804
    assert report["preprocessing_drift_fixed"]["val"] == 384
    assert report["preprocessing_drift_fixed"]["test"] == 608
    assert report["initial_train_val_processed_leakage_rows"] == 124
    assert report["after_clean_overlap"]["raw_content_overlap"] == 0
    assert report["after_clean_overlap"]["processed_content_overlap"] == 0
    assert report["after_clean_overlap"]["group_key_overlap"] == 0
    assert report["clean_counts"]["test"] == 3166
    assert report["test_near_duplicate_audit"]["near_duplicate_count"] == 0


def test_evaluate_checkpoint_from_raw_option() -> None:
    """Bước 3.2: Kiểm tra tùy chọn --from-raw của evaluate_checkpoint.py đưa raw_content qua VietnamesePreprocessor."""
    import pandas as pd
    from scripts.evaluate_checkpoint import parse_eval_args, prepare_evaluation_texts

    args = parse_eval_args(["--from-raw"])
    assert args.from_raw is True

    sample_df = pd.DataFrame(
        {
            "raw_content": ["Thầy dạy môn lập trình Python ở ictu siêuuuu dễ hỉu 🥰."],
            "processed_content": ["cũ_chưa_chuẩn_hóa"],
            "sentiment": [2],
        }
    )
    texts_default = prepare_evaluation_texts(sample_df, from_raw=False)
    assert texts_default == ["cũ_chưa_chuẩn_hóa"]

    texts_from_raw = prepare_evaluation_texts(sample_df, from_raw=True)
    assert texts_from_raw[0] != "cũ_chưa_chuẩn_hóa"
    assert "hiểu" in texts_from_raw[0]
    assert "tích_cực" in texts_from_raw[0]
