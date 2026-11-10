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

