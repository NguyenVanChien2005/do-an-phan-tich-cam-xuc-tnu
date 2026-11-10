"""Kiểm thử kiểm duyệt theo ID (sau khi sắp xếp lại thứ tự dòng), giữ nguyên dự đoán gốc,
kiểm soát phiên bản đồng thời và không lộ bí mật/PII trong export."""

from __future__ import annotations

from pathlib import Path

from tnu_sentiment.constants import DataOrigin, ReviewStatus
from tnu_sentiment.domain.models import FilterCriteria, ReviewUpdateRequest
from tnu_sentiment.repositories.demo_repo import DemoFeedbackRepository
from tnu_sentiment.services.review_service import ReviewService


def test_review_saves_by_id_without_overwriting_original_prediction() -> None:
    fixture_file = Path("fixtures/demo_feedbacks.json").resolve()
    repo = DemoFeedbackRepository(fixture_path=fixture_file)

    flt = FilterCriteria(data_origins=[DataOrigin.DEMO.value])
    # Lấy trang dữ liệu sắp xếp confidence tăng dần (thứ tự dòng khác hoàn toàn thứ tự ID!)
    paged = repo.list_paginated(flt, page=1, page_size=5, sort_by="confidence_asc")
    target = paged.items[0]
    orig_pred_sent = target.predicted_sentiment_label
    orig_pred_topic = target.predicted_topic_code
    orig_ver = target.record_version

    # Đảo ngược danh sách edited_rows để chứng minh việc lưu dựa vào feedback_id chứ không phải vị trí dòng!
    edited_rows = [
        {
            "feedback_id": r.feedback_id,
            "review_status": (
                ReviewStatus.VERIFIED.value
                if r.feedback_id == target.feedback_id
                else r.review_status
            ),
            "reviewed_sentiment_label": (
                2 if r.feedback_id == target.feedback_id else r.reviewed_sentiment_label
            ),
            "reviewed_topic_code": (
                "CURRICULUM"
                if r.feedback_id == target.feedback_id
                else r.reviewed_topic_code
            ),
            "review_note": (
                "Ghi chú của sv sđt 0901234567 đã được rà soát"
                if r.feedback_id == target.feedback_id
                else r.review_note
            ),
        }
        for r in reversed(paged.items)
    ]

    reqs, errs = ReviewService.build_diff_requests(
        original_records=paged.items,
        edited_rows=edited_rows,
        reviewer_name="nguyenvanchien_reviewer",
    )
    assert errs == []
    assert len(reqs) == 1
    assert reqs[0].feedback_id == target.feedback_id

    applied, save_errs = ReviewService.save_reviews(repo, reqs)
    assert applied == 1
    assert save_errs == []

    updated = repo.get_by_id(target.feedback_id)
    assert updated is not None
    # Dự đoán gốc của mô hình TUYỆT ĐỐI giữ nguyên!
    assert updated.predicted_sentiment_label == orig_pred_sent
    assert updated.predicted_topic_code == orig_pred_topic
    # Nhãn kiểm duyệt mới đã được lưu riêng kèm tăng record_version
    assert updated.review_status == ReviewStatus.VERIFIED.value
    assert updated.reviewed_sentiment_label == 2
    assert updated.reviewed_topic_code == "CURRICULUM"
    assert updated.record_version == orig_ver + 1
    # Ghi chú duyệt cũng được tự động che PII!
    assert "0901234567" not in (updated.review_note or "")
    assert "<PHONE>" in (updated.review_note or "")

    # Kiểm tra xung đột cập nhật đồng thời (stale expected_version)
    stale_req = ReviewUpdateRequest(
        feedback_id=target.feedback_id,
        expected_version=orig_ver,  # Phiên bản cũ!
        new_review_status=ReviewStatus.NEEDS_REVIEW.value,
        reviewed_sentiment_label=None,
        reviewed_topic_code=None,
        reviewed_by="other_reviewer",
        review_note="Cố ghi đè bằng version cũ",
    )
    applied_2, errs_2 = ReviewService.save_reviews(repo, [stale_req])
    assert applied_2 == 0
    assert len(errs_2) == 1

    # Kiểm tra lịch sử kiểm duyệt & trạng thái chưa xác thực danh tính
    logs = ReviewService.get_recent_logs(repo, feedback_id=target.feedback_id)
    assert len(logs) == 1
    assert logs[0].is_authenticated_user is False

    # Kiểm tra xuất dữ liệu không chứa author_hash hay secret
    exported = repo.export_records(flt, limit=50)
    assert len(exported) > 0
    for row in exported:
        assert "author_hash" not in row
        assert "secret_salt" not in row
