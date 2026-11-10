"""Dịch vụ kiểm duyệt phản hồi: lưu theo ID bản ghi, giữ nguyên dự đoán gốc và ghi lịch sử."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from tnu_sentiment.constants import ReviewStatus
from tnu_sentiment.domain.models import (
    FeedbackRecord,
    ReviewLogEntry,
    ReviewUpdateRequest,
)
from tnu_sentiment.nlp.pii_masker import mask_pii
from tnu_sentiment.repositories.base import FeedbackRepository


class ReviewService:
    """Xử lý kiểm duyệt phản hồi theo đúng khóa định danh feedback_id."""

    @staticmethod
    def build_diff_requests(
        original_records: List[FeedbackRecord],
        edited_rows: List[Dict[str, object]],
        reviewer_name: str,
    ) -> Tuple[List[ReviewUpdateRequest], List[str]]:
        """Đối chiếu bảng chỉnh sửa theo feedback_id (không dựa vào thứ tự dòng sau sắp xếp).
        Chỉ tạo yêu cầu cập nhật cho các bản ghi thực sự có thay đổi ở cột kiểm duyệt.
        """
        reviewer_clean, _ = mask_pii((reviewer_name or "").strip())
        orig_by_id: Dict[int, FeedbackRecord] = {
            r.feedback_id: r for r in original_records
        }

        requests: List[ReviewUpdateRequest] = []
        validation_errors: List[str] = []

        for row in edited_rows:
            raw_id = row.get("feedback_id")
            if raw_id is None:
                continue
            fid = int(raw_id)
            orig = orig_by_id.get(fid)
            if orig is None:
                validation_errors.append(f"Không tìm thấy bản ghi gốc cho ID #{fid}.")
                continue

            new_status = str(row.get("review_status") or orig.review_status).strip()
            if new_status not in {s.value for s in ReviewStatus}:
                validation_errors.append(
                    f"ID #{fid}: trạng thái duyệt '{new_status}' không hợp lệ."
                )
                continue

            raw_sent = row.get("reviewed_sentiment_label")
            if raw_sent is None or str(raw_sent).strip() in ("", "None", "nan", "Chưa chọn"):
                new_sent: Optional[int] = None
            else:
                new_sent = int(raw_sent)

            raw_topic = row.get("reviewed_topic_code")
            if raw_topic is None or str(raw_topic).strip() in ("", "None", "nan", "Chưa chọn"):
                new_topic: Optional[str] = None
            else:
                new_topic = str(raw_topic).strip()

            raw_note = row.get("review_note")
            new_note = (
                str(raw_note).strip()
                if raw_note is not None and str(raw_note).strip() not in ("", "None", "nan")
                else None
            )

            changed = (
                new_status != orig.review_status
                or new_sent != orig.reviewed_sentiment_label
                or new_topic != orig.reviewed_topic_code
                or (new_note or None) != (orig.review_note or None)
            )
            if not changed:
                continue

            if not reviewer_clean:
                validation_errors.append(
                    "Vui lòng nhập tên người duyệt (khai báo thủ công) trước khi lưu thay đổi kiểm duyệt."
                )
                break

            if new_status == ReviewStatus.VERIFIED.value and new_sent not in (0, 1, 2):
                # Nếu người duyệt chuyển sang VERIFIED mà chưa chọn nhãn mới, cần chọn tường minh
                validation_errors.append(
                    f"ID #{fid}: khi đánh dấu 'Đã xác nhận (VERIFIED)', vui lòng chọn nhãn cảm xúc được duyệt."
                )
                continue

            requests.append(
                ReviewUpdateRequest(
                    feedback_id=fid,
                    expected_version=orig.record_version,
                    new_review_status=new_status,
                    reviewed_sentiment_label=new_sent,
                    reviewed_topic_code=new_topic,
                    reviewed_by=reviewer_clean,
                    review_note=new_note,
                )
            )

        return requests, validation_errors

    @staticmethod
    def save_reviews(
        repository: FeedbackRepository,
        updates: List[ReviewUpdateRequest],
    ) -> Tuple[int, List[str]]:
        if not updates:
            return 0, []
        return repository.apply_review_updates(updates)

    @staticmethod
    def get_recent_logs(
        repository: FeedbackRepository,
        feedback_id: Optional[int] = None,
        limit: int = 50,
    ) -> List[ReviewLogEntry]:
        return repository.list_review_logs(feedback_id=feedback_id, limit=limit)

