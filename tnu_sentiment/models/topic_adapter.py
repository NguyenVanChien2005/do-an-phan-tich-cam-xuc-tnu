"""Adapter phân loại chủ đề: phản ánh trung thực trạng thái gán thủ công khi chưa có mô hình."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from tnu_sentiment.constants import INITIAL_TOPICS, USER_STATUS_LABELS
from tnu_sentiment.domain.models import ComponentHealth

VALID_TOPIC_CODES = {code for code, _name, _idx in INITIAL_TOPICS}


class ManualTopicAdapter:
    """Khi chưa có checkpoint phân loại chủ đề riêng, hệ thống cho phép chọn thủ công
    bằng mã chủ đề ổn định và hiển thị rõ 'Chủ đề được gán thủ công' (không sinh ngẫu nhiên).
    """

    def __init__(self, reason: str = "Chưa cấu hình checkpoint phân loại chủ đề riêng.") -> None:
        self._reason = reason

    @property
    def is_ready(self) -> bool:
        return False

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            ready=False,
            user_badge=USER_STATUS_LABELS["topic_manual"],
            technical_detail=(
                f"{self._reason} Chủ đề được chọn thủ công hoặc theo tệp Import bằng mã ổn định "
                "(LECTURER, CURRICULUM, FACILITY, OTHER_SUPPORT, UNCLASSIFIED)."
            ),
        )


def load_topic_adapter(
    checkpoint_dir: Optional[Path],
    model_version: str = "",
) -> ManualTopicAdapter:
    """Kiểm tra cấu hình mô hình chủ đề. Nếu chưa có hoặc không hợp lệ, dùng chế độ gán thủ công."""
    if checkpoint_dir is None or not str(checkpoint_dir).strip():
        return ManualTopicAdapter("Chưa cấu hình TOPIC_CHECKPOINT_DIR.")
    if not checkpoint_dir.exists():
        return ManualTopicAdapter(f"Không tìm thấy thư mục TOPIC_CHECKPOINT_DIR: {checkpoint_dir}")

    cfg_file = checkpoint_dir / "config.json"
    if not cfg_file.exists():
        return ManualTopicAdapter(f"Thiếu config.json trong TOPIC_CHECKPOINT_DIR: {checkpoint_dir}")

    try:
        with cfg_file.open("r", encoding="utf-8") as fh:
            cfg: Dict[str, object] = json.load(fh)
        id2label = cfg.get("id2label")
        if not isinstance(id2label, dict) or not set(str(v) for v in id2label.values()).issubset(
            VALID_TOPIC_CODES
        ):
            return ManualTopicAdapter(
                "Checkpoint chủ đề không khai báo ánh xạ id2label sang mã topic_code ổn định."
            )
    except Exception as exc:
        return ManualTopicAdapter(f"Lỗi kiểm tra cấu hình mô hình chủ đề: {exc}")

    return ManualTopicAdapter(
        f"Đã phát hiện cấu hình {model_version or checkpoint_dir.name}, "
        "nhưng hiện ưu tiên gán chủ đề thủ công để đảm bảo kiểm chứng."
    )

