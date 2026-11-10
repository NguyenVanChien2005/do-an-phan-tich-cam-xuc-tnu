"""Adapter mô hình phân tích cảm xúc PhoBERT với kiểm tra nghiêm ngặt tính hợp lệ của checkpoint."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from tnu_sentiment.constants import (
    SENTIMENT_ID_TO_CODE,
    SENTIMENT_ID_TO_VI,
    USER_STATUS_LABELS,
)
from tnu_sentiment.domain.models import ComponentHealth, SentimentPrediction


class ModelNotReadyError(RuntimeError):
    """Ném ra khi gọi suy luận thật nhưng chưa có checkpoint hợp lệ."""


class BaseSentimentAdapter(ABC):
    @property
    @abstractmethod
    def is_ready(self) -> bool: ...

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @abstractmethod
    def health(self) -> ComponentHealth: ...

    @abstractmethod
    def predict(self, processed_text: str) -> SentimentPrediction: ...


class UnavailableSentimentAdapter(BaseSentimentAdapter):
    """Adapter đại diện trạng thái thiếu hoặc lỗi checkpoint — tuyệt đối không sinh nhãn giả."""

    def __init__(self, reason: str, configured_version: str = "unavailable") -> None:
        self._reason = reason
        self._version = configured_version

    @property
    def is_ready(self) -> bool:
        return False

    @property
    def model_version(self) -> str:
        return self._version

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            ready=False,
            user_badge=USER_STATUS_LABELS["sentiment_missing"],
            technical_detail=self._reason,
        )

    def predict(self, processed_text: str) -> SentimentPrediction:
        raise ModelNotReadyError(self._reason)


def validate_sentiment_checkpoint_dir(ckpt_dir: Optional[Path]) -> None:
    """Kiểm tra cấu trúc tệp và siêu dữ liệu config.json trước khi nạp trọng số.
    Tuyệt đối từ chối base model chưa có classification head hoặc sai ánh xạ nhãn.
    """
    if ckpt_dir is None:
        raise ValueError("Chưa cấu hình đường dẫn SENTIMENT_CHECKPOINT_DIR.")
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Thư mục checkpoint không tồn tại: {ckpt_dir}")

    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Thiếu tệp config.json trong checkpoint: {ckpt_dir}")

    has_weights = (ckpt_dir / "model.safetensors").exists() or (
        ckpt_dir / "pytorch_model.bin"
    ).exists()
    if not has_weights:
        raise FileNotFoundError(
            f"Thiếu tệp trọng số (model.safetensors hoặc pytorch_model.bin) tại: {ckpt_dir}"
        )

    has_tokenizer = (
        (ckpt_dir / "vocab.txt").exists() and (ckpt_dir / "bpe.codes").exists()
    ) or (ckpt_dir / "tokenizer.json").exists()
    if not has_tokenizer:
        raise FileNotFoundError(
            f"Thiếu tệp từ điển tokenizer (vocab.txt + bpe.codes hoặc tokenizer.json) tại: {ckpt_dir}"
        )

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    architectures = cfg.get("architectures") or []
    if not any("SequenceClassification" in str(arch) for arch in architectures):
        raise ValueError(
            f"Checkpoint không phải mô hình phân loại chuỗi đã fine-tune (architectures={architectures}). "
            "Hệ thống từ chối nạp base model với classification head khởi tạo ngẫu nhiên."
        )

    id2label_raw = cfg.get("id2label") or {}
    expected = {"0": "Negative", "1": "Neutral", "2": "Positive"}
    normalized_map = {str(k): str(v) for k, v in id2label_raw.items()}
    if normalized_map != expected:
        raise ValueError(
            f"Ánh xạ id2label trong checkpoint ({normalized_map}) không khớp quy ước chuẩn {expected}."
        )


class PhoBertSentimentAdapter(BaseSentimentAdapter):
    """Adapter suy luận thực tế bằng checkpoint PhoBERT đã fine-tune."""

    def __init__(
        self,
        checkpoint_dir: Path,
        model_version: str = "PhoBERT-base-v2-Focal",
        max_length: int = 256,
    ) -> None:
        validate_sentiment_checkpoint_dir(checkpoint_dir)

        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._checkpoint_dir = checkpoint_dir
        self._model_version = model_version

        cfg = AutoConfig.from_pretrained(str(checkpoint_dir), local_files_only=True)
        # Với RoBERTa/PhoBERT (pad_token_id=1), giới hạn token tối đa an toàn là max_position_embeddings - 2
        max_pos = int(getattr(cfg, "max_position_embeddings", 258))
        pad_id = int(getattr(cfg, "pad_token_id", 1))
        model_cap = max_pos - pad_id - 1 if max_pos > 4 else 256
        self._max_length = min(max(16, max_length), model_cap)

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), local_files_only=True
        )
        model_loaded, loading_info = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_dir),
            local_files_only=True,
            output_loading_info=True,
        )
        missing_keys = loading_info.get("missing_keys") or []
        if missing_keys:
            raise ValueError(
                f"Checkpoint thiếu trọng số đã huấn luyện ({missing_keys}). "
                "Từ chối chạy với tầng phân loại khởi tạo mới."
            )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = model_loaded.to(self._device)
        self._model.eval()

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return self._model_version

    def health(self) -> ComponentHealth:
        return ComponentHealth(
            ready=True,
            user_badge=USER_STATUS_LABELS["sentiment_ready"],
            technical_detail=(
                f"Đã nạp checkpoint hợp lệ: {self._checkpoint_dir.name} "
                f"(version={self._model_version}, device={self._device}, max_length={self._max_length})"
            ),
        )

    def predict(self, processed_text: str) -> SentimentPrediction:
        if not processed_text or not processed_text.strip():
            raise ValueError("Văn bản sau tiền xử lý rỗng, không thể suy luận.")

        t0 = time.perf_counter()

        # Kiểm tra độ dài token thực tế trước khi cắt (truncation)
        raw_ids = self._tokenizer.encode(
            processed_text, add_special_tokens=True, truncation=False
        )
        full_token_count = len(raw_ids)
        was_truncated = full_token_count > self._max_length

        encoded = self._tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=False,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with self._torch.no_grad():
            outputs = self._model(**encoded)
            probs_tensor = self._torch.softmax(outputs.logits, dim=-1)[0]
            probs_list = [float(x) for x in probs_tensor.detach().cpu().tolist()]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        probabilities: Dict[int, float] = {
            0: round(probs_list[0], 6),
            1: round(probs_list[1], 6),
            2: round(probs_list[2], 6),
        }
        pred_id = max(probabilities, key=lambda k: probabilities[k])
        confidence = probabilities[pred_id]

        return SentimentPrediction(
            label_id=pred_id,
            label_code=SENTIMENT_ID_TO_CODE[pred_id],
            label_vi=SENTIMENT_ID_TO_VI[pred_id],
            confidence=confidence,
            probabilities=probabilities,
            model_version=self._model_version,
            was_truncated=was_truncated,
            token_count=full_token_count,
            max_length=self._max_length,
            inference_time_ms=round(elapsed_ms, 2),
            measurement_scope=(
                f"Tokenize + PhoBERT forward pass trên {self._device.upper()} "
                "(không bao gồm thời gian khởi tạo/nạp checkpoint ban đầu)"
            ),
            is_illustrative_demo=False,
        )


def load_sentiment_adapter(
    checkpoint_dir: Optional[Path],
    model_version: str,
    max_length: int,
) -> BaseSentimentAdapter:
    """Khởi tạo adapter cảm xúc an toàn; trả về UnavailableSentimentAdapter nếu thiếu/lỗi."""
    try:
        if checkpoint_dir is None:
            return UnavailableSentimentAdapter(
                "Chưa cấu hình SENTIMENT_CHECKPOINT_DIR trong môi trường.",
                configured_version=model_version,
            )
        return PhoBertSentimentAdapter(
            checkpoint_dir=checkpoint_dir,
            model_version=model_version,
            max_length=max_length,
        )
    except Exception as exc:
        return UnavailableSentimentAdapter(
            f"Không thể nạp checkpoint cảm xúc: {exc}",
            configured_version=model_version,
        )

