"""Module huấn luyện tinh chỉnh PhoBERT-base-v2 với hàm mất mát Multi-Class Focal Loss
(gamma = 2.0, label smoothing epsilon = 0.05) và hàm đánh giá độc lập trên tập Test (3.166 mẫu).

Đồ án tốt nghiệp: Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên
Sinh viên: Nguyễn Văn Chiến - MSSV: DTC235200079 - GVHD: TS. Hà Thị Thanh
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


class MultiClassFocalLoss(nn.Module):
    """Hàm mất mát Multi-Class Focal Loss kết hợp trọng số lớp alpha và làm mượt nhãn (Label Smoothing).

    Công thức:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    với gamma = 2.0 và label smoothing epsilon = 0.05 nhằm khắc phục mất cân bằng dữ liệu
    (đặc biệt là lớp Trung tính chỉ chiếm 4,32% tập dữ liệu gốc).
    """

    def __init__(
        self,
        alpha: Optional[Sequence[float] | torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        num_classes: int = 3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.num_classes = int(num_classes)
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.register_buffer("alpha", alpha.float())
            else:
                self.register_buffer("alpha", torch.tensor(list(alpha), dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets = targets.view(-1)
        pt = probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(dim=0, index=targets)
            focal_weight = alpha_t * focal_weight

        # Áp dụng Label Smoothing epsilon = 0.05
        if self.label_smoothing > 0.0:
            with torch.no_grad():
                smooth_targets = torch.full_like(
                    log_probs,
                    fill_value=self.label_smoothing / (self.num_classes - 1),
                )
                smooth_targets.scatter_(
                    dim=-1,
                    index=targets.unsqueeze(1),
                    value=1.0 - self.label_smoothing,
                )
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction="none")

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLossPhoBERTTrainer:
    """Custom HuggingFace Trainer tích hợp MultiClassFocalLoss (gamma=2.0, epsilon=0.05)."""

    def __init__(
        self,
        *args: Any,
        class_weights: Optional[Sequence[float] | torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        **kwargs: Any,
    ) -> None:
        from transformers import Trainer

        self._base_trainer = Trainer(*args, **kwargs)
        self.focal_loss_fn = MultiClassFocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
            num_classes=3,
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_classification_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Hàm tính bộ chỉ số đánh giá đa lớp (tối ưu theo tiêu chí metric_for_best_model = 'macro_f1')."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    per_cls_f1 = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(weighted_f1, 4),
        "f1_negative": round(float(per_cls_f1[0]), 4),
        "f1_neutral": round(float(per_cls_f1[1]), 4),
        "f1_positive": round(float(per_cls_f1[2]), 4),
    }


def evaluate_checkpoint_on_test_split(
    checkpoint_dir: str | Path,
    test_csv_path: str | Path,
    batch_size: int = 32,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Nạp checkpoint tốt nhất (Epoch 4) và chạy đánh giá độc lập trên 3.166 câu Test."""
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt_path = Path(checkpoint_dir)
    df_test = pd.read_csv(test_csv_path)
    sentences = df_test["sentence"].astype(str).tolist()
    y_true = df_test["sentiment"].astype(int).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_path), local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch_texts = sentences[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            all_logits.append(out.logits.detach().cpu().numpy())

    logits_np = np.vstack(all_logits)
    metrics = compute_classification_metrics((logits_np, y_true))
    metrics["n_samples"] = int(len(y_true))
    metrics["evaluation_split"] = "Test (3.166)"
    return metrics
