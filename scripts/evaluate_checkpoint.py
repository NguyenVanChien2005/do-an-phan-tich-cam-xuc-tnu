"""Script đánh giá thực nghiệm checkpoint thực tế trên tập Validation (1.709 mẫu) và Test (3.166 mẫu).
Sinh các tệp artifact JSON đã xác thực (kèm mã băm SHA-256 của tập đánh giá) tại artifacts/evaluation/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tnu_sentiment.constants import SENTIMENT_ID_TO_VI
from tnu_sentiment.nlp.pii_masker import mask_pii
from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor


def prepare_evaluation_texts(
    df: pd.DataFrame,
    from_raw: bool = False,
    preprocessor: Optional[VietnamesePreprocessor] = None,
) -> List[str]:
    """Chuẩn bị danh sách văn bản đầu vào cho mô hình.
    Nếu from_raw=True: đưa raw_content qua VietnamesePreprocessor.preprocess() để đo chất lượng end-to-end thực tế.
    """
    if from_raw:
        prep = preprocessor or VietnamesePreprocessor()
        source_col = "raw_content" if "raw_content" in df.columns else "sentence"
        return [prep.preprocess(str(txt)) for txt in df[source_col].astype(str).tolist()]
    return df["processed_content"].astype(str).tolist()


def parse_eval_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá thực nghiệm checkpoint PhoBERT-v2 và TF-IDF Baseline trên Validation & Test."
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Đưa raw_content qua VietnamesePreprocessor trước khi dự đoán để đo chất lượng thực tế của ứng dụng.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "TNU_Sentiment_Dataset_Final",
        help="Thư mục chứa tnu_train_final.csv, tnu_val_final.csv, tnu_test_final.csv.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "phobert_tnu_best_model-20261101T002214Z-1-001" / "phobert_tnu_best_model",
        help="Thư mục chứa trọng số checkpoint PhoBERT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "evaluation",
        help="Thư mục lưu kết quả JSON đánh giá.",
    )
    return parser.parse_args(argv)


def compute_eval_set_id(df: pd.DataFrame, split_name: str, label_col: str) -> str:
    """Tính định danh duy nhất (SHA-256 rút gọn) từ toàn bộ cặp (nội dung, nhãn chuẩn) của tập đánh giá."""
    hasher = hashlib.sha256()
    hasher.update(f"{split_name}|{label_col}|{len(df)}\n".encode("utf-8"))
    for idx, row in df.iterrows():
        txt = str(row["processed_content"]).strip()
        lbl = int(row[label_col])
        hasher.update(f"{idx}:{lbl}:{txt}\n".encode("utf-8"))
    return f"{split_name.lower()}-{label_col}-sha256-{hasher.hexdigest()[:16]}"


def categorize_error(raw_text: str, true_lbl: int, pred_lbl: int) -> tuple[str, str]:
    low = raw_text.lower()
    if any(k in low for k in ("nhưng", "tuy nhiên", "mặc dù", "tuy ", "dù ")):
        return (
            "Câu đa ý / Tương phản",
            "Câu chứa mệnh đề khen - chê đan xen; mô hình phân loại toàn câu dễ nghiêng về vế sau.",
        )
    if true_lbl == 1 or pred_lbl == 1:
        return (
            "Ranh giới Trung tính",
            "Câu góp ý nhẹ hoặc thông báo thiếu từ khóa cảm xúc mạnh dễ nhầm giữa Trung tính và Tích cực/Tiêu cực.",
        )
    if len(raw_text.split()) <= 5:
        return (
            "Ngữ cảnh ngắn",
            "Phản hồi rất ngắn, thiếu ngữ cảnh bổ trợ dẫn tới dự đoán lệch cực cảm xúc.",
        )
    return (
        "Ngữ nghĩa gián tiếp / Hàm ý",
        "Biểu đạt nhận xét gián tiếp hoặc từ vựng chuyên biệt trong học đường.",
    )


def build_sentiment_artifact(
    run_id: str,
    model_name: str,
    checkpoint_version: str,
    dataset_version: str,
    split: str,
    eval_set_id: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray | None,
    notes: str,
) -> Dict[str, object]:
    labels = [0, 1, 2]
    label_names = ["Tiêu cực (0)", "Trung tính (1)", "Tích cực (2)"]

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p = float(np.mean(prec))
    macro_r = float(np.mean(rec))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = [
        {
            "class_id": str(lbl),
            "class_name": label_names[i],
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i]),
        }
        for i, lbl in enumerate(labels)
    ]

    # Trích xuất tối đa 25 ca lỗi thực tế có nhãn chuẩn (đã che PII)
    mis_indices = np.where(y_true != y_pred)[0]
    if confidences is not None and len(mis_indices) > 0:
        # Ưu tiên hiển thị các ca dự đoán sai có confidence cao để phân tích lỗi học thuật
        mis_indices = sorted(
            mis_indices, key=lambda idx: float(confidences[idx]), reverse=True
        )

    error_cases: List[Dict[str, object]] = []
    for idx in mis_indices[:25]:
        raw_txt = str(df.iloc[int(idx)]["raw_content"])
        safe_txt, _ = mask_pii(raw_txt)
        t_lbl = int(y_true[int(idx)])
        p_lbl = int(y_pred[int(idx)])
        cat, note = categorize_error(safe_txt, t_lbl, p_lbl)
        error_cases.append(
            {
                "sample_id": f"{split.upper()}-{int(idx) + 1:04d}",
                "masked_text": safe_txt,
                "true_label": f"{SENTIMENT_ID_TO_VI[t_lbl]} ({t_lbl})",
                "predicted_label": f"{SENTIMENT_ID_TO_VI[p_lbl]} ({p_lbl})",
                "confidence": (
                    round(float(confidences[int(idx)]), 4)
                    if confidences is not None
                    else None
                ),
                "error_category": cat,
                "analysis_note": note,
            }
        )

    now_iso = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%Y-%m-%d %H:%M:%S %z")
    return {
        "run_id": run_id,
        "task": "SENTIMENT",
        "model_name": model_name,
        "checkpoint_version": checkpoint_version,
        "dataset_version": dataset_version,
        "split": split,
        "evaluation_set_id": eval_set_id,
        "metric_definition_version": "sklearn-multiclass-v1",
        "n_samples": int(len(df)),
        "evaluated_at": now_iso,
        "result_source": "ARTIFACT_VERIFIED",
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_metrics": per_class,
        "confusion_matrix_labels": label_names,
        "confusion_matrix": cm,
        "error_cases": error_cases,
        "notes": notes,
    }


def predict_phobert_batch(
    ckpt_dir: Path, texts: List[str], batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir), local_files_only=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds: List[int] = []
    confs: List[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            batch_pred = np.argmax(probs, axis=-1)
            batch_conf = np.max(probs, axis=-1)
            preds.extend(batch_pred.tolist())
            confs.extend(batch_conf.tolist())

    return np.array(preds, dtype=int), np.array(confs, dtype=float)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_eval_args(argv)
    ds_dir = args.dataset_dir
    ckpt_dir = args.checkpoint_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(ds_dir / "tnu_train_final.csv")
    val_df = pd.read_csv(ds_dir / "tnu_val_final.csv")
    test_df = pd.read_csv(ds_dir / "tnu_test_final.csv")

    preprocessor = VietnamesePreprocessor() if args.from_raw else None
    train_texts = prepare_evaluation_texts(train_df, from_raw=args.from_raw, preprocessor=preprocessor)
    val_texts = prepare_evaluation_texts(val_df, from_raw=args.from_raw, preprocessor=preprocessor)
    test_texts = prepare_evaluation_texts(test_df, from_raw=args.from_raw, preprocessor=preprocessor)

    val_eval_df = val_df.copy()
    test_eval_df = test_df.copy()
    val_eval_df["processed_content"] = val_texts
    test_eval_df["processed_content"] = test_texts

    val_set_id = compute_eval_set_id(val_eval_df, "Validation", "sentiment")
    test_set_id = compute_eval_set_id(test_eval_df, "Test", "sentiment")

    mode_label = "from-raw (VietnamesePreprocessor)" if args.from_raw else "pre-tokenized processed_content"
    print(f"Evaluating PhoBERT ({mode_label}) on Validation ({len(val_df)}) & Test ({len(test_df)})...")
    val_preds, val_confs = predict_phobert_batch(ckpt_dir, val_texts)
    test_preds, test_confs = predict_phobert_batch(ckpt_dir, test_texts)

    phobert_val = build_sentiment_artifact(
        run_id="phobert-focal-val-v1",
        model_name="PhoBERT-base-v2 (Fine-tuned)",
        checkpoint_version="phobert_tnu_best_model",
        dataset_version="TNU_Sentiment_Dataset_Final_v1",
        split="Validation",
        eval_set_id=val_set_id,
        df=val_eval_df,
        y_true=val_df["sentiment"].to_numpy(dtype=int),
        y_pred=val_preds,
        confidences=val_confs,
        notes=f"Đánh giá trực tiếp từ checkpoint phobert_tnu_best_model trên tnu_val_final.csv ({mode_label}).",
    )
    phobert_test = build_sentiment_artifact(
        run_id="phobert-focal-test-v1",
        model_name="PhoBERT-base-v2 (Fine-tuned)",
        checkpoint_version="phobert_tnu_best_model",
        dataset_version="TNU_Sentiment_Dataset_Final_v1",
        split="Test",
        eval_set_id=test_set_id,
        df=test_eval_df,
        y_true=test_df["sentiment"].to_numpy(dtype=int),
        y_pred=test_preds,
        confidences=test_confs,
        notes=f"Đánh giá trực tiếp từ checkpoint phobert_tnu_best_model trên tnu_test_final.csv ({mode_label}).",
    )

    print("Evaluating TF-IDF + LogisticRegression baseline on identical evaluation sets...")
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
    X_train = vec.fit_transform(train_texts)
    y_train = train_df["sentiment"].to_numpy(dtype=int)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    val_prob_lr = clf.predict_proba(vec.transform(val_texts))
    test_prob_lr = clf.predict_proba(vec.transform(test_texts))

    tfidf_val = build_sentiment_artifact(
        run_id="tfidf-logreg-val-v1",
        model_name="TF-IDF + Logistic Regression (Baseline)",
        checkpoint_version="sklearn-tfidf-lr-balanced-v1",
        dataset_version="TNU_Sentiment_Dataset_Final_v1",
        split="Validation",
        eval_set_id=val_set_id,
        df=val_eval_df,
        y_true=val_df["sentiment"].to_numpy(dtype=int),
        y_pred=np.argmax(val_prob_lr, axis=-1),
        confidences=np.max(val_prob_lr, axis=-1),
        notes=f"Mô hình cơ sở (Baseline) huấn luyện trên tnu_train_final.csv, đánh giá cùng tập tnu_val_final.csv ({mode_label}).",
    )
    tfidf_test = build_sentiment_artifact(
        run_id="tfidf-logreg-test-v1",
        model_name="TF-IDF + Logistic Regression (Baseline)",
        checkpoint_version="sklearn-tfidf-lr-balanced-v1",
        dataset_version="TNU_Sentiment_Dataset_Final_v1",
        split="Test",
        eval_set_id=test_set_id,
        df=test_eval_df,
        y_true=test_df["sentiment"].to_numpy(dtype=int),
        y_pred=np.argmax(test_prob_lr, axis=-1),
        confidences=np.max(test_prob_lr, axis=-1),
        notes=f"Mô hình cơ sở (Baseline) huấn luyện trên tnu_train_final.csv, đánh giá cùng tập tnu_test_final.csv ({mode_label}).",
    )

    artifacts = [
        ("phobert_val_verified.json", phobert_val),
        ("phobert_test_verified.json", phobert_test),
        ("tfidf_baseline_val_verified.json", tfidf_val),
        ("tfidf_baseline_test_verified.json", tfidf_test),
    ]
    for fn, payload in artifacts:
        with (out_dir / fn).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(
            f"Saved {fn}: split={payload['split']} n={payload['n_samples']} "
            f"acc={payload['accuracy']} macro_f1={payload['macro_f1']}"
        )


if __name__ == "__main__":
    main()

