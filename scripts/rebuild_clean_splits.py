"""Script chuẩn hóa lại tập dữ liệu train/val/test bằng VietnamesePreprocessor và khử rò rỉ (0 overlap) giữa train và val."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor

DEFAULT_DATASET_DIR = ROOT_DIR / "TNU_Sentiment_Dataset_Final"
DEFAULT_AUDIT_PATH = ROOT_DIR / "artifacts" / "evaluation" / "clean_splits_audit.json"

# Biểu thức chính quy nhận diện các mã/tên trường thành viên ĐHTN trong cả raw_content lẫn processed_content
_TNU_UNIT_CANON_RE = re.compile(
    r"(?:"
    r"trường\s+trường\s+đại_học\s+công_nghệ_thông_tin\s+và\s+truyền_thông|"
    r"trường\s+đại_học\s+công_nghệ_thông_tin\s+và\s+truyền_thông|"
    r"trường\s+trường\s+đại_học\s+kỹ_thuật\s+công_nghiệp|"
    r"trường\s+đại_học\s+kỹ_thuật\s+công_nghiệp|"
    r"trường\s+trường\s+đại_học\s+kinh_tế\s+và\s+quản_trị\s+kinh_doanh|"
    r"trường\s+đại_học\s+kinh_tế\s+và\s+quản_trị\s+kinh_doanh|"
    r"trường\s+trường\s+đại_học\s+sư_phạm|"
    r"trường\s+đại_học\s+sư_phạm|"
    r"trường\s+trường\s+đại_học\s+nông_lâm(?:_yếu)?|"
    r"trường\s+đại_học\s+nông_lâm(?:_yếu)?|"
    r"trường\s+trường\s+đại_học\s+y_dược|"
    r"trường\s+đại_học\s+y_dược|"
    r"trường\s+trường\s+đại_học\s+khoa_học|"
    r"trường\s+đại_học\s+khoa_học|"
    r"trường_trường\s+ngoại_ngữ|"
    r"trường\s+trường\s+ngoại_ngữ|"
    r"trường\s+ngoại_ngữ|"
    r"trường_khoa\s+quốc_tế|"
    r"trường\s+khoa\s+quốc_tế|"
    r"khoa\s+quốc_tế|"
    r"trường\s+đại_học\s+thái_nguyên|"
    r"đại_học\s+thái_nguyên|"
    r"\b(?:ictu|dtc|tnut|dtk|tueba|dte|tnue|dts|tuaf|dtn|tump|dty|tnus|dtz|sfl|dtf|istnu|đhtn|dhtn)\b"
    r")",
    flags=re.IGNORECASE,
)


def compute_template_group_key(raw_content: str, processed_content: str) -> str:
    """Sinh group_key chuẩn hóa để gom các biến thể cùng 1 template miền ĐHTN hoặc cùng processed_content về 1 nhóm."""
    raw_norm = re.sub(r"\s+", " ", str(raw_content).strip().lower())
    raw_masked = _TNU_UNIT_CANON_RE.sub("<tnu_unit>", raw_norm)
    raw_masked = re.sub(r"(?:trường\s+)+<tnu_unit>", "<tnu_unit>", raw_masked)
    if "<tnu_unit>" in raw_masked:
        return re.sub(r"\s+", " ", raw_masked).strip()

    proc_norm = re.sub(r"\s+", " ", str(processed_content).strip().lower())
    proc_masked = _TNU_UNIT_CANON_RE.sub("<tnu_unit>", proc_norm)
    proc_masked = re.sub(r"(?:trường\s+)+<tnu_unit>", "<tnu_unit>", proc_masked)
    return re.sub(r"\s+", " ", proc_masked).strip()


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Tính độ tương đồng Jaccard trên tập từ (word tokens)."""
    set_a = set(str(text_a).lower().split())
    set_b = set(str(text_b).lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_char_ngram_jaccard(text_a: str, text_b: str, n: int = 4) -> float:
    """Tính độ tương đồng Jaccard trên tập character n-grams."""
    s_a = re.sub(r"\s+", " ", str(text_a).lower().strip())
    s_b = re.sub(r"\s+", " ", str(text_b).lower().strip())
    if len(s_a) < n or len(s_b) < n:
        return 1.0 if s_a == s_b else 0.0
    ngrams_a = {s_a[i : i + n] for i in range(len(s_a) - n + 1)}
    ngrams_b = {s_b[i : i + n] for i in range(len(s_b) - n + 1)}
    union_len = len(ngrams_a | ngrams_b)
    return len(ngrams_a & ngrams_b) / union_len if union_len else 0.0


def audit_test_near_duplicates(
    domain_texts: list[str],
    test_texts: list[str],
    jaccard_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Kiểm tra near-duplicate (Jaccard / Char 4-gram) giữa các câu miền ĐHTN và tập Test độc lập."""
    unique_domain = sorted({str(t).strip() for t in domain_texts if str(t).strip()})
    unique_test = sorted({str(t).strip() for t in test_texts if str(t).strip()})

    max_word_jaccard = 0.0
    max_char_jaccard = 0.0
    near_dup_pairs: list[Dict[str, Any]] = []

    domain_token_sets = [(t, set(t.lower().split())) for t in unique_domain]
    test_token_sets = [(t, set(t.lower().split())) for t in unique_test]

    for d_txt, d_set in domain_token_sets:
        if not d_set:
            continue
        for t_txt, t_set in test_token_sets:
            if not t_set:
                continue
            inter = len(d_set & t_set)
            if inter == 0:
                continue
            w_jac = inter / len(d_set | t_set)
            if w_jac > max_word_jaccard:
                max_word_jaccard = w_jac
            if w_jac >= 0.5:
                c_jac = compute_char_ngram_jaccard(d_txt, t_txt, n=4)
                if c_jac > max_char_jaccard:
                    max_char_jaccard = c_jac
                if w_jac >= jaccard_threshold or c_jac >= jaccard_threshold:
                    near_dup_pairs.append(
                        {
                            "domain_text": d_txt,
                            "test_text": t_txt,
                            "word_jaccard": round(w_jac, 4),
                            "char_4gram_jaccard": round(c_jac, 4),
                        }
                    )

    return {
        "domain_unique_count": len(unique_domain),
        "test_unique_count": len(unique_test),
        "jaccard_threshold": jaccard_threshold,
        "max_word_jaccard": round(max_word_jaccard, 4),
        "max_char_4gram_jaccard": round(max_char_jaccard, 4),
        "near_duplicate_count": len(near_dup_pairs),
        "near_duplicate_pairs": near_dup_pairs[:10],
    }


def rebuild_clean_splits(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Optional[Path] = None,
    preprocessor: Optional[VietnamesePreprocessor] = None,
    write_files: bool = True,
    audit_output_path: Optional[Path] = DEFAULT_AUDIT_PATH,
) -> Dict[str, Any]:
    """Chạy lại VietnamesePreprocessor.preprocess trên train/val/test, khử trùng lặp nội bộ và loại bỏ 100% rò rỉ train/val."""
    prep = preprocessor or VietnamesePreprocessor()
    target_dir = output_dir or dataset_dir

    train_path = dataset_dir / "tnu_train_final.csv"
    val_path = dataset_dir / "tnu_val_final.csv"
    test_path = dataset_dir / "tnu_test_final.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    orig_counts = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }

    # Ghi nhận số dòng rò rỉ trước khi làm sạch
    initial_proc_leakage_rows = int(
        val_df["processed_content"].isin(set(train_df["processed_content"])).sum()
    )

    # 1. Chạy lại VietnamesePreprocessor.preprocess(raw_content) trên toàn bộ train, val, test
    train_new_proc = train_df["raw_content"].astype(str).map(prep.preprocess)
    val_new_proc = val_df["raw_content"].astype(str).map(prep.preprocess)
    test_new_proc = test_df["raw_content"].astype(str).map(prep.preprocess)

    drift_counts = {
        "train": int((train_new_proc != train_df["processed_content"].astype(str)).sum()),
        "val": int((val_new_proc != val_df["processed_content"].astype(str)).sum()),
        "test": int((test_new_proc != test_df["processed_content"].astype(str)).sum()),
    }

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["processed_content"] = train_new_proc
    val_df["processed_content"] = val_new_proc
    test_df["processed_content"] = test_new_proc

    # 2. Tính group_key cho train và val để gom nhóm các câu cùng processed_content hoặc cùng template gốc
    train_df["group_key"] = [
        compute_template_group_key(r, p)
        for r, p in zip(train_df["raw_content"], train_df["processed_content"])
    ]
    val_df["group_key"] = [
        compute_template_group_key(r, p)
        for r, p in zip(val_df["raw_content"], val_df["processed_content"])
    ]

    # Xác định 28 group_key sinh từ bộ template miền ĐHTN (tnu_domain_templates: 714 dòng ở train + 126 dòng ở val)
    val_tail_keys = set(val_df.iloc[1583:]["group_key"]) if len(val_df) > 1583 else set()
    template_group_keys: Set[str] = (
        val_tail_keys
        | {k for k in set(train_df["group_key"]) | set(val_df["group_key"]) if "<tnu_unit>" in k}
    )

    # Phân hoạch độc quyền (disjoint group split) các template_group_keys giữa train và val theo nhãn cảm xúc
    # để mọi biến thể của cùng 1 template chỉ nằm ở train HOẶC val (0 overlap).
    combined_templates = pd.concat(
        [
            train_df[train_df["group_key"].isin(template_group_keys)],
            val_df[val_df["group_key"].isin(template_group_keys)],
        ],
        ignore_index=True,
    )
    # Khử trùng lặp nội bộ trên (processed_content)
    combined_templates = combined_templates.drop_duplicates(subset=["processed_content"], keep="first")
    combined_templates = combined_templates.drop_duplicates(subset=["raw_content"], keep="first")

    val_assigned_template_keys: Set[str] = set()
    for _sentiment_val, grp in combined_templates.groupby("sentiment", sort=True):
        unique_keys = sorted(grp["group_key"].unique())
        # Chọn ~20% số nhóm template (tối thiểu 1 nhóm nếu có >=2 nhóm) đưa vào tập Validation
        n_val_groups = max(1, round(len(unique_keys) * 0.2)) if len(unique_keys) >= 2 else 0
        for k in unique_keys[:n_val_groups]:
            val_assigned_template_keys.add(k)

    train_assigned_template_keys = template_group_keys - val_assigned_template_keys

    # Tách phần dữ liệu gốc (không thuộc template_group_keys) và khử trùng lặp nội bộ + khử giao thoa train/val
    base_train = train_df[~train_df["group_key"].isin(template_group_keys)].copy()
    base_val = val_df[~val_df["group_key"].isin(template_group_keys)].copy()

    base_train = base_train.drop_duplicates(subset=["processed_content"], keep="first")
    base_train = base_train.drop_duplicates(subset=["raw_content"], keep="first")
    base_train = base_train.drop_duplicates(subset=["group_key"], keep="first")

    base_val = base_val.drop_duplicates(subset=["processed_content"], keep="first")
    base_val = base_val.drop_duplicates(subset=["raw_content"], keep="first")
    base_val = base_val.drop_duplicates(subset=["group_key"], keep="first")

    # Loại bỏ khỏi base_val mọi dòng trùng processed_content, raw_content hoặc group_key với base_train
    train_proc_set = set(base_train["processed_content"])
    train_raw_set = set(base_train["raw_content"])
    train_key_set = set(base_train["group_key"])

    base_val = base_val[
        ~base_val["processed_content"].isin(train_proc_set)
        & ~base_val["raw_content"].isin(train_raw_set)
        & ~base_val["group_key"].isin(train_key_set)
    ].copy()

    # Ghép các nhóm template đã phân hoạch độc quyền vào train và val
    tpl_train = combined_templates[
        combined_templates["group_key"].isin(train_assigned_template_keys)
    ].copy()
    tpl_val = combined_templates[
        combined_templates["group_key"].isin(val_assigned_template_keys)
    ].copy()

    cols = [c for c in ["sentence", "sentiment", "topic", "raw_content", "processed_content"] if c in train_df.columns]
    clean_train_df = pd.concat([base_train[cols], tpl_train[cols]], ignore_index=True)
    clean_val_df = pd.concat([base_val[cols], tpl_val[cols]], ignore_index=True)
    clean_test_df = test_df[cols].copy()

    # Kiểm tra hậu điều kiện nghiêm ngặt: 0 dòng giao thoa giữa train và val
    clean_train_keys = {
        compute_template_group_key(r, p)
        for r, p in zip(clean_train_df["raw_content"], clean_train_df["processed_content"])
    }
    clean_val_keys = {
        compute_template_group_key(r, p)
        for r, p in zip(clean_val_df["raw_content"], clean_val_df["processed_content"])
    }

    overlap_raw = set(clean_train_df["raw_content"]) & set(clean_val_df["raw_content"])
    overlap_proc = set(clean_train_df["processed_content"]) & set(clean_val_df["processed_content"])
    overlap_group = clean_train_keys & clean_val_keys

    assert len(overlap_raw) == 0, f"Còn rò rỉ raw_content giữa train và val: {len(overlap_raw)}"
    assert len(overlap_proc) == 0, f"Còn rò rỉ processed_content giữa train và val: {len(overlap_proc)}"
    assert len(overlap_group) == 0, f"Còn rò rỉ group_key giữa train và val: {len(overlap_group)}"
    assert len(clean_test_df) == 3166, f"Tập test phải giữ nguyên 3.166 dòng, thực tế: {len(clean_test_df)}"

    # Kiểm tra near-duplicate giữa các mẫu miền ĐHTN và tập Test độc lập (3.166 câu)
    near_dup_audit = audit_test_near_duplicates(
        domain_texts=combined_templates["processed_content"].astype(str).tolist(),
        test_texts=clean_test_df["processed_content"].astype(str).tolist(),
        jaccard_threshold=0.85,
    )

    if write_files:
        target_dir.mkdir(parents=True, exist_ok=True)
        clean_train_df.to_csv(target_dir / "tnu_train_final.csv", index=False, encoding="utf-8")
        clean_val_df.to_csv(target_dir / "tnu_val_final.csv", index=False, encoding="utf-8")
        clean_test_df.to_csv(target_dir / "tnu_test_final.csv", index=False, encoding="utf-8")

    report = {
        "original_counts": orig_counts,
        "clean_counts": {
            "train": len(clean_train_df),
            "val": len(clean_val_df),
            "test": len(clean_test_df),
        },
        "preprocessing_drift_fixed": drift_counts,
        "initial_train_val_processed_leakage_rows": initial_proc_leakage_rows,
        "after_clean_overlap": {
            "raw_content_overlap": len(overlap_raw),
            "processed_content_overlap": len(overlap_proc),
            "group_key_overlap": len(overlap_group),
        },
        "template_groups": {
            "total_template_groups": len(template_group_keys),
            "train_template_groups": len(train_assigned_template_keys),
            "val_template_groups": len(val_assigned_template_keys),
        },
        "test_near_duplicate_audit": near_dup_audit,
        "train_df": clean_train_df,
        "val_df": clean_val_df,
        "test_df": clean_test_df,
    }

    if audit_output_path is not None and write_files:
        audit_output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_report = {
            k: v for k, v in report.items() if k not in {"train_df", "val_df", "test_df"}
        }
        audit_output_path.write_text(
            json.dumps(serializable_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Đồng nhất tiền xử lý VietnamesePreprocessor và khử rò rỉ train/val (0 overlap)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Thư mục chứa tnu_train_final.csv, tnu_val_final.csv, tnu_test_final.csv gốc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục ghi các tệp CSV đã làm sạch (mặc định ghi vào thư mục TNU_Sentiment_Dataset_Final_Clean hoặc --output-dir).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Ghi đè trực tiếp lên thư mục --dataset-dir.",
    )
    args = parser.parse_args(argv)

    out_dir = args.dataset_dir if args.in_place else (args.output_dir or (ROOT_DIR / "TNU_Sentiment_Dataset_Final_Clean"))
    report = rebuild_clean_splits(
        dataset_dir=args.dataset_dir,
        output_dir=out_dir,
        write_files=True,
    )
    summary = {k: v for k, v in report.items() if k not in {"train_df", "val_df", "test_df"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
