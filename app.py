"""Ứng dụng Web Streamlit + Plotly + SQLAlchemy hoàn chỉnh 4 phân hệ — TNU Sentiment.

Đề tài: Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên
Sinh viên thực hiện: Nguyễn Văn Chiến - MSSV: DTC235200079 - Lớp: CN CNTT K22A (ICTU)
Giảng viên hướng dẫn: TS. Hà Thị Thanh
"""

from __future__ import annotations

import html
import io
import os
import time
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from src.llm_fallback import (
    SENTIMENT_ID_TO_TEXT,
    SENTIMENT_ID_TO_VI,
    TOPIC_ID_TO_NAME,
    call_gemini_LowConfidence_router,
    infer_topic_by_keywords,
    is_gemini_configured,
)
from src.preprocessor import TNUStudentFeedbackPreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ==============================================================================
# 1. HẰNG SỐ GIAO DIỆN, BẢNG MÀU HỌC THUẬT & DANH MỤC 9 ĐƠN VỊ THÀNH VIÊN ĐHTN
# ==============================================================================
PALETTE = {
    "navy": "#1E3A8A",
    "bg": "#F8FAFC",
    "surface": "#FFFFFF",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#475569",
    "positive": "#1A73E8",
    "neutral": "#64748B",
    "negative": "#D93025",
    "warning": "#F59E0B",
}

SENTIMENT_COLOR_MAP = {
    "Positive": PALETTE["positive"],
    "Neutral": PALETTE["neutral"],
    "Negative": PALETTE["negative"],
    "Tích cực (Positive)": PALETTE["positive"],
    "Trung tính (Neutral)": PALETTE["neutral"],
    "Tiêu cực (Negative)": PALETTE["negative"],
    "Tích cực": PALETTE["positive"],
    "Trung tính": PALETTE["neutral"],
    "Tiêu cực": PALETTE["negative"],
}

TNU_UNITS: Dict[int, Tuple[str, str]] = {
    1: ("ICTU", "Trường ĐH Công nghệ Thông tin và Truyền thông (ICTU)"),
    2: ("TNUT", "Trường ĐH Kỹ thuật Công nghiệp (TNUT)"),
    3: ("TUEBA", "Trường ĐH Kinh tế và Quản trị Kinh doanh (TUEBA)"),
    4: ("TNUE", "Trường ĐH Sư phạm (TNUE)"),
    5: ("TUAF", "Trường ĐH Nông Lâm (TUAF)"),
    6: ("TUMP", "Trường ĐH Y - Dược (TUMP)"),
    7: ("TNUS", "Trường ĐH Khoa học (TNUS)"),
    8: ("SFL", "Trường Ngoại ngữ (SFL)"),
    9: ("ISTNU", "Khoa Quốc tế (ISTNU)"),
}
UNIT_CODE_TO_ID: Dict[str, int] = {code: uid for uid, (code, _) in TNU_UNITS.items()}

DATA_SOURCES: Dict[int, str] = {
    1: "Hệ thống khảo sát người học cuối học kỳ",
    2: "Hệ thống quản lý học tập trực tuyến (LMS)",
    3: "Diễn đàn & Fanpage cộng đồng sinh viên ĐHTN",
    4: "Hòm thư góp ý trực tuyến & Import CSV/XLSX",
}
SOURCE_NAME_TO_ID: Dict[str, int] = {
    "SURVEY_PORTAL": 1,
    "LMS_ELEARNING": 2,
    "LMS_FEEDBACK": 2,
    "FANPAGE_FORUM": 3,
    "STUDENT_FORUM": 3,
    "QA_IMPORT_BOX": 4,
    "QA_BOX": 4,
    "CSV_IMPORT": 4,
}

# 6 câu mẫu kiểm thử điển hình phục vụ Tab 1 (Live AI Sandbox)
SAMPLE_TEST_CASES: Dict[str, Dict[str, Any]] = {
    "1. Câu khen có teencode & emoji": {
        "text": "Thầy dạy môn lập trình Python giảng bài siêuuuu dễ hỉu 🥰, nhiệt tình hỗ trợ sv làm đồ án.",
        "unit_code": "ICTU",
        "source_id": 1,
    },
    "2. Câu chê cơ sở vật chất": {
        "text": "Phòng thực hành máy tính điều hòa hỏng nóng ko chịu đc, máy chiếu mờ và mạng wifi chập chờn.",
        "unit_code": "TNUT",
        "source_id": 4,
    },
    "3. Câu trung tính thông báo": {
        "text": "Học phần này có thời lượng 3 tín chỉ bao gồm 30 tiết lý thuyết và 30 tiết thực hành tại nhà C5.",
        "unit_code": "ICTU",
        "source_id": 2,
    },
    "4. Câu mỉa mai có ':)))'": {
        "text": "Hệ thống đăng ký tín chỉ mượt ghê cơ, quay mòng mòng từ 7h sáng tới trưa vẫn chưa vào được trang chủ :)))",
        "unit_code": "TUEBA",
        "source_id": 3,
    },
    "5. Câu chứa PII (Mã SV DTC235200079 / SĐT / Email)": {
        "text": "Em là sv mã số DTC235200079, sđt 0912345678, email chien.ictu@tnu.edu.vn xin phản ánh lịch thi trùng ca sáng.",
        "unit_code": "ICTU",
        "source_id": 4,
    },
    "6. Câu ghép đa cực tính (vừa khen vừa chê)": {
        "text": "Giảng viên dạy rất nhiệt tình và dễ hiểu nhưng máy chiếu phòng học mờ quá cần bảo trì ngay.",
        "unit_code": "TNUE",
        "source_id": 1,
    },
}


# ==============================================================================
# 2. KHỞI TẠO TÀI NGUYÊN CACHE (@st.cache_resource) & NẠP MÔ HÌNH MINH BẠCH
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_preprocessor() -> TNUStudentFeedbackPreprocessor:
    return TNUStudentFeedbackPreprocessor()


def _resolve_checkpoint_path() -> Optional[Path]:
    """Kiểm tra sự tồn tại của checkpoint đã tinh chỉnh tại các đường dẫn chuẩn."""
    env_dir = os.getenv("SENTIMENT_CHECKPOINT_DIR", "").strip()
    candidates: List[Path] = []
    if env_dir:
        p_env = Path(env_dir)
        candidates.append(p_env if p_env.is_absolute() else (PROJECT_ROOT / p_env))
    candidates.extend(
        [
            PROJECT_ROOT / "phobert_tnu_best_model",
            PROJECT_ROOT / "models" / "phobert_tnu_sentiment",
            PROJECT_ROOT
            / "phobert_tnu_best_model-20261101T002214Z-1-001"
            / "phobert_tnu_best_model",
        ]
    )
    for cand in candidates:
        if (
            cand.exists()
            and cand.is_dir()
            and (cand / "config.json").exists()
            and (
                (cand / "model.safetensors").exists()
                or (cand / "pytorch_model.bin").exists()
            )
        ):
            return cand
    return None


@st.cache_resource(show_spinner="Đang khởi tạo mô hình PhoBERT-base-v2...")
def load_ai_models() -> Dict[str, Any]:
    """Nạp checkpoint PhoBERT đã tinh chỉnh một lần duy nhất vào bộ nhớ."""
    ckpt_dir = _resolve_checkpoint_path()
    topic_ckpt_env = os.getenv("TOPIC_CHECKPOINT_DIR", "").strip()
    topic_ready = bool(topic_ckpt_env and Path(topic_ckpt_env).exists())

    if ckpt_dir is not None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(ckpt_dir), local_files_only=True
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            return {
                "is_finetuned": True,
                "sentiment_status": "🟢 PhoBERT Fine-tuned Loaded",
                "topic_status": "🟢 Topic Model Ready" if topic_ready else "🟡 Keyword/LLM Router",
                "checkpoint_dir": str(ckpt_dir),
                "model_name": "PhoBERT-base-v2-Focal",
                "tokenizer": tokenizer,
                "model": model,
                "torch": torch,
                "device": device,
            }
        except Exception as exc:
            fallback_err = str(exc)
    else:
        fallback_err = "Không tìm thấy trọng số model.safetensors tại ./phobert_tnu_best_model"

    return {
        "is_finetuned": False,
        "sentiment_status": "🟠 Base Model Fallback",
        "topic_status": "🟢 Topic Model Ready" if topic_ready else "🟡 Keyword/LLM Router",
        "checkpoint_dir": fallback_err,
        "model_name": "Base-Fallback-Lexicon",
        "tokenizer": None,
        "model": None,
        "torch": None,
        "device": "cpu",
    }


@st.cache_resource(show_spinner=False)
def get_mysql_engine() -> Tuple[Optional[Engine], str]:
    """Khởi tạo kết nối MySQL 8.0 nếu có cấu hình; tự động chuyển về CSV Fallback nếu offline."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    host = os.getenv("MYSQL_HOST", "").strip()
    user = os.getenv("MYSQL_USER", "").strip()
    db_name = os.getenv("MYSQL_DATABASE", "").strip()
    password = os.getenv("MYSQL_PASSWORD", "")
    port = os.getenv("MYSQL_PORT", "3306").strip() or "3306"

    if not db_url and host and user and db_name:
        db_url = (
            f"mysql+pymysql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port}/{db_name}?charset=utf8mb4"
        )
    if not db_url:
        return None, "🟡 CSV Fallback Mode (Offline)"

    try:
        eng = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": 3},
        )
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return eng, "🟢 MySQL 8.0 Connected"
    except Exception:
        return None, "🟡 CSV Fallback Mode (Offline)"


# ==============================================================================
# 3. SUY LUẬN ĐỒNG BỘ TIỀN XỬ LÝ (CHỐNG TRAINING-SERVING SKEW) & HYBRID ROUTER
# ==============================================================================
def predict_feedback_sentiment_and_topic(
    raw_text: str,
    preprocessor: TNUStudentFeedbackPreprocessor,
    model_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Thực thi Pipeline tiền xử lý 6 bước trước khi đưa vào Tokenizer & PhoBERT (max_length=128)
    kèm cơ chế định tuyến lai (Hybrid Confidence-based Router) khi confidence < 0.65.
    """
    t0 = time.perf_counter()

    # BẮT BUỘC: Chạy tiền xử lý 6 bước để triệt tiêu Training-Serving Skew
    steps = preprocessor.preprocess_with_steps(raw_text)
    processed_text = preprocessor.preprocess(raw_text)
    masked_text = steps["step1_nfc_pii"]
    content_hash = preprocessor.compute_content_hash(raw_text)

    if model_bundle.get("is_finetuned") and model_bundle.get("model") is not None:
        torch = model_bundle["torch"]
        tokenizer = model_bundle["tokenizer"]
        model = model_bundle["model"]
        device = model_bundle["device"]

        encoded = tokenizer(
            processed_text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            probs_t = torch.softmax(outputs.logits, dim=-1)[0]
            probs = [float(x) for x in probs_t.detach().cpu().tolist()]
    else:
        # Lexicon Fallback khi chạy trên môi trường chưa tải file model.safetensors (515MB)
        lowered = processed_text.lower()
        pos_cues = ("tích_cực", "nhiệt_tình", "tận_tâm", "dễ hiểu", "tốt", "hay", "vui_tính")
        neg_cues = (
            "tiêu_cực",
            "hỏng",
            "mờ",
            "chập_chờn",
            "nóng",
            "quay mòng",
            "chậm",
            "kém",
            "không hiểu",
        )
        p_score = sum(1 for w in pos_cues if w in lowered)
        n_score = sum(1 for w in neg_cues if w in lowered)
        if n_score > p_score:
            probs = [0.84, 0.08, 0.08]
        elif p_score > n_score:
            probs = [0.06, 0.08, 0.86]
        else:
            probs = [0.12, 0.76, 0.12]

    prob_neg, prob_neu, prob_pos = (
        round(probs[0], 4),
        round(probs[1], 4),
        round(probs[2], 4),
    )
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])

    topic_id, topic_name, topic_router_src = infer_topic_by_keywords(processed_text)
    review_status = "AUTO_APPROVED"
    router_info: Optional[Dict[str, Any]] = None

    # Định tuyến lai (Hybrid Confidence-based Router) khi confidence < 0.65
    if confidence < 0.65:
        review_status = "NEEDS_REVIEW"
        router_info = call_gemini_LowConfidence_router(
            masked_text=masked_text,
            phobert_label=pred_label,
            phobert_confidence=confidence,
        )
        topic_id = int(router_info["suggested_topic_id"])
        topic_name = str(router_info["suggested_topic_name"])
        topic_router_src = str(router_info["router_mode"])

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "raw_content": masked_text,
        "processed_content": processed_text,
        "steps": steps,
        "content_hash": content_hash,
        "sentiment_label": pred_label,
        "sentiment_text": SENTIMENT_ID_TO_TEXT[pred_label],
        "sentiment_vi": SENTIMENT_ID_TO_VI[pred_label],
        "confidence_score": round(confidence, 4),
        "prob_negative": prob_neg,
        "prob_neutral": prob_neu,
        "prob_positive": prob_pos,
        "topic_id": topic_id,
        "topic_name": topic_name,
        "topic_router_source": topic_router_src,
        "review_status": review_status,
        "router_info": router_info,
        "model_used": model_bundle["model_name"],
        "latency_ms": latency_ms,
    }


# ==============================================================================
# 4. TẦNG DỮ LIỆU ĐỒNG BỘ MYSQL 8.0 (KHOẢNG NỬA MỞ TỐI ƯU B-TREE INDEX) & CSV FALLBACK
# ==============================================================================
def _load_offline_seed_dataframe(preprocessor: TNUStudentFeedbackPreprocessor) -> pd.DataFrame:
    """Nạp dữ liệu từ tệp CSV Seed khi không có kết nối MySQL và chuẩn hóa đầy đủ các cột."""
    if "offline_feedbacks_df" in st.session_state:
        return st.session_state["offline_feedbacks_df"]

    csv_candidates = [
        PROJECT_ROOT / "dataset_final" / "tnu_mysql_seed_data.csv",
        PROJECT_ROOT / "TNU_Sentiment_Dataset_Final" / "tnu_mysql_seed_data.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        empty_df = pd.DataFrame(
            columns=[
                "feedback_id",
                "content_hash",
                "unit_id",
                "unit_code",
                "unit_name",
                "source_id",
                "source_name",
                "topic_id",
                "topic_name",
                "raw_content",
                "processed_content",
                "sentiment_label",
                "sentiment_text",
                "confidence_score",
                "prob_negative",
                "prob_neutral",
                "prob_positive",
                "model_used",
                "is_synthetic",
                "review_status",
                "reviewed_sentiment_label",
                "reviewed_topic_id",
                "reviewed_by",
                "review_note",
                "feedback_date",
            ]
        )
        st.session_state["offline_feedbacks_df"] = empty_df
        return empty_df

    df = pd.read_csv(csv_path, encoding="utf-8")
    df["feedback_id"] = np.arange(1, len(df) + 1)
    df["raw_content"] = df["raw_content"].astype(str).apply(preprocessor.redact_pii_only)
    df["content_hash"] = [
        preprocessor.compute_content_hash(f"{row_txt}_{idx}")
        for idx, row_txt in enumerate(df["raw_content"])
    ]
    df["unit_id"] = df["unit_id"].astype(int).clip(1, 9)
    df["unit_code"] = df["unit_id"].map(lambda u: TNU_UNITS[u][0])
    df["unit_name"] = df["unit_id"].map(lambda u: TNU_UNITS[u][1])
    df["source_id"] = df["source_id"].astype(int).clip(1, 4)
    df["source_name"] = df["source_id"].map(DATA_SOURCES)
    df["topic_id"] = df["topic_id"].astype(int).clip(1, 4)
    df["topic_name"] = df["topic_id"].map(TOPIC_ID_TO_NAME)
    df["is_synthetic"] = 1  # Tệp seed 3.500 dòng được gắn rõ is_synthetic = 1
    df["review_status"] = np.where(
        df["is_verified"] == 1,
        "VERIFIED",
        np.where(df["confidence_score"] < 0.65, "NEEDS_REVIEW", "AUTO_APPROVED"),
    )
    df["reviewed_sentiment_label"] = np.where(
        df["is_verified"] == 1, df["sentiment_label"], None
    )
    df["reviewed_topic_id"] = np.where(df["is_verified"] == 1, df["topic_id"], None)
    df["reviewed_by"] = np.where(df["is_verified"] == 1, "seed_reviewer", None)
    df["review_note"] = ""
    df["feedback_date"] = pd.to_datetime(df["feedback_date"], errors="coerce").fillna(
        pd.Timestamp("2026-10-01 08:00:00")
    )

    st.session_state["offline_feedbacks_df"] = df
    return df


def query_filtered_feedbacks(
    engine: Optional[Engine],
    preprocessor: TNUStudentFeedbackPreprocessor,
    data_mode: str,
    start_date: date,
    end_date: date,
    selected_unit_ids: List[int],
    selected_source_ids: List[int],
    selected_topic_ids: List[int],
) -> pd.DataFrame:
    """Truy vấn dữ liệu phản hồi theo bộ lọc.
    BẮT BUỘC dùng khoảng nửa mở [start_dt, end_dt) để tận dụng B-Tree Index idx_dashboard_filter,
    tuyệt đối KHÔNG dùng hàm DATE(f.feedback_date) BETWEEN.
    """
    start_dt = datetime.combine(start_date, dt_time.min)
    end_dt = datetime.combine(end_date + timedelta(days=1), dt_time.min)

    if engine is not None:
        clauses = [
            "f.feedback_date >= :start_dt",
            "f.feedback_date < :end_dt",
        ]
        params: Dict[str, Any] = {"start_dt": start_dt, "end_dt": end_dt}
        expanding_params = []

        if data_mode == "Chỉ dữ liệu thực tế (is_synthetic = 0)":
            clauses.append("f.is_synthetic = 0")
        elif data_mode == "Dữ liệu Seed Demo (is_synthetic = 1)":
            clauses.append("f.is_synthetic = 1")

        if selected_unit_ids:
            clauses.append("f.unit_id IN :unit_ids")
            params["unit_ids"] = list(selected_unit_ids)
            expanding_params.append(bindparam("unit_ids", expanding=True))

        if selected_source_ids:
            clauses.append("f.source_id IN :source_ids")
            params["source_ids"] = list(selected_source_ids)
            expanding_params.append(bindparam("source_ids", expanding=True))

        if selected_topic_ids:
            clauses.append("COALESCE(f.reviewed_topic_id, f.topic_id) IN :topic_ids")
            params["topic_ids"] = list(selected_topic_ids)
            expanding_params.append(bindparam("topic_ids", expanding=True))

        where_sql = " AND ".join(clauses)
        stmt = text(
            f"""
            SELECT
                f.feedback_id,
                f.content_hash,
                f.unit_id,
                u.unit_code,
                u.unit_name,
                f.source_id,
                s.source_name,
                f.topic_id,
                t.topic_name,
                f.raw_content,
                f.processed_content,
                f.sentiment_label,
                f.sentiment_text,
                f.confidence_score,
                f.prob_negative,
                f.prob_neutral,
                f.prob_positive,
                f.model_used,
                f.is_synthetic,
                f.review_status,
                f.reviewed_sentiment_label,
                f.reviewed_topic_id,
                f.reviewed_by,
                f.review_note,
                f.feedback_date
            FROM student_feedbacks f
            JOIN tnu_units u ON f.unit_id = u.unit_id
            JOIN data_sources s ON f.source_id = s.source_id
            JOIN topics t ON f.topic_id = t.topic_id
            WHERE {where_sql}
            ORDER BY f.feedback_date DESC
            """
        )
        if expanding_params:
            stmt = stmt.bindparams(*expanding_params)

        with engine.connect() as conn:
            df = pd.read_sql(stmt, conn, params=params)
        if not df.empty:
            df["feedback_date"] = pd.to_datetime(df["feedback_date"])
        return df

    # Chế độ CSV Fallback (Offline)
    base_df = _load_offline_seed_dataframe(preprocessor).copy()
    mask = (base_df["feedback_date"] >= pd.Timestamp(start_dt)) & (
        base_df["feedback_date"] < pd.Timestamp(end_dt)
    )
    if data_mode == "Chỉ dữ liệu thực tế (is_synthetic = 0)":
        mask &= base_df["is_synthetic"] == 0
    elif data_mode == "Dữ liệu Seed Demo (is_synthetic = 1)":
        mask &= base_df["is_synthetic"] == 1

    if selected_unit_ids:
        mask &= base_df["unit_id"].isin(selected_unit_ids)
    if selected_source_ids:
        mask &= base_df["source_id"].isin(selected_source_ids)
    if selected_topic_ids:
        eff_topic = base_df["reviewed_topic_id"].fillna(base_df["topic_id"]).astype(int)
        mask &= eff_topic.isin(selected_topic_ids)

    return base_df.loc[mask].sort_values("feedback_date", ascending=False).reset_index(drop=True)


def save_feedback_record(
    engine: Optional[Engine],
    preprocessor: TNUStudentFeedbackPreprocessor,
    result: Dict[str, Any],
    unit_id: int,
    source_id: int,
    is_synthetic: int = 0,
    feedback_dt: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """Lưu bản ghi phản hồi mới và kiểm tra content_hash chống trùng lặp tuyệt đối."""
    fb_time = feedback_dt or datetime.now()
    c_hash = result["content_hash"]
    a_hash = preprocessor.compute_author_hash(f"{unit_id}:{source_id}")

    if engine is not None:
        check_stmt = text(
            "SELECT feedback_id FROM student_feedbacks WHERE content_hash = :c_hash LIMIT 1"
        )
        ins_stmt = text(
            """
            INSERT INTO student_feedbacks (
                content_hash, author_hash, unit_id, source_id, topic_id,
                raw_content, processed_content,
                sentiment_label, sentiment_text, confidence_score,
                prob_negative, prob_neutral, prob_positive,
                model_used, latency_ms, is_synthetic, review_status, feedback_date
            ) VALUES (
                :content_hash, :author_hash, :unit_id, :source_id, :topic_id,
                :raw_content, :processed_content,
                :sentiment_label, :sentiment_text, :confidence_score,
                :prob_negative, :prob_neutral, :prob_positive,
                :model_used, :latency_ms, :is_synthetic, :review_status, :feedback_date
            )
            """
        )
        with engine.begin() as conn:
            exists = conn.execute(check_stmt, {"c_hash": c_hash}).fetchone()
            if exists is not None:
                return (
                    False,
                    f"Bản ghi bị chặn do trùng mã băm SHA-256 (content_hash = {c_hash[:12]}... đã tồn tại).",
                )
            conn.execute(
                ins_stmt,
                {
                    "content_hash": c_hash,
                    "author_hash": a_hash,
                    "unit_id": unit_id,
                    "source_id": source_id,
                    "topic_id": result["topic_id"],
                    "raw_content": result["raw_content"],
                    "processed_content": result["processed_content"],
                    "sentiment_label": result["sentiment_label"],
                    "sentiment_text": result["sentiment_text"],
                    "confidence_score": result["confidence_score"],
                    "prob_negative": result["prob_negative"],
                    "prob_neutral": result["prob_neutral"],
                    "prob_positive": result["prob_positive"],
                    "model_used": result["model_used"],
                    "latency_ms": result["latency_ms"],
                    "is_synthetic": is_synthetic,
                    "review_status": result["review_status"],
                    "feedback_date": fb_time,
                },
            )
        return True, f"Đã lưu phản hồi thành công vào MySQL 8.0 (content_hash: {c_hash[:12]}...)."

    # Lưu trong bộ nhớ phiên khi ở CSV Fallback Mode
    df = _load_offline_seed_dataframe(preprocessor)
    if c_hash in set(df["content_hash"].astype(str)):
        return (
            False,
            f"Bản ghi bị chặn do trùng mã băm SHA-256 (content_hash = {c_hash[:12]}... đã tồn tại trong phiên).",
        )

    next_id = int(df["feedback_id"].max()) + 1 if not df.empty else 1
    u_code, u_name = TNU_UNITS.get(unit_id, ("ICTU", "ICTU"))
    new_row = pd.DataFrame(
        [
            {
                "feedback_id": next_id,
                "content_hash": c_hash,
                "unit_id": unit_id,
                "unit_code": u_code,
                "unit_name": u_name,
                "source_id": source_id,
                "source_name": DATA_SOURCES.get(source_id, "Khảo sát"),
                "topic_id": result["topic_id"],
                "topic_name": TOPIC_ID_TO_NAME.get(result["topic_id"], "Khác"),
                "raw_content": result["raw_content"],
                "processed_content": result["processed_content"],
                "sentiment_label": result["sentiment_label"],
                "sentiment_text": result["sentiment_text"],
                "confidence_score": result["confidence_score"],
                "prob_negative": result["prob_negative"],
                "prob_neutral": result["prob_neutral"],
                "prob_positive": result["prob_positive"],
                "model_used": result["model_used"],
                "is_synthetic": is_synthetic,
                "review_status": result["review_status"],
                "reviewed_sentiment_label": None,
                "reviewed_topic_id": None,
                "reviewed_by": None,
                "review_note": "",
                "feedback_date": pd.Timestamp(fb_time),
            }
        ]
    )
    st.session_state["offline_feedbacks_df"] = pd.concat([new_row, df], ignore_index=True)
    return (
        True,
        f"Đã lưu phản hồi vào bộ nhớ phiên CSV Fallback (ID #{next_id}, content_hash: {c_hash[:12]}...).",
    )


# ==============================================================================
# 5. CUSTOM CSS & PLOTLY TEMPLATE CHUẨN HỌC THUẬT
# ==============================================================================
def inject_enterprise_css() -> None:
    st.markdown(
        f"""
        <style>
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 2.5rem;
                max-width: 1440px;
            }}
            .tnu-top-banner {{
                background: linear-gradient(135deg, {PALETTE['navy']} 0%, #1E40AF 100%);
                color: #FFFFFF;
                padding: 1.1rem 1.4rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                box-shadow: 0 4px 12px rgba(30, 58, 138, 0.12);
            }}
            .tnu-top-banner h1 {{
                font-size: 1.4rem;
                font-weight: 700;
                margin: 0 0 0.25rem 0;
                color: #FFFFFF;
            }}
            .tnu-top-banner p {{
                font-size: 0.9rem;
                margin: 0;
                color: #E2E8F0;
            }}
            .tnu-status-strip {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
                margin-top: 0.75rem;
            }}
            .tnu-status-pill {{
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.25);
                color: #FFFFFF;
                padding: 0.25rem 0.75rem;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
            }}
            .tnu-kpi-card {{
                background: {PALETTE['surface']};
                border: 1px solid {PALETTE['border']};
                border-top: 4px solid {PALETTE['navy']};
                border-radius: 12px;
                padding: 1rem 1.15rem;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
                height: 100%;
            }}
            .tnu-kpi-title {{
                font-size: 0.84rem;
                font-weight: 600;
                color: {PALETTE['muted']};
                margin-bottom: 0.3rem;
            }}
            .tnu-kpi-val {{
                font-size: 1.65rem;
                font-weight: 700;
                color: {PALETTE['text']};
                line-height: 1.2;
            }}
            .tnu-kpi-sub {{
                font-size: 0.8rem;
                color: {PALETTE['muted']};
                margin-top: 0.35rem;
            }}
            .tnu-step-box {{
                background: #F8FAFC;
                border: 1px solid {PALETTE['border']};
                border-left: 4px solid {PALETTE['navy']};
                border-radius: 8px;
                padding: 0.55rem 0.85rem;
                margin-bottom: 0.5rem;
                font-size: 0.86rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=PALETTE["text"]), x=0.0),
        height=height,
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        margin=dict(l=16, r=16, t=48, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def render_scorecard(title: str, value: str, subtitle: str, border_color: str) -> None:
    st.markdown(
        f"""
        <div class="tnu-kpi-card" style="border-top-color: {html.escape(border_color)};">
            <div class="tnu-kpi-title">{html.escape(title)}</div>
            <div class="tnu-kpi-val" style="color: {html.escape(border_color)};">{html.escape(value)}</div>
            <div class="tnu-kpi-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 6. CHƯƠNG TRÌNH CHÍNH — 4 PHÂN HỆ GIAO DIỆN (TAB 1 -> TAB 4)
# ==============================================================================
def main() -> None:
    st.set_page_config(
        page_title="TNU Sentiment — Hệ thống Phân tích Cảm xúc Phản hồi Cộng đồng ĐHTN",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_enterprise_css()

    preprocessor = get_preprocessor()
    model_bundle = load_ai_models()
    engine, db_status = get_mysql_engine()

    # Top Status Banner hiển thị minh bạch trạng thái 3 thành phần
    st.markdown(
        f"""
        <div class="tnu-top-banner">
            <h1>🏛️ TNU SENTIMENT — HỆ THỐNG PHÂN TÍCH CẢM XÚC Ý KIẾN PHẢN HỒI ĐẠI HỌC THÁI NGUYÊN</h1>
            <p>Sinh viên thực hiện: <strong>Nguyễn Văn Chiến</strong> (MSSV: <code>DTC235200079</code> · Lớp: CN CNTT K22A — ICTU) · Giảng viên hướng dẫn: <strong>TS. Hà Thị Thanh</strong></p>
            <div class="tnu-status-strip">
                <span class="tnu-status-pill">CSDL: {html.escape(db_status)}</span>
                <span class="tnu-status-pill">Model Cảm xúc: {html.escape(model_bundle['sentiment_status'])}</span>
                <span class="tnu-status-pill">Model Chủ đề: {html.escape(model_bundle['topic_status'])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # SIDEBAR: BỘ LỌC TOÀN CỤC CHO DASHBOARD & QUẢN LÝ DỮ LIỆU
    # --------------------------------------------------------------------------
    st.sidebar.markdown("### 🎛️ Bộ lọc Giám sát Toàn ĐHTN")
    data_mode = st.sidebar.radio(
        "Chế độ dữ liệu",
        options=[
            "Tất cả",
            "Chỉ dữ liệu thực tế (is_synthetic = 0)",
            "Dữ liệu Seed Demo (is_synthetic = 1)",
        ],
        index=0,
        key="sb_data_mode",
    )

    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        start_date = st.date_input("Từ ngày", value=date(2026, 8, 1), key="sb_start_date")
    with col_d2:
        end_date = st.date_input("Đến ngày", value=date(2026, 11, 30), key="sb_end_date")

    selected_unit_ids = st.sidebar.multiselect(
        "Đơn vị thành viên (9 trường/khoa ĐHTN)",
        options=list(TNU_UNITS.keys()),
        default=[],
        format_func=lambda uid: f"{TNU_UNITS[uid][0]} — {TNU_UNITS[uid][1]}",
        placeholder="Tất cả 9 đơn vị thành viên",
        key="sb_units",
    )

    selected_source_ids = st.sidebar.multiselect(
        "Nguồn thu thập",
        options=list(DATA_SOURCES.keys()),
        default=[],
        format_func=lambda sid: DATA_SOURCES[sid],
        placeholder="Tất cả nguồn thu thập",
        key="sb_sources",
    )

    selected_topic_ids = st.sidebar.multiselect(
        "Chủ đề phản ánh",
        options=list(TOPIC_ID_TO_NAME.keys()),
        default=[],
        format_func=lambda tid: TOPIC_ID_TO_NAME[tid],
        placeholder="Tất cả 4 chủ đề",
        key="sb_topics",
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Thông tin Đồ án & Cấu hình", expanded=False):
        st.markdown(
            "- **SV:** Nguyễn Văn Chiến (`DTC235200079`)\n"
            "- **Lớp:** CN CNTT K22A — Trường ĐH CNTT&TT (ICTU)\n"
            "- **GVHD:** TS. Hà Thị Thanh\n"
            f"- **Gemini Router:** `{'Đã bật API Key' if is_gemini_configured() else 'Keyword Fallback'}`"
        )

    # Khởi tạo 4 Tab chính theo đặc tả Enterprise UI/UX
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "💬 Tab 1: Phân tích Trực tiếp & Kiểm thử Mô hình",
            "📊 Tab 2: Dashboard Giám sát Cảm xúc Toàn ĐHTN",
            "📁 Tab 3: Quản lý, Import Hàng loạt & Kiểm duyệt",
            "🏆 Tab 4: Đánh giá & Đối sánh Chất lượng Mô hình",
        ]
    )

    # ==========================================================================
    # TAB 1: 💬 PHÂN TÍCH TRỰC TIẾP & KIỂM THỬ MÔ HÌNH (LIVE AI SANDBOX)
    # ==========================================================================
    with tab1:
        col_left, col_right = st.columns([0.45, 0.55], gap="large")

        with col_left:
            with st.container(border=True):
                st.markdown("#### 1. Nhập Phản hồi & Chọn Ca Kiểm thử Điển hình")

                sample_keys = ["— Tự nhập ý kiến phản hồi mới —"] + list(SAMPLE_TEST_CASES.keys())

                def _on_sample_selected() -> None:
                    sel = st.session_state.get("t1_sample_choice", sample_keys[0])
                    if sel in SAMPLE_TEST_CASES:
                        item = SAMPLE_TEST_CASES[sel]
                        st.session_state["t1_raw_text"] = item["text"]
                        st.session_state["t1_unit_code"] = item["unit_code"]
                        st.session_state["t1_source_id"] = item["source_id"]

                st.selectbox(
                    "Chọn nhanh 6 câu mẫu kiểm thử điển hình",
                    options=sample_keys,
                    key="t1_sample_choice",
                    on_change=_on_sample_selected,
                )

                if "t1_raw_text" not in st.session_state:
                    st.session_state["t1_raw_text"] = SAMPLE_TEST_CASES[
                        "1. Câu khen có teencode & emoji"
                    ]["text"]

                raw_input = st.text_area(
                    "Nội dung ý kiến đánh giá của người học (Tiếng Việt)",
                    height=140,
                    key="t1_raw_text",
                )

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    unit_codes = [code for code, _ in TNU_UNITS.values()]
                    chosen_unit_code = st.selectbox(
                        "Đơn vị thành viên (9 đơn vị ĐHTN)",
                        options=unit_codes,
                        format_func=lambda c: f"{c} — {TNU_UNITS[UNIT_CODE_TO_ID[c]][1]}",
                        key="t1_unit_code",
                    )
                with m_col2:
                    chosen_source_id = st.selectbox(
                        "Nguồn thu thập",
                        options=list(DATA_SOURCES.keys()),
                        format_func=lambda sid: DATA_SOURCES[sid],
                        key="t1_source_id",
                    )

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    analyze_btn = st.button(
                        "🚀 Phân tích Cảm xúc",
                        type="primary",
                        width="stretch",
                        disabled=(len(raw_input.strip()) == 0),
                    )
                with b_col2:
                    last_res: Optional[Dict[str, Any]] = st.session_state.get("t1_last_result")
                    save_btn = st.button(
                        "💾 Lưu vào CSDL",
                        width="stretch",
                        disabled=(last_res is None or not raw_input.strip()),
                    )

                if analyze_btn and raw_input.strip():
                    st.session_state["t1_last_result"] = predict_feedback_sentiment_and_topic(
                        raw_text=raw_input,
                        preprocessor=preprocessor,
                        model_bundle=model_bundle,
                    )

                if save_btn and last_res is not None:
                    ok, msg = save_feedback_record(
                        engine=engine,
                        preprocessor=preprocessor,
                        result=last_res,
                        unit_id=UNIT_CODE_TO_ID[chosen_unit_code],
                        source_id=int(chosen_source_id),
                        is_synthetic=0,
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.warning(msg)

        with col_right:
            with st.container(border=True):
                st.markdown("#### 2. Kết quả Suy luận AI & Tường minh Pipeline NLP 6 Bước")
                res: Optional[Dict[str, Any]] = st.session_state.get("t1_last_result")
                if res is None:
                    res = predict_feedback_sentiment_and_topic(
                        raw_text=raw_input,
                        preprocessor=preprocessor,
                        model_bundle=model_bundle,
                    )
                    st.session_state["t1_last_result"] = res

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Nhãn Cảm xúc", res["sentiment_vi"])
                mc2.metric("Chủ đề Phản ánh", res["topic_name"])
                mc3.metric(
                    "Độ tự tin & Latency",
                    f"{res['confidence_score'] * 100:.2f}% ({res['latency_ms']} ms)",
                )

                if res["confidence_score"] < 0.65 or res["review_status"] == "NEEDS_REVIEW":
                    st.warning(
                        f"⚠️ **Cảnh báo Độ tự tin thấp (`confidence = {res['confidence_score']:.2%}` < `65%`):** "
                        f"Bản ghi tự động gắn cờ `review_status = 'NEEDS_REVIEW'` và kích hoạt định tuyến lai "
                        f"(`{res['topic_router_source']}`)."
                    )
                    if res.get("router_info"):
                        ri = res["router_info"]
                        st.info(
                            f"**Gợi ý từ Hybrid Router ({ri['router_mode']}):** "
                            f"Cảm xúc `{ri['suggested_sentiment_text']}` · Chủ đề `{ri['suggested_topic_name']}` — "
                            f"{ri['explanation']}"
                        )

                # Biểu đồ thanh ngang phân bố xác suất Softmax 3 lớp
                prob_df = pd.DataFrame(
                    [
                        {
                            "Lớp cảm xúc": "Tiêu cực (Negative)",
                            "Xác suất (%)": round(res["prob_negative"] * 100, 2),
                        },
                        {
                            "Lớp cảm xúc": "Trung tính (Neutral)",
                            "Xác suất (%)": round(res["prob_neutral"] * 100, 2),
                        },
                        {
                            "Lớp cảm xúc": "Tích cực (Positive)",
                            "Xác suất (%)": round(res["prob_positive"] * 100, 2),
                        },
                    ]
                )
                fig_prob = px.bar(
                    prob_df,
                    x="Xác suất (%)",
                    y="Lớp cảm xúc",
                    color="Lớp cảm xúc",
                    orientation="h",
                    text="Xác suất (%)",
                    color_discrete_map=SENTIMENT_COLOR_MAP,
                )
                fig_prob.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig_prob.update_layout(showlegend=False)
                fig_prob.update_xaxes(range=[0, 112])
                st.plotly_chart(
                    style_plotly(fig_prob, "Phân bố Xác suất Softmax 3 Lớp Cảm xúc", height=230),
                    width="stretch",
                )

                st.markdown("##### 🔍 Kết quả tường minh từng bước trong Pipeline NLP 6 Bước (`src/preprocessor.py`)")
                steps_dict = res["steps"]
                step_labels = [
                    ("Bước 1 — Chuẩn hóa Unicode NFC, xóa URL/HTML & Ẩn danh PII (Nghị định 13/2023/NĐ-CP)", "step1_nfc_pii"),
                    ("Bước 2 — Ánh xạ Emoji & Emoticon (ưu tiên khóa dài `:)))`, `:((` trước `:)`)", "step2_emoji"),
                    ("Bước 3 — Thu gọn ký tự lặp kéo dài (`siêuuuu` → `siêu`) & Chuyển chữ thường", "step3_elongated"),
                    ("Bước 4 & 5 — Giải mã Teencode học đường ĐHTN & Chuẩn hóa thanh điệu tiếng Việt", "step4_5_teencode_tone"),
                    ("Bước 6 — Tách từ tiếng Việt bằng PyVi (`ViTokenizer`) đưa vào PhoBERT", "step6_segmented"),
                ]
                for title_step, key_step in step_labels:
                    val_step = html.escape(str(steps_dict.get(key_step, "")))
                    st.markdown(
                        f'<div class="tnu-step-box"><strong>{html.escape(title_step)}:</strong><br/><code>{val_step}</code></div>',
                        unsafe_allow_html=True,
                    )
                st.caption(f"Mã băm chống trùng lặp SHA-256 (`content_hash`): `{res['content_hash']}`")

    # ==========================================================================
    # TAB 2: 📊 DASHBOARD GIÁM SÁT CẢM XÚC TOÀN ĐẠI HỌC THÁI NGUYÊN
    # ==========================================================================
    with tab2:
        df_filtered = query_filtered_feedbacks(
            engine=engine,
            preprocessor=preprocessor,
            data_mode=data_mode,
            start_date=start_date,
            end_date=end_date,
            selected_unit_ids=selected_unit_ids,
            selected_source_ids=selected_source_ids,
            selected_topic_ids=selected_topic_ids,
        )

        if data_mode != "Chỉ dữ liệu thực tế (is_synthetic = 0)":
            st.caption(
                "ℹ️ Lưu ý học thuật: Bộ lọc hiện bao gồm các bản ghi mô phỏng/Seed Demo (`is_synthetic = 1`). "
                "Chọn **'Chỉ dữ liệu thực tế (is_synthetic = 0)'** ở Sidebar khi thống kê riêng phản hồi thu thập thực tế."
            )

        n_total = len(df_filtered)
        if n_total == 0:
            st.info(
                "Chưa có bản ghi phản hồi nào khớp với bộ lọc hiện tại (Empty State an toàn — không xảy ra lỗi chia cho 0)."
            )
        else:
            # Ưu tiên nhãn đã kiểm duyệt (reviewed_sentiment_label) nếu có, ngược lại dùng nhãn AI (sentiment_label)
            eff_sent = (
                df_filtered["reviewed_sentiment_label"]
                .fillna(df_filtered["sentiment_label"])
                .astype(int)
            )
            df_filtered = df_filtered.assign(
                effective_sentiment=eff_sent,
                sentiment_vi=eff_sent.map(
                    {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
                ),
            )

            n_pos = int((eff_sent == 2).sum())
            n_neu = int((eff_sent == 1).sum())
            n_neg = int((eff_sent == 0).sum())
            pos_pct = round(100.0 * n_pos / n_total, 2)
            neg_pct = round(100.0 * n_neg / n_total, 2)
            nss_score = round(pos_pct - neg_pct, 2)
            avg_conf = float(df_filtered["confidence_score"].mean()) * 100.0

            # Hàng 1: 4 thẻ KPI Scorecards
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_scorecard(
                    "1. Tổng số ý kiến hợp lệ (N)",
                    f"{n_total:,}",
                    f"Trung tính: {n_neu:,} mẫu · Độ tự tin AI TB: {avg_conf:.1f}%",
                    PALETTE["navy"],
                )
            with k2:
                render_scorecard(
                    "2. Tỷ lệ Tích cực (% Positive)",
                    f"{pos_pct:.2f}%",
                    f"{n_pos:,} / {n_total:,} ý kiến phản hồi",
                    PALETTE["positive"],
                )
            with k3:
                render_scorecard(
                    "3. Tỷ lệ Tiêu cực (% Negative)",
                    f"{neg_pct:.2f}%",
                    f"{n_neg:,} / {n_total:,} ý kiến phản hồi",
                    PALETTE["negative"],
                )
            with k4:
                render_scorecard(
                    "4. Chỉ số Cảm xúc Ròng (NSS)",
                    f"{nss_score:+.2f}",
                    f"Công thức: %Positive - %Negative · Confidence TB: {avg_conf:.2f}%",
                    PALETTE["positive"] if nss_score >= 0 else PALETTE["negative"],
                )

            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

            # Hàng 2: Xu hướng cảm xúc theo thời gian (px.line) & Cơ cấu theo nguồn thu thập (px.bar)
            r2_c1, r2_c2 = st.columns(2, gap="medium")
            with r2_c1:
                with st.container(border=True):
                    trend_df = (
                        df_filtered.assign(
                            day=df_filtered["feedback_date"].dt.strftime("%Y-%m-%d")
                        )
                        .groupby(["day", "sentiment_vi"], as_index=False)
                        .size()
                        .rename(columns={"size": "Số lượng"})
                    )
                    fig_line = px.line(
                        trend_df,
                        x="day",
                        y="Số lượng",
                        color="sentiment_vi",
                        markers=True,
                        color_discrete_map=SENTIMENT_COLOR_MAP,
                    )
                    st.plotly_chart(
                        style_plotly(fig_line, "1. Xu hướng Cảm xúc theo Thời gian"),
                        width="stretch",
                    )

            with r2_c2:
                with st.container(border=True):
                    src_df = (
                        df_filtered.groupby(["source_name", "sentiment_vi"], as_index=False)
                        .size()
                        .rename(columns={"size": "Số lượng"})
                    )
                    fig_src = px.bar(
                        src_df,
                        x="Số lượng",
                        y="source_name",
                        color="sentiment_vi",
                        orientation="h",
                        barmode="stack",
                        color_discrete_map=SENTIMENT_COLOR_MAP,
                    )
                    st.plotly_chart(
                        style_plotly(fig_src, "2. Cơ cấu Cảm xúc theo Nguồn thu thập"),
                        width="stretch",
                    )

            # Hàng 3: Phân bố cảm xúc theo 4 chủ đề & So sánh giữa 9 trường/đơn vị ĐHTN
            r3_c1, r3_c2 = st.columns(2, gap="medium")
            with r3_c1:
                with st.container(border=True):
                    top_df = (
                        df_filtered.groupby(["topic_name", "sentiment_vi"], as_index=False)
                        .size()
                        .rename(columns={"size": "Số lượng"})
                    )
                    fig_top = px.bar(
                        top_df,
                        x="topic_name",
                        y="Số lượng",
                        color="sentiment_vi",
                        barmode="group",
                        color_discrete_map=SENTIMENT_COLOR_MAP,
                    )
                    st.plotly_chart(
                        style_plotly(fig_top, "3. Phân bố Cảm xúc theo 4 Chủ đề Phản ánh"),
                        width="stretch",
                    )

            with r3_c2:
                with st.container(border=True):
                    unit_df = (
                        df_filtered.groupby(["unit_code", "sentiment_vi"], as_index=False)
                        .size()
                        .rename(columns={"size": "Số lượng"})
                    )
                    fig_unit = px.bar(
                        unit_df,
                        x="unit_code",
                        y="Số lượng",
                        color="sentiment_vi",
                        barmode="stack",
                        color_discrete_map=SENTIMENT_COLOR_MAP,
                    )
                    st.plotly_chart(
                        style_plotly(
                            fig_unit,
                            "4. So sánh Phản hồi giữa 9 Trường / Đơn vị Thành viên ĐHTN",
                        ),
                        width="stretch",
                    )

            # Hàng 4: Bảng cảnh báo sớm (Early Warning Data Grid)
            st.markdown("#### 🚨 Bảng Cảnh báo Sớm (Early Warning Data Grid — Các phản hồi Tiêu cực độ tự tin cao nhất)")
            neg_grid = (
                df_filtered[df_filtered["effective_sentiment"] == 0]
                .sort_values("confidence_score", ascending=False)
                .head(20)
            )
            if neg_grid.empty:
                st.success("Không có phản hồi tiêu cực nào trong phạm vi bộ lọc hiện tại.")
            else:
                st.dataframe(
                    neg_grid[
                        [
                            "feedback_id",
                            "feedback_date",
                            "unit_code",
                            "topic_name",
                            "source_name",
                            "raw_content",
                            "confidence_score",
                            "review_status",
                        ]
                    ].rename(
                        columns={
                            "feedback_id": "ID",
                            "feedback_date": "Thời gian",
                            "unit_code": "Đơn vị",
                            "topic_name": "Chủ đề",
                            "source_name": "Nguồn",
                            "raw_content": "Nội dung (Đã che PII)",
                            "confidence_score": "Độ tự tin AI",
                            "review_status": "Trạng thái duyệt",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    # ==========================================================================
    # TAB 3: 📁 QUẢN LÝ, IMPORT HÀNG LOẠT & KIỂM DUYỆT (HUMAN-IN-THE-LOOP)
    # ==========================================================================
    with tab3:
        sub_import, sub_review = st.tabs(
            [
                "📥 Sub-tab 3.1: Batch Import (Nhập hàng loạt CSV/XLSX)",
                "✍️ Sub-tab 3.2: Active Learning Review (Hậu kiểm các ca độ tự tin thấp)",
            ]
        )

        with sub_import:
            with st.container(border=True):
                st.markdown("#### Tải File Mẫu & Import Hàng loạt có Kiểm tra Trùng lặp (`content_hash`)")
                sample_csv_df = pd.DataFrame(
                    [
                        {
                            "sentence": "Giảng viên hướng dẫn thực hành môn AI rất tận tình, dễ hiểu.",
                            "unit_code": "ICTU",
                            "source_name": "SURVEY_PORTAL",
                            "feedback_date": "2026-10-20 08:30:00",
                        },
                        {
                            "sentence": "Phòng học nhà C4 máy chiếu bị mờ và quạt trần kêu to quá.",
                            "unit_code": "TNUT",
                            "source_name": "QA_IMPORT_BOX",
                            "feedback_date": "2026-10-21 14:15:00",
                        },
                    ]
                )
                st.download_button(
                    "⬇️ Tải file CSV mẫu (sentence, unit_code, source_name, feedback_date)",
                    data=sample_csv_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="tnu_batch_import_template.csv",
                    mime="text/csv",
                )

                up_file = st.file_uploader(
                    "Chọn tệp dữ liệu phản hồi (.csv hoặc .xlsx)",
                    type=["csv", "xlsx"],
                    key="t3_batch_uploader",
                )
                if up_file is not None:
                    try:
                        if up_file.name.lower().endswith(".xlsx"):
                            imp_df = pd.read_excel(io.BytesIO(up_file.getvalue()))
                        else:
                            imp_df = pd.read_csv(io.BytesIO(up_file.getvalue()))
                    except Exception as exc:
                        st.error(f"Không thể đọc tệp tải lên: {exc}")
                        imp_df = pd.DataFrame()

                    required_cols = {"sentence", "unit_code", "source_name", "feedback_date"}
                    missing_cols = required_cols - set(imp_df.columns)
                    if missing_cols:
                        st.error(f"Tệp thiếu các cột bắt buộc: {', '.join(sorted(missing_cols))}")
                    elif not imp_df.empty:
                        st.dataframe(imp_df.head(15), width="stretch", hide_index=True)
                        if st.button(
                            f"🚀 Chạy Phân tích & Import Hàng loạt ({len(imp_df)} dòng)",
                            type="primary",
                        ):
                            prog = st.progress(0.0, text="Đang xử lý hàng loạt...")
                            saved_n, dup_n, err_n = 0, 0, 0
                            total_rows = len(imp_df)
                            for i, r in imp_df.iterrows():
                                txt = str(r.get("sentence", "")).strip()
                                if not txt:
                                    err_n += 1
                                    continue
                                u_code = str(r.get("unit_code", "ICTU")).strip().upper()
                                u_id = UNIT_CODE_TO_ID.get(u_code, 1)
                                s_name = str(r.get("source_name", "CSV_IMPORT")).strip().upper()
                                s_id = SOURCE_NAME_TO_ID.get(s_name, 4)
                                pred_res = predict_feedback_sentiment_and_topic(
                                    raw_text=txt,
                                    preprocessor=preprocessor,
                                    model_bundle=model_bundle,
                                )
                                ok, _ = save_feedback_record(
                                    engine=engine,
                                    preprocessor=preprocessor,
                                    result=pred_res,
                                    unit_id=u_id,
                                    source_id=s_id,
                                    is_synthetic=0,
                                )
                                if ok:
                                    saved_n += 1
                                else:
                                    dup_n += 1
                                prog.progress(
                                    (i + 1) / max(1, total_rows),
                                    text=f"Đã xử lý {i + 1}/{total_rows} dòng...",
                                )
                            st.success(
                                f"Hoàn tất Batch Import: Đã lưu mới **{saved_n}** dòng · "
                                f"Bỏ qua do trùng `content_hash`: **{dup_n}** dòng · Lỗi dữ liệu: **{err_n}** dòng."
                            )

        with sub_review:
            with st.container(border=True):
                st.markdown(
                    "#### Hàng đợi Hậu kiểm Chủ động (Active Learning — `confidence_score < 0.65` hoặc `NEEDS_REVIEW`)"
                )
                st.caption(
                    "Giữ nguyên nhãn dự đoán gốc (`sentiment_label`, `topic_id`) để phân tích lỗi mô hình; "
                    "cập nhật nhãn hậu kiểm vào `reviewed_sentiment_label`, `reviewed_topic_id` và chuyển `review_status = 'VERIFIED'`."
                )

                all_df = query_filtered_feedbacks(
                    engine=engine,
                    preprocessor=preprocessor,
                    data_mode=data_mode,
                    start_date=start_date,
                    end_date=end_date,
                    selected_unit_ids=selected_unit_ids,
                    selected_source_ids=selected_source_ids,
                    selected_topic_ids=selected_topic_ids,
                )
                review_queue = all_df[
                    (all_df["confidence_score"] < 0.65)
                    | (all_df["review_status"] == "NEEDS_REVIEW")
                ].head(50)

                if review_queue.empty:
                    st.info("Hiện không có bản ghi nào có `confidence_score < 0.65` hoặc ở trạng thái `NEEDS_REVIEW`.")
                else:
                    reviewer_user = st.text_input(
                        "Cán bộ hậu kiểm", value="admin_ictu", key="t3_reviewer_user"
                    )
                    edit_source = pd.DataFrame(
                        {
                            "feedback_id": review_queue["feedback_id"],
                            "unit_code": review_queue["unit_code"],
                            "raw_content": review_queue["raw_content"],
                            "ai_sentiment": review_queue["sentiment_label"].map(SENTIMENT_ID_TO_TEXT),
                            "confidence_score": review_queue["confidence_score"],
                            "ai_topic": review_queue["topic_id"].map(TOPIC_ID_TO_NAME),
                            "review_status": review_queue["review_status"],
                            "reviewed_sentiment_label": review_queue["reviewed_sentiment_label"]
                            .fillna(review_queue["sentiment_label"])
                            .astype(int),
                            "reviewed_topic_id": review_queue["reviewed_topic_id"]
                            .fillna(review_queue["topic_id"])
                            .astype(int),
                            "review_note": review_queue["review_note"].fillna(""),
                        }
                    )

                    edited_queue = st.data_editor(
                        edit_source,
                        width="stretch",
                        hide_index=True,
                        disabled=[
                            "feedback_id",
                            "unit_code",
                            "raw_content",
                            "ai_sentiment",
                            "confidence_score",
                            "ai_topic",
                        ],
                        column_config={
                            "feedback_id": st.column_config.NumberColumn("ID", width="small"),
                            "unit_code": st.column_config.TextColumn("Đơn vị", width="small"),
                            "raw_content": st.column_config.TextColumn("Nội dung phản hồi", width="large"),
                            "ai_sentiment": st.column_config.TextColumn("AI Dự đoán (Gốc)"),
                            "confidence_score": st.column_config.NumberColumn("Confidence", format="%.4f"),
                            "ai_topic": st.column_config.TextColumn("Chủ đề AI (Gốc)"),
                            "review_status": st.column_config.SelectboxColumn(
                                "Trạng thái duyệt ✎",
                                options=["NEEDS_REVIEW", "VERIFIED", "AUTO_APPROVED"],
                                required=True,
                            ),
                            "reviewed_sentiment_label": st.column_config.SelectboxColumn(
                                "Nhãn cảm xúc duyệt (0:Neg, 1:Neu, 2:Pos) ✎",
                                options=[0, 1, 2],
                                required=True,
                            ),
                            "reviewed_topic_id": st.column_config.SelectboxColumn(
                                "Mã chủ đề duyệt (1..4) ✎",
                                options=[1, 2, 3, 4],
                                required=True,
                            ),
                            "review_note": st.column_config.TextColumn("Ghi chú hậu kiểm ✎"),
                        },
                        key="t3_active_learning_editor",
                    )

                    if st.button("💾 Lưu Cập nhật Hậu kiểm (Active Learning)", type="primary"):
                        updated_count = 0
                        now_dt = datetime.now()
                        if engine is not None:
                            upd_stmt = text(
                                """
                                UPDATE student_feedbacks
                                SET review_status = :r_status,
                                    reviewed_sentiment_label = :r_sent,
                                    reviewed_topic_id = :r_topic,
                                    reviewed_by = :r_by,
                                    reviewed_at = :r_at,
                                    review_note = :r_note,
                                    is_verified = CASE WHEN :r_status = 'VERIFIED' THEN 1 ELSE is_verified END
                                WHERE feedback_id = :fid
                                """
                            )
                            with engine.begin() as conn:
                                for _, erow in edited_queue.iterrows():
                                    conn.execute(
                                        upd_stmt,
                                        {
                                            "r_status": str(erow["review_status"]),
                                            "r_sent": int(erow["reviewed_sentiment_label"]),
                                            "r_topic": int(erow["reviewed_topic_id"]),
                                            "r_by": reviewer_user,
                                            "r_at": now_dt,
                                            "r_note": str(erow["review_note"]),
                                            "fid": int(erow["feedback_id"]),
                                        },
                                    )
                                    updated_count += 1
                        else:
                            df_mem = _load_offline_seed_dataframe(preprocessor)
                            for _, erow in edited_queue.iterrows():
                                fid = int(erow["feedback_id"])
                                idx_mask = df_mem["feedback_id"] == fid
                                df_mem.loc[idx_mask, "review_status"] = str(erow["review_status"])
                                df_mem.loc[idx_mask, "reviewed_sentiment_label"] = int(
                                    erow["reviewed_sentiment_label"]
                                )
                                df_mem.loc[idx_mask, "reviewed_topic_id"] = int(
                                    erow["reviewed_topic_id"]
                                )
                                df_mem.loc[idx_mask, "reviewed_by"] = reviewer_user
                                df_mem.loc[idx_mask, "review_note"] = str(erow["review_note"])
                                updated_count += 1
                        st.success(f"Đã cập nhật hậu kiểm thành công cho {updated_count} bản ghi!")

    # ==========================================================================
    # TAB 4: 🏆 ĐÁNH GIÁ & ĐỐI SÁNH CHẤT LƯỢNG MÔ HÌNH (MODEL BENCHMARK & ERROR ANALYSIS)
    # ==========================================================================
    with tab4:
        st.markdown("### 🏆 Kết quả Thực nghiệm & Đối sánh Mô hình (Tách biệt Tập Validation & Tập Test)")

        st.info(
            "**Ghi chú học thuật (Tương thích Đề cương H02 — PI 3.1):** Do bài toán phân tích cảm xúc và phân loại chủ đề "
            "là bài toán **phân loại rời rạc đa lớp (Multi-class Classification)** trên dữ liệu mất cân bằng (lớp Neutral chỉ chiếm **4,32%**), "
            "đồ án sử dụng bộ chỉ số **Accuracy, Precision, Recall, Macro-F1, Weighted-F1 và Confusion Matrix**; "
            "không sử dụng độ đo sai số tuyệt đối trung bình (MAE) vốn chỉ áp dụng cho bài toán hồi quy liên tục."
        )

        # Bảng 3.1: Tập Kiểm thử độc lập UIT-VSFC (Test Split: N = 3.166 mẫu)
        st.markdown(
            "#### Bảng 3.1: Kết quả đối sánh chính thức trên Tập Kiểm thử độc lập UIT-VSFC (`Evaluation Split = Test (3.166 mẫu)`)"
        )
        test_bench_df = pd.DataFrame(
            [
                {
                    "Mô hình thực nghiệm": "Naive Bayes + TF-IDF",
                    "Evaluation Split": "Test (3.166 mẫu)",
                    "Accuracy (%)": 86.83,
                    "Macro Precision (%)": 59.01,
                    "Macro Recall (%)": 60.00,
                    "Macro-F1 (%)": 59.45,
                    "Weighted-F1 (%)": 84.51,
                    "F1 - Negative (%)": 89.93,
                    "F1 - Neutral (%)": 0.00,
                    "F1 - Positive (%)": 88.41,
                },
                {
                    "Mô hình thực nghiệm": "Maximum Entropy + TF-IDF",
                    "Evaluation Split": "Test (3.166 mẫu)",
                    "Accuracy (%)": 89.17,
                    "Macro Precision (%)": 78.61,
                    "Macro Recall (%)": 63.34,
                    "Macro-F1 (%)": 64.99,
                    "Weighted-F1 (%)": 87.28,
                    "F1 - Negative (%)": 91.54,
                    "F1 - Neutral (%)": 12.09,
                    "F1 - Positive (%)": 91.34,
                },
                {
                    "Mô hình thực nghiệm": "Linear SVM (Balanced) + TF-IDF",
                    "Evaluation Split": "Test (3.166 mẫu)",
                    "Accuracy (%)": 89.29,
                    "Macro Precision (%)": 72.89,
                    "Macro Recall (%)": 74.01,
                    "Macro-F1 (%)": 73.41,
                    "Weighted-F1 (%)": 88.85,
                    "F1 - Negative (%)": 91.89,
                    "F1 - Neutral (%)": 36.75,
                    "F1 - Positive (%)": 91.58,
                },
                {
                    "Mô hình thực nghiệm": "PhoBERT-base-v2 + Focal Loss (Đợt 1)",
                    "Evaluation Split": "Test (3.166 mẫu)",
                    "Accuracy (%)": 93.78,
                    "Macro Precision (%)": 84.12,
                    "Macro Recall (%)": 81.95,
                    "Macro-F1 (%)": 82.91,
                    "Weighted-F1 (%)": 93.46,
                    "F1 - Negative (%)": 95.23,
                    "F1 - Neutral (%)": 57.86,
                    "F1 - Positive (%)": 95.63,
                },
            ]
        )
        st.dataframe(test_bench_df, width="stretch", hide_index=True)

        st.markdown(
            "> **Nhận xét học thuật (Tránh ngụy biện toán học $1 - \\text{F1}$):** "
            "Chỉ số $\\text{F1-score}$ là trung bình điều hòa của $\\text{Precision}$ và $\\text{Recall}$ "
            "($\\text{F1} = \\frac{2 \\cdot \\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$), "
            "trong khi tỷ lệ bỏ sót mẫu thực tế của một lớp $c$ phải được tính bằng $1 - \\text{Recall}_c = \\frac{\\text{FN}_c}{\\text{TP}_c + \\text{FN}_c}$. "
            "Do lớp Trung tính (Neutral) chỉ chiếm **4,32%** tập dữ liệu, mô hình Naive Bayes hoàn toàn không học được ranh giới quyết định của lớp này "
            "(`Recall-Neutral = 0,00%`, `F1-Neutral = 0,00%`), trong khi Maximum Entropy cũng bị hai lớp đa số lấn át khiến `F1-Neutral` chỉ đạt `12,09%` trên tập Test. "
            "Khi áp dụng cơ chế cân bằng trọng số lớp ở Linear SVM (`class_weight='balanced'`), `F1-Neutral` tăng gấp hơn 3 lần lên `36,75%` (`Macro-F1` đạt `73,41%`). "
            "Khi chuyển sang kiến trúc Transformer tiền huấn luyện trên **140GB văn bản tiếng Việt** (20GB Wikipedia/tin tức + 120GB OSCAR-2301) "
            "kết hợp hàm mất mát **Multi-Class Focal Loss** ($\\gamma = 2,0$, label smoothing $\\epsilon = 0,05$), `F1-Neutral` trên tập Test tăng vọt lên **57,86%** (`Macro-F1` đạt **82,91%**)."
        )

        # Bảng 3.2: Tập Phát triển mở rộng (Validation Split: N = 1.709 mẫu - Đỉnh Epoch 4)
        st.markdown(
            "#### Bảng 3.2: Nhật ký tinh chỉnh sâu PhoBERT-base-v2 + Focal Loss trên Tập Phát triển mở rộng (`Validation Split: N = 1.709 mẫu`)"
        )
        val_bench_df = pd.DataFrame(
            [
                {
                    "Mô hình thực nghiệm": "PhoBERT-base-v2 + Focal Loss (Đợt 2 - Đỉnh Epoch 4)",
                    "Evaluation Split": "Validation Mở rộng (1.709 mẫu)",
                    "Cấu hình Huấn luyện": "num_train_epochs=20, EarlyStopping(patience=6, metric='macro_f1'), dừng tại Epoch 10, đỉnh tại Epoch 4",
                    "Accuracy (%)": 95.61,
                    "Macro Precision (%)": 91.17,
                    "Macro Recall (%)": 91.10,
                    "Macro-F1 (%)": 91.13,
                    "Weighted-F1 (%)": 95.57,
                    "F1 - Negative (%)": 96.74,
                    "F1 - Neutral (%)": 80.00,
                    "F1 - Positive (%)": 96.65,
                },
            ]
        )
        st.dataframe(val_bench_df, width="stretch", hide_index=True)

        st.caption(
            "Quy mô dữ liệu chuẩn hóa: Tập UIT-VSFC gốc gồm **16.175 câu** (`11.426` Train + `1.583` Validation + `3.166` Test). "
            "Đồ án bổ sung **840 mẫu câu mang từ vựng miền ĐHTN** (chia 85% = `714` mẫu vào Train → `12.140` mẫu; "
            "15% = `126` mẫu vào Validation → `1.709` mẫu; giữ nguyên `3.166` mẫu Test độc lập), nâng tổng quy mô lên **17.015 câu**."
        )

        # Biểu đồ cột nhóm Plotly so sánh trực diện các mô hình trên Tập Test (3.166 mẫu) và Tập Validation (1.709 mẫu)
        b_col1, b_col2 = st.columns([0.52, 0.48], gap="medium")
        with b_col1:
            with st.container(border=True):
                melt_df = test_bench_df.melt(
                    id_vars=["Mô hình thực nghiệm"],
                    value_vars=["Accuracy (%)", "Macro-F1 (%)", "Weighted-F1 (%)", "F1 - Neutral (%)"],
                    var_name="Chỉ số đánh giá",
                    value_name="Điểm số (%)",
                )
                fig_bench = px.bar(
                    melt_df,
                    x="Chỉ số đánh giá",
                    y="Điểm số (%)",
                    color="Mô hình thực nghiệm",
                    barmode="group",
                    text="Điểm số (%)",
                    color_discrete_sequence=[
                        "#94A3B8",
                        "#64748B",
                        "#0EA5E9",
                        PALETTE["navy"],
                    ],
                )
                fig_bench.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig_bench.update_yaxes(range=[0, 110])
                st.plotly_chart(
                    style_plotly(
                        fig_bench,
                        "Đối sánh các Mô hình trên Tập Kiểm thử Độc lập UIT-VSFC (N = 3.166)",
                        height=390,
                    ),
                    width="stretch",
                )

        with b_col2:
            with st.container(border=True):
                cm_img_path = PROJECT_ROOT / "confusion_matrix_phobert_tnu.png"
                if cm_img_path.exists():
                    st.image(
                        str(cm_img_path),
                        caption="Ma trận nhầm lẫn (Confusion Matrix) của PhoBERT-base-v2 + Focal Loss trên Tập Test (3.166 mẫu) và Tập Validation Mở rộng (1.709 mẫu)",
                        width="stretch",
                    )


if __name__ == "__main__":
    main()
