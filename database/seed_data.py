"""Script nạp dữ liệu khởi tạo (3.500 bản ghi mô phỏng từ tnu_mysql_seed_data.csv)
và bảng điểm thực nghiệm vào MySQL 8.0 theo đúng schema 3NF (database/schema.sql).

Chạy lệnh:
    python database/seed_data.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessor import TNUStudentFeedbackPreprocessor  # noqa: E402


def resolve_seed_csv_path() -> Path:
    """Tự động tìm tệp tnu_mysql_seed_data.csv trong dataset_final/ hoặc TNU_Sentiment_Dataset_Final/."""
    candidates = [
        PROJECT_ROOT / "dataset_final" / "tnu_mysql_seed_data.csv",
        PROJECT_ROOT / "TNU_Sentiment_Dataset_Final" / "tnu_mysql_seed_data.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Không tìm thấy tệp tnu_mysql_seed_data.csv trong dataset_final/ hoặc TNU_Sentiment_Dataset_Final/."
    )


def build_mysql_url() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    explicit_url = os.getenv("DATABASE_URL", "").strip()
    if explicit_url:
        return explicit_url
    host = os.getenv("MYSQL_HOST", "127.0.0.1").strip()
    port = os.getenv("MYSQL_PORT", "3306").strip()
    user = quote_plus(os.getenv("MYSQL_USER", "root").strip())
    password = quote_plus(os.getenv("MYSQL_PASSWORD", ""))
    db_name = os.getenv("MYSQL_DATABASE", "tnu_sentiment_db").strip()
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8mb4"


def parse_seed_date(raw_val: object) -> datetime:
    s = str(raw_val or "").strip()
    for fmt in ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.now()


def seed_database() -> None:
    db_url = build_mysql_url()
    seed_csv = resolve_seed_csv_path()
    schema_sql_path = PROJECT_ROOT / "database" / "schema.sql"

    print(f"[1/3] Kết nối MySQL và khởi tạo cấu trúc bảng từ: {schema_sql_path.name}")
    engine = create_engine(db_url, pool_pre_ping=True)

    sql_script = schema_sql_path.read_text(encoding="utf-8")
    statements = [stmt.strip() for stmt in sql_script.split(";") if stmt.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    print(f"[2/3] Đọc tệp dữ liệu Seed Demo: {seed_csv}")
    df = pd.read_csv(seed_csv, encoding="utf-8")
    preprocessor = TNUStudentFeedbackPreprocessor()

    insert_stmt = text(
        """
        INSERT IGNORE INTO student_feedbacks (
            content_hash, author_hash, unit_id, source_id, topic_id,
            raw_content, processed_content,
            sentiment_label, sentiment_text, confidence_score,
            prob_negative, prob_neutral, prob_positive,
            model_used, is_synthetic, is_verified,
            review_status, reviewed_sentiment_label, reviewed_topic_id,
            reviewed_by, reviewed_at, feedback_date
        ) VALUES (
            :content_hash, :author_hash, :unit_id, :source_id, :topic_id,
            :raw_content, :processed_content,
            :sentiment_label, :sentiment_text, :confidence_score,
            :prob_negative, :prob_neutral, :prob_positive,
            :model_used, :is_synthetic, :is_verified,
            :review_status, :reviewed_sentiment_label, :reviewed_topic_id,
            :reviewed_by, :reviewed_at, :feedback_date
        )
        """
    )

    batch_params = []
    seen_hashes: set[str] = set()
    skipped_dups = 0

    for idx, row in df.iterrows():
        raw_text = preprocessor.redact_pii_only(str(row.get("raw_content", "")))
        if not raw_text:
            continue
        c_hash = preprocessor.compute_content_hash(raw_text)
        if c_hash in seen_hashes:
            skipped_dups += 1
            continue
        seen_hashes.add(c_hash)

        proc_text = str(row.get("processed_content", "")).strip() or preprocessor.preprocess(raw_text)
        s_label = int(row.get("sentiment_label", 1))
        s_text = str(row.get("sentiment_text", "Neutral")).strip()
        conf = float(row.get("confidence_score", 0.80))
        p_neg = float(row.get("prob_negative", 0.0))
        p_neu = float(row.get("prob_neutral", 0.0))
        p_pos = float(row.get("prob_positive", 0.0))
        u_id = int(row.get("unit_id", 1))
        src_id = int(row.get("source_id", 1))
        t_id = int(row.get("topic_id", 1))
        verified_flag = int(row.get("is_verified", 0))
        fb_dt = parse_seed_date(row.get("feedback_date"))

        if verified_flag == 1:
            rev_status = "VERIFIED"
            rev_sent = s_label
            rev_topic = t_id
            rev_by = "seed_admin"
            rev_at = fb_dt
        elif conf < 0.65:
            rev_status = "NEEDS_REVIEW"
            rev_sent = None
            rev_topic = None
            rev_by = None
            rev_at = None
        else:
            rev_status = "AUTO_APPROVED"
            rev_sent = None
            rev_topic = None
            rev_by = None
            rev_at = None

        batch_params.append(
            {
                "content_hash": c_hash,
                "author_hash": str(row.get("author_hash", ""))[:64]
                or preprocessor.compute_author_hash(f"seed_{idx}"),
                "unit_id": u_id if 1 <= u_id <= 9 else 1,
                "source_id": src_id if 1 <= src_id <= 4 else 1,
                "topic_id": t_id if 1 <= t_id <= 4 else 4,
                "raw_content": raw_text,
                "processed_content": proc_text,
                "sentiment_label": s_label,
                "sentiment_text": s_text,
                "confidence_score": round(conf, 4),
                "prob_negative": round(p_neg, 4),
                "prob_neutral": round(p_neu, 4),
                "prob_positive": round(p_pos, 4),
                "model_used": str(row.get("model_used", "PhoBERT-base-v2-Focal")),
                "is_synthetic": 1,  # Đánh dấu rõ dữ liệu Seed Demo / mô phỏng (is_synthetic = 1)
                "is_verified": verified_flag,
                "review_status": rev_status,
                "reviewed_sentiment_label": rev_sent,
                "reviewed_topic_id": rev_topic,
                "reviewed_by": rev_by,
                "reviewed_at": rev_at,
                "feedback_date": fb_dt,
            }
        )

    print(
        f"[3/3] Thực thi nạp {len(batch_params):,} bản ghi duy nhất "
        f"(đã lọc {skipped_dups:,} dòng trùng content_hash)..."
    )
    with engine.begin() as conn:
        conn.execute(insert_stmt, batch_params)

    print("Hoàn tất nạp dữ liệu khởi tạo và bảng đối sánh thực nghiệm vào MySQL 8.0!")


if __name__ == "__main__":
    seed_database()

