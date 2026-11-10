"""Quản lý cấu hình ứng dụng từ biến môi trường và tệp .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from tnu_sentiment.constants import AppMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    """Cấu hình tập trung cho toàn bộ các tầng Repository, Service, Model và UI."""

    default_mode: AppMode
    timezone_name: str
    database_url: str | None
    mysql_host: str
    mysql_port: int
    mysql_database: str
    mysql_user: str
    mysql_configured: bool

    sentiment_checkpoint_dir: Path | None
    sentiment_model_version: str
    sentiment_max_length: int

    topic_checkpoint_dir: Path | None
    topic_model_version: str

    confidence_review_threshold: float
    negative_alert_ratio_threshold: float
    negative_alert_min_samples: int

    max_input_chars: int
    import_max_file_mb: int
    import_max_rows: int
    page_size_negative_review: int
    page_size_data_table: int

    evaluation_artifacts_dir: Path
    fixtures_dir: Path

    secret_salt: str

    llm_enabled: bool
    llm_api_base_url: str
    llm_api_key: str
    llm_model_name: str
    llm_timeout_seconds: int

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone_name)

    def safe_db_summary(self) -> str:
        """Trả về chuỗi mô tả kết nối DB đã che hoàn toàn mật khẩu."""
        if not self.mysql_configured:
            return "Chưa cấu hình thông số kết nối MySQL (.env)"
        return f"mysql+pymysql://{self.mysql_user}:***@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"


def _resolve_optional_path(raw: str) -> Path | None:
    val = raw.strip()
    if not val:
        return None
    p = Path(val)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def load_app_config(env_file: Path | None = None) -> AppConfig:
    """Đọc cấu hình từ tệp .env (nếu có) và biến môi trường hệ thống."""
    target_env = env_file or (PROJECT_ROOT / ".env")
    if target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)

    raw_mode = os.getenv("APP_MODE", "demo").strip().upper()
    default_mode = AppMode.REAL if raw_mode == "REAL" else AppMode.DEMO

    tz_name = os.getenv("APP_TIMEZONE", "Asia/Ho_Chi_Minh").strip() or "Asia/Ho_Chi_Minh"

    explicit_db_url = os.getenv("DATABASE_URL", "").strip()
    mysql_host = os.getenv("MYSQL_HOST", "").strip()
    mysql_port = int(os.getenv("MYSQL_PORT", "3306").strip() or "3306")
    mysql_database = os.getenv("MYSQL_DATABASE", "").strip()
    mysql_user = os.getenv("MYSQL_USER", "").strip()
    mysql_password = os.getenv("MYSQL_PASSWORD", "")

    database_url: str | None = None
    mysql_configured = False
    if explicit_db_url:
        database_url = explicit_db_url
        mysql_configured = True
    elif mysql_host and mysql_database and mysql_user:
        encoded_user = quote_plus(mysql_user)
        encoded_pw = quote_plus(mysql_password)
        database_url = (
            f"mysql+pymysql://{encoded_user}:{encoded_pw}"
            f"@{mysql_host}:{mysql_port}/{mysql_database}?charset=utf8mb4"
        )
        mysql_configured = True

    default_ckpt_rel = "phobert_tnu_best_model-20261101T002214Z-1-001/phobert_tnu_best_model"
    raw_sentiment_ckpt = os.getenv("SENTIMENT_CHECKPOINT_DIR", default_ckpt_rel)
    sentiment_ckpt = _resolve_optional_path(raw_sentiment_ckpt)

    topic_ckpt = _resolve_optional_path(os.getenv("TOPIC_CHECKPOINT_DIR", ""))

    eval_dir_raw = os.getenv("EVALUATION_ARTIFACTS_DIR", "artifacts/evaluation").strip()
    eval_dir = _resolve_optional_path(eval_dir_raw) or (PROJECT_ROOT / "artifacts" / "evaluation")

    return AppConfig(
        default_mode=default_mode,
        timezone_name=tz_name,
        database_url=database_url,
        mysql_host=mysql_host or "chưa cấu hình",
        mysql_port=mysql_port,
        mysql_database=mysql_database or "chưa cấu hình",
        mysql_user=mysql_user or "chưa cấu hình",
        mysql_configured=mysql_configured,
        sentiment_checkpoint_dir=sentiment_ckpt,
        sentiment_model_version=os.getenv(
            "SENTIMENT_MODEL_VERSION", "PhoBERT-base-v2-Focal"
        ).strip(),
        sentiment_max_length=int(os.getenv("SENTIMENT_MAX_LENGTH", "256")),
        topic_checkpoint_dir=topic_ckpt,
        topic_model_version=os.getenv("TOPIC_MODEL_VERSION", "").strip(),
        confidence_review_threshold=float(
            os.getenv("CONFIDENCE_REVIEW_THRESHOLD", "0.65")
        ),
        negative_alert_ratio_threshold=float(
            os.getenv("NEGATIVE_ALERT_RATIO_THRESHOLD", "0.35")
        ),
        negative_alert_min_samples=int(
            os.getenv("NEGATIVE_ALERT_MIN_SAMPLES", "30")
        ),
        max_input_chars=int(os.getenv("MAX_INPUT_CHARS", "2000")),
        import_max_file_mb=int(os.getenv("IMPORT_MAX_FILE_MB", "10")),
        import_max_rows=int(os.getenv("IMPORT_MAX_ROWS", "5000")),
        page_size_negative_review=15,
        page_size_data_table=15,
        evaluation_artifacts_dir=eval_dir,
        fixtures_dir=(PROJECT_ROOT / "fixtures").resolve(),
        secret_salt=os.getenv("SECRET_SALT", "tnu-local-dev-salt-non-secret"),
        llm_enabled=os.getenv("LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes"},
        llm_api_base_url=os.getenv("LLM_API_BASE_URL", "").strip(),
        llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "").strip(),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "10")),
    )

