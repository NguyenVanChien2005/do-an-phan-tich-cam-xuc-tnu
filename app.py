"""Điểm khởi chạy chính của ứng dụng Streamlit — TNU Sentiment."""

from __future__ import annotations

import importlib.metadata
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from tnu_sentiment.config import AppConfig, load_app_config
from tnu_sentiment.constants import AppMode
from tnu_sentiment.domain.models import FilterCriteria, SystemHealthStatus
from tnu_sentiment.models.llm_helper import OptionalLLMHelper
from tnu_sentiment.models.sentiment_adapter import (
    BaseSentimentAdapter,
    load_sentiment_adapter,
)
from tnu_sentiment.models.topic_adapter import load_topic_adapter
from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor
from tnu_sentiment.repositories.base import FeedbackRepository
from tnu_sentiment.repositories.demo_repo import DemoFeedbackRepository
from tnu_sentiment.repositories.evaluation_repo import FileEvaluationRepository
from tnu_sentiment.repositories.mysql_repo import MySQLFeedbackRepository
from tnu_sentiment.services.analysis_service import AnalysisService
from tnu_sentiment.services.benchmark_service import BenchmarkService
from tnu_sentiment.services.import_service import ImportService
from tnu_sentiment.services.review_service import ReviewService
from tnu_sentiment.services.stats_service import StatsService
from tnu_sentiment.ui.components import (
    ensure_shared_filters,
    render_app_header,
    render_page_footer,
    render_sidebar_project_info,
)
from tnu_sentiment.ui.pages.analyze import render_analyze_page
from tnu_sentiment.ui.pages.data_review import render_data_review_page
from tnu_sentiment.ui.pages.evaluation import render_evaluation_page
from tnu_sentiment.ui.pages.overview import render_overview_page
from tnu_sentiment.ui.theme import inject_academic_css


@st.cache_resource(show_spinner="Đang khởi tạo và kiểm tra checkpoint mô hình cảm xúc...")
def get_cached_sentiment_adapter(
    ckpt_path_str: str,
    model_version: str,
    max_length: int,
) -> BaseSentimentAdapter:
    ckpt_path = Path(ckpt_path_str) if ckpt_path_str else None
    return load_sentiment_adapter(
        checkpoint_dir=ckpt_path,
        model_version=model_version,
        max_length=max_length,
    )


@st.cache_resource(show_spinner=False)
def get_cached_mysql_engine(database_url: str) -> Tuple[Optional[Engine], Optional[str]]:
    """Chỉ lưu cache đối tượng Engine quản lý connection pool; KHÔNG chia sẻ Session giao dịch giữa người dùng."""
    if not database_url:
        return None, "Chưa cấu hình biến môi trường kết nối MySQL (DATABASE_URL / MYSQL_HOST)."
    try:
        eng = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            connect_args={"connect_timeout": 4},
        )
        return eng, None
    except Exception as exc:
        return None, f"Lỗi khởi tạo SQLAlchemy Engine: {exc}"


@st.cache_resource(show_spinner=False)
def get_cached_preprocessor() -> VietnamesePreprocessor:
    return VietnamesePreprocessor(require_segmenter=False)


def _pkg_ver(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "chưa cài đặt"


@st.cache_resource(show_spinner=False)
def get_environment_versions() -> Dict[str, str]:
    return {
        "Python": platform.python_version(),
        "Streamlit": _pkg_ver("streamlit"),
        "Plotly": _pkg_ver("plotly"),
        "SQLAlchemy": _pkg_ver("sqlalchemy"),
        "PyMySQL": _pkg_ver("pymysql"),
        "PyTorch": _pkg_ver("torch"),
        "Transformers": _pkg_ver("transformers"),
        "pyvi (Segmenter)": _pkg_ver("pyvi"),
        "OpenPyXL": _pkg_ver("openpyxl"),
    }


def _get_session_demo_repo(config: AppConfig) -> DemoFeedbackRepository:
    if "session_demo_repo" not in st.session_state:
        st.session_state["session_demo_repo"] = DemoFeedbackRepository(
            fixture_path=config.fixtures_dir / "demo_feedbacks.json",
            tz_name=config.timezone_name,
        )
    return st.session_state["session_demo_repo"]


def _resolve_latest_updated(
    repo: FeedbackRepository, mode: AppMode, db_ready: bool
) -> Optional[datetime]:
    if mode == AppMode.REAL and not db_ready:
        return None
    try:
        flt: FilterCriteria = ensure_shared_filters(mode)
        paged = repo.list_paginated(filters=flt, page=1, page_size=1, sort_by="newest")
        if paged.items:
            return paged.items[0].updated_at
    except Exception:
        return None
    return None


def main() -> None:
    st.set_page_config(
        page_title="TNU Sentiment — Phân tích cảm xúc phản hồi cộng đồng",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_academic_css()

    config = load_app_config()

    if "active_app_mode" not in st.session_state:
        st.session_state["active_app_mode"] = config.default_mode.value

    # --------------------------------------------------------------------------
    # Sidebar: Điều hướng 4 phân hệ & Chuyển đổi chế độ vận hành rõ ràng
    # --------------------------------------------------------------------------
    st.sidebar.markdown("### Điều hướng phân hệ")
    nav_pages = {
        "overview": "Tổng quan",
        "analyze": "Phân tích phản hồi",
        "data_review": "Dữ liệu và kiểm duyệt",
        "evaluation": "Đánh giá mô hình",
    }
    selected_nav = st.sidebar.radio(
        "Chọn chức năng làm việc",
        options=list(nav_pages.keys()),
        format_func=lambda k: nav_pages[k],
        key="sidebar_main_nav",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Chế độ vận hành")
    mode_options = [AppMode.DEMO.value, AppMode.REAL.value]
    mode_labels = {
        AppMode.DEMO.value: "Chế độ demo giao diện",
        AppMode.REAL.value: "Chế độ kết nối dữ liệu thật (MySQL)",
    }
    current_mode_val = st.session_state["active_app_mode"]
    chosen_mode_val = st.sidebar.radio(
        "Nguồn dữ liệu làm việc",
        options=mode_options,
        index=mode_options.index(current_mode_val)
        if current_mode_val in mode_options
        else 0,
        format_func=lambda m: mode_labels[m],
        help=(
            "Chế độ thật kết nối MySQL và không bao giờ tự động chuyển sang dữ liệu giả khi lỗi kết nối. "
            "Chế độ demo dùng bộ dữ liệu mô phỏng cô lập trong phiên."
        ),
    )
    if chosen_mode_val != current_mode_val:
        st.session_state["active_app_mode"] = chosen_mode_val
        st.rerun()

    active_mode = AppMode(st.session_state["active_app_mode"])

    if active_mode == AppMode.DEMO:
        if st.sidebar.button(
            "Khôi phục dữ liệu mẫu Demo ban đầu",
            width="stretch",
            help="Đặt lại toàn bộ thay đổi tạm trong phiên Demo về trạng thái fixture gốc.",
        ):
            st.session_state.pop("session_demo_repo", None)
            st.rerun()

    st.sidebar.markdown("---")
    render_sidebar_project_info()

    # --------------------------------------------------------------------------
    # Khởi tạo Adapters, Repositories & Services theo chế độ hiện hành
    # --------------------------------------------------------------------------
    sentiment_adapter = get_cached_sentiment_adapter(
        str(config.sentiment_checkpoint_dir or ""),
        config.sentiment_model_version,
        config.sentiment_max_length,
    )
    topic_adapter = load_topic_adapter(
        config.topic_checkpoint_dir, config.topic_model_version
    )
    llm_helper = OptionalLLMHelper(config)

    if active_mode == AppMode.DEMO:
        repository: FeedbackRepository = _get_session_demo_repo(config)
    else:
        engine, eng_err = get_cached_mysql_engine(config.database_url or "")
        repository = MySQLFeedbackRepository(
            engine=engine,
            safe_conn_summary=config.safe_db_summary(),
            tz_name=config.timezone_name,
            init_error=eng_err,
        )

    db_health = repository.health()
    health = SystemHealthStatus(
        mode=active_mode.value,
        database=db_health,
        sentiment_model=sentiment_adapter.health(),
        topic_model=topic_adapter.health(),
        llm_helper=llm_helper.health(),
    )

    latest_updated = _resolve_latest_updated(repository, active_mode, db_health.ready)
    render_app_header(
        health=health,
        latest_updated_at=latest_updated,
        env_versions=get_environment_versions(),
    )

    def _switch_to_demo() -> None:
        st.session_state["active_app_mode"] = AppMode.DEMO.value

    # --------------------------------------------------------------------------
    # Định tuyến lười (Lazy Routing): chỉ khởi tạo & chạy phân hệ đang mở
    # --------------------------------------------------------------------------
    if selected_nav == "overview":
        stats_svc = StatsService(config)
        render_overview_page(
            config=config,
            mode=active_mode,
            health=health,
            repository=repository,
            stats_service=stats_svc,
            on_switch_demo=_switch_to_demo,
        )
    elif selected_nav == "analyze":
        strict_prep = VietnamesePreprocessor(require_segmenter=sentiment_adapter.is_ready)
        analysis_svc = AnalysisService(
            config=config,
            sentiment_adapter=sentiment_adapter,
            preprocessor=strict_prep,
            llm_helper=llm_helper,
        )
        render_analyze_page(
            config=config,
            mode=active_mode,
            health=health,
            repository=repository,
            analysis_service=analysis_svc,
            llm_helper=llm_helper,
        )
    elif selected_nav == "data_review":
        prep = get_cached_preprocessor()
        import_svc = ImportService(
            config=config,
            sentiment_adapter=sentiment_adapter,
            preprocessor=VietnamesePreprocessor(require_segmenter=sentiment_adapter.is_ready),
        )
        _ = prep
        review_svc = ReviewService()
        render_data_review_page(
            config=config,
            mode=active_mode,
            health=health,
            repository=repository,
            import_service=import_svc,
            review_service=review_svc,
            on_switch_demo=_switch_to_demo,
        )
    elif selected_nav == "evaluation":
        eval_repo = FileEvaluationRepository(
            artifacts_dir=config.evaluation_artifacts_dir,
            demo_fixture_file=config.fixtures_dir / "demo_benchmark_runs.json",
        )
        bench_svc = BenchmarkService()
        render_evaluation_page(
            mode=active_mode,
            eval_repository=eval_repo,
            benchmark_service=bench_svc,
        )

    render_page_footer()


if __name__ == "__main__":
    main()
