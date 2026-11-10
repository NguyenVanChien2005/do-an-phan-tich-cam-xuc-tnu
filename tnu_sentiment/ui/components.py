"""Các thành phần giao diện dùng chung: Header trạng thái, Banner trung thực dữ liệu, Bộ lọc đồng bộ và Footer."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Callable, Dict, List, Optional

import streamlit as st

from tnu_sentiment.constants import (
    AppMode,
    COLORS,
    DATA_ORIGIN_LABELS,
    DataOrigin,
    LABEL_POLICY_LABELS,
    LabelPolicy,
    PROCESSING_STATUS_LABELS,
    PROJECT_META,
    REVIEW_STATUS_LABELS,
)
from tnu_sentiment.domain.models import (
    FilterCriteria,
    SystemHealthStatus,
)
from tnu_sentiment.repositories.base import FeedbackRepository
from tnu_sentiment.ui.theme import render_pill_html


def render_app_header(
    health: SystemHealthStatus,
    latest_updated_at: Optional[datetime],
    env_versions: Dict[str, str],
) -> None:
    """Hiển thị Header cố định ở đầu trang cùng các badge trạng thái hướng tới người dùng
    và expander 'Thông tin hệ thống' chứa chi tiết kỹ thuật.
    """
    # 1. Badge dữ liệu
    if health.mode == AppMode.DEMO.value:
        db_badge = render_pill_html(
            f"● {health.database.user_badge}", "#EFF6FF", COLORS["brand_primary"], "#BFDBFE"
        )
    elif health.database.ready:
        db_badge = render_pill_html(
            f"● {health.database.user_badge}",
            COLORS["positive_bg"],
            COLORS["positive_fg"],
            "#BFDBFE",
        )
    else:
        db_badge = render_pill_html(
            f"▲ {health.database.user_badge}",
            COLORS["negative_bg"],
            COLORS["negative_fg"],
            "#FECACA",
        )

    # 2. Badge mô hình cảm xúc
    if health.sentiment_model.ready:
        sent_badge = render_pill_html(
            f"● {health.sentiment_model.user_badge}",
            COLORS["positive_bg"],
            COLORS["positive_fg"],
            "#BFDBFE",
        )
    else:
        sent_badge = render_pill_html(
            f"○ {health.sentiment_model.user_badge}",
            COLORS["needs_review_bg"],
            COLORS["needs_review_fg"],
            "#FDE68A",
        )

    # 3. Badge mô hình chủ đề
    topic_badge = render_pill_html(
        f"◇ {health.topic_model.user_badge}",
        COLORS["neutral_bg"],
        COLORS["neutral_fg"],
        COLORS["border"],
    )

    updated_html = ""
    if latest_updated_at is not None:
        safe_ts = html.escape(latest_updated_at.strftime("%d/%m/%Y %H:%M"))
        updated_html = render_pill_html(
            f"Cập nhật gần nhất: {safe_ts}",
            "#F8FAFC",
            COLORS["text_secondary"],
            COLORS["border"],
        )

    safe_short = html.escape(PROJECT_META["short_name"])
    safe_sub = html.escape(PROJECT_META["subtitle"])

    header_html = f"""
    <div class="tnu-header-card">
        <div class="tnu-header-top">
            <div>
                <h1 class="tnu-title">{safe_short}</h1>
                <p class="tnu-subtitle">{safe_sub}</p>
            </div>
            <div class="tnu-badge-row">
                {db_badge}
                {sent_badge}
                {topic_badge}
                {updated_html}
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    with st.expander("Thông tin hệ thống (Chi tiết kỹ thuật & Môi trường)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Trạng thái thành phần độc lập**")
            st.write(f"- **Chế độ vận hành:** `{health.mode}`")
            st.write(f"- **Nguồn dữ liệu:** {health.database.technical_detail}")
            st.write(f"- **Mô hình cảm xúc:** {health.sentiment_model.technical_detail}")
            st.write(f"- **Phân loại chủ đề:** {health.topic_model.technical_detail}")
            st.write(f"- **LLM tùy chọn:** {health.llm_helper.technical_detail}")
        with col_b:
            st.markdown("**Phiên bản môi trường thực tế**")
            for k, v in env_versions.items():
                st.write(f"- **{k}:** `{v}`")


def render_demo_banner() -> None:
    """Banner bắt buộc trên mọi trang khi vận hành ở Chế độ Demo giao diện."""
    banner = """
    <div class="tnu-banner-demo">
        <strong>Dữ liệu mô phỏng — chỉ phục vụ minh họa giao diện.</strong><br/>
        Các bản ghi hiển thị được nạp từ tệp fixture ẩn danh (<code>DEMO</code>).
        Mọi thao tác lưu mới, import hoặc kiểm duyệt chỉ thay đổi bản sao tạm trong phiên hiện tại
        và <strong>sẽ mất khi phiên làm việc kết thúc</strong> (bạn có thể xuất CSV trước khi đóng nếu cần).
    </div>
    """
    st.markdown(banner, unsafe_allow_html=True)


def render_mixed_origin_warning(origin_breakdown: Dict[str, int]) -> None:
    """Hiển thị cơ cấu nguồn gốc dữ liệu và cảnh báo khi chọn 'Tất cả' hoặc trộn nhóm dữ liệu."""
    parts = [
        f"<strong>{html.escape(k)}</strong>: {int(v):,} mẫu"
        for k, v in sorted(origin_breakdown.items())
        if v > 0
    ]
    breakdown_str = " · ".join(parts) if parts else "0 mẫu"
    warn_html = f"""
    <div class="tnu-banner-warn">
        <strong>Cảnh báo dữ liệu hỗn hợp / phi khảo sát thực tế:</strong>
        Bộ lọc hiện không giới hạn riêng ở nhóm <code>TNU_REAL</code>. Tuyệt đối không diễn giải các tỷ lệ
        bên dưới như kết luận thực tế về cộng đồng Đại học Thái Nguyên.<br/>
        <span>Cơ cấu nhóm dữ liệu đang chọn: {breakdown_str}.</span>
    </div>
    """
    st.markdown(warn_html, unsafe_allow_html=True)


def render_db_unavailable_guard(
    health: SystemHealthStatus,
    on_switch_demo: Callable[[], None],
) -> None:
    """Thông báo lỗi kết nối MySQL rõ ràng và KHÔNG tự động chuyển sang Demo nếu người dùng chưa xác nhận."""
    st.error(
        "Không thể truy vấn dữ liệu thực tế do kết nối MySQL chưa sẵn sàng. "
        "Theo nguyên tắc trung thực dữ liệu, hệ thống KHÔNG tự động điền dữ liệu mô phỏng thay thế."
    )
    st.caption(f"Chi tiết kỹ thuật: {health.database.technical_detail}")
    if st.button("Chuyển sang Chế độ demo giao diện", type="primary", key="btn_guard_to_demo"):
        on_switch_demo()
        st.rerun()


def render_kpi_card(label: str, value_str: str, sub_str: str, accent_color: str = "") -> None:
    safe_lbl = html.escape(label)
    safe_val = html.escape(value_str)
    safe_sub = html.escape(sub_str)
    val_style = f' style="color:{html.escape(accent_color)};"' if accent_color else ""
    card_html = f"""
    <div class="tnu-kpi-card">
        <div class="tnu-kpi-label">{safe_lbl}</div>
        <div class="tnu-kpi-value"{val_style}>{safe_val}</div>
        <div class="tnu-kpi-sub">{safe_sub}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def get_default_filters(mode: AppMode) -> FilterCriteria:
    default_origin = (
        DataOrigin.DEMO.value if mode == AppMode.DEMO else DataOrigin.TNU_REAL.value
    )
    return FilterCriteria(
        data_origins=[default_origin],
        select_all_origins=False,
        label_policy=LabelPolicy.VERIFIED_FIRST,
    )


def ensure_shared_filters(mode: AppMode) -> FilterCriteria:
    state_key = f"shared_filters_{mode.value}"
    if state_key not in st.session_state:
        st.session_state[state_key] = get_default_filters(mode)
    return st.session_state[state_key]


def reset_shared_filters(mode: AppMode) -> None:
    state_key = f"shared_filters_{mode.value}"
    st.session_state[state_key] = get_default_filters(mode)
    # Tăng bộ đếm reset để làm mới key của các widget lọc liên kết
    ver_key = f"filter_widget_ver_{mode.value}"
    st.session_state[ver_key] = int(st.session_state.get(ver_key, 0)) + 1


def render_shared_filter_bar(
    repository: FeedbackRepository,
    mode: AppMode,
    include_search: bool = False,
) -> FilterCriteria:
    """Bộ lọc đồng bộ và giữ nguyên trạng thái khi chuyển qua lại giữa Tổng quan và Dữ liệu."""
    current = ensure_shared_filters(mode)
    w_ver = int(st.session_state.get(f"filter_widget_ver_{mode.value}", 0))
    prefix = f"flt_{mode.value}_{w_ver}"

    units = repository.list_units()
    sources = repository.list_sources()
    topics = repository.list_topics()

    unit_options = {u.code: f"{u.code} — {u.name}" for u in units}
    source_options = {s.code: s.name for s in sources}
    topic_options = {t.code: t.name for t in topics}

    origin_choices: List[str] = ["ALL_ORIGINS"] + [o.value for o in DataOrigin]
    origin_labels: Dict[str, str] = {
        "ALL_ORIGINS": "Tất cả nhóm dữ liệu (Hỗn hợp)",
        **DATA_ORIGIN_LABELS,
    }

    current_origin_sel = (
        "ALL_ORIGINS"
        if current.select_all_origins
        else (
            current.data_origins[0]
            if len(current.data_origins) == 1
            else DataOrigin.TNU_REAL.value
        )
    )

    with st.container(border=True):
        top_c1, top_c2, top_c3, top_c4 = st.columns([1.35, 1.35, 1.3, 0.65])
        with top_c1:
            chosen_origin = st.selectbox(
                "Nhóm dữ liệu",
                options=origin_choices,
                index=origin_choices.index(current_origin_sel)
                if current_origin_sel in origin_choices
                else 0,
                format_func=lambda k: origin_labels.get(k, k),
                key=f"{prefix}_origin",
                help="Mặc định chế độ thực tế chỉ tính trên TNU_REAL. Chọn Tất cả sẽ hiển thị cảnh báo dữ liệu hỗn hợp.",
            )
        with top_c2:
            policy_opts = [LabelPolicy.VERIFIED_FIRST, LabelPolicy.VERIFIED_ONLY]
            chosen_policy = st.selectbox(
                "Chính sách nhãn áp dụng",
                options=policy_opts,
                index=policy_opts.index(current.label_policy),
                format_func=lambda p: LABEL_POLICY_LABELS[p.value],
                key=f"{prefix}_policy",
            )
        with top_c3:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                chosen_from = st.date_input(
                    "Từ ngày",
                    value=current.date_from,
                    format="DD/MM/YYYY",
                    key=f"{prefix}_dfrom",
                )
            with d_col2:
                chosen_to = st.date_input(
                    "Đến ngày",
                    value=current.date_to,
                    format="DD/MM/YYYY",
                    key=f"{prefix}_dto",
                )
        with top_c4:
            st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
            if st.button("Đặt lại bộ lọc", width="stretch", key=f"{prefix}_reset"):
                reset_shared_filters(mode)
                st.rerun()

        bot_c1, bot_c2, bot_c3, bot_c4 = st.columns(4)
        with bot_c1:
            chosen_units = st.multiselect(
                "Đơn vị (tham khảo từ Repository)",
                options=list(unit_options.keys()),
                default=[u for u in current.unit_codes if u in unit_options],
                format_func=lambda k: unit_options.get(k, k),
                placeholder="Tất cả đơn vị",
                key=f"{prefix}_units",
            )
        with bot_c2:
            chosen_sources = st.multiselect(
                "Nguồn thu thập",
                options=list(source_options.keys()),
                default=[s for s in current.source_codes if s in source_options],
                format_func=lambda k: source_options.get(k, k),
                placeholder="Tất cả nguồn",
                key=f"{prefix}_sources",
            )
        with bot_c3:
            chosen_topics = st.multiselect(
                "Chủ đề",
                options=list(topic_options.keys()),
                default=[t for t in current.topic_codes if t in topic_options],
                format_func=lambda k: topic_options.get(k, k),
                placeholder="Tất cả chủ đề",
                key=f"{prefix}_topics",
            )
        with bot_c4:
            chosen_reviews = st.multiselect(
                "Trạng thái kiểm duyệt",
                options=list(REVIEW_STATUS_LABELS.keys()),
                default=[r for r in current.review_statuses if r in REVIEW_STATUS_LABELS],
                format_func=lambda k: REVIEW_STATUS_LABELS.get(k, k),
                placeholder="Mọi trạng thái duyệt",
                key=f"{prefix}_reviews",
            )

        with st.expander("Bộ lọc nâng cao (Trạng thái xử lý & Tìm kiếm từ khóa)", expanded=False):
            adv_c1, adv_c2 = st.columns([1, 1.5])
            with adv_c1:
                chosen_procs = st.multiselect(
                    "Trạng thái xử lý kỹ thuật",
                    options=list(PROCESSING_STATUS_LABELS.keys()),
                    default=[
                        p for p in current.processing_statuses if p in PROCESSING_STATUS_LABELS
                    ],
                    format_func=lambda k: PROCESSING_STATUS_LABELS.get(k, k),
                    placeholder="Tất cả (PENDING/DONE/ERROR)",
                    key=f"{prefix}_procs",
                )
            with adv_c2:
                chosen_search = st.text_input(
                    "Tìm trong nội dung phản hồi (đã che PII)",
                    value=current.search_query if include_search or current.search_query else "",
                    placeholder="Nhập từ khóa cần lọc...",
                    key=f"{prefix}_search",
                )

    if chosen_from and chosen_to and chosen_from > chosen_to:
        st.warning("Ngày bắt đầu lớn hơn ngày kết thúc — vui lòng điều chỉnh lại khoảng ngày.")

    updated = FilterCriteria(
        data_origins=(
            [o.value for o in DataOrigin]
            if chosen_origin == "ALL_ORIGINS"
            else [chosen_origin]
        ),
        select_all_origins=(chosen_origin == "ALL_ORIGINS"),
        date_from=chosen_from,
        date_to=chosen_to,
        unit_codes=chosen_units,
        source_codes=chosen_sources,
        topic_codes=chosen_topics,
        processing_statuses=chosen_procs,
        review_statuses=chosen_reviews,
        search_query=chosen_search.strip(),
        label_policy=chosen_policy,
    )
    st.session_state[f"shared_filters_{mode.value}"] = updated
    return updated


def render_sidebar_project_info() -> None:
    with st.sidebar.expander("Thông tin đồ án tốt nghiệp", expanded=False):
        st.markdown(f"**Đề tài:** {PROJECT_META['thesis_title']}")
        st.markdown(f"- **Sinh viên:** {PROJECT_META['student_name']}")
        st.markdown(f"- **MSSV:** `{PROJECT_META['student_id']}`")
        st.markdown(f"- **Lớp:** {PROJECT_META['class_name']}")
        st.markdown(f"- **Đơn vị:** {PROJECT_META['institution']}")
        st.markdown(f"- **GVHD:** {PROJECT_META['advisor_name']}")


def render_page_footer() -> None:
    safe_text = (
        f"Đồ án tốt nghiệp CNTT — {html.escape(PROJECT_META['student_name'])} "
        f"({html.escape(PROJECT_META['student_id'])}, {html.escape(PROJECT_META['class_name'])}) · "
        f"GVHD: {html.escape(PROJECT_META['advisor_name'])} · "
        f"{html.escape(PROJECT_META['institution'])}"
    )
    st.markdown(f'<div class="tnu-footer">{safe_text}</div>', unsafe_allow_html=True)
