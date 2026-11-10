"""Phân hệ 1: Tổng quan — KPI, biểu đồ xu hướng/cơ cấu cảm xúc và bảng phản hồi tiêu cực cần xem xét."""

from __future__ import annotations

import html
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from tnu_sentiment.config import AppConfig
from tnu_sentiment.constants import (
    AppMode,
    COLORS,
    DATA_ORIGIN_LABELS,
    LABEL_POLICY_LABELS,
    REVIEW_STATUS_LABELS,
    TOPIC_SOURCE_LABELS,
)
from tnu_sentiment.domain.models import SystemHealthStatus
from tnu_sentiment.repositories.base import FeedbackRepository
from tnu_sentiment.services.stats_service import StatsService
from tnu_sentiment.ui.components import (
    render_db_unavailable_guard,
    render_demo_banner,
    render_kpi_card,
    render_mixed_origin_warning,
    render_shared_filter_bar,
)
from tnu_sentiment.ui.theme import SENTIMENT_PLOTLY_COLOR_MAP, apply_tnu_plotly_style

_SENTIMENT_ORDER = ["Tiêu cực", "Trung tính", "Tích cực"]


def render_overview_page(
    config: AppConfig,
    mode: AppMode,
    health: SystemHealthStatus,
    repository: FeedbackRepository,
    stats_service: StatsService,
    on_switch_demo: Callable[[], None],
) -> None:
    st.subheader("Tổng quan phản hồi cộng đồng")

    if mode == AppMode.DEMO:
        render_demo_banner()

    if mode == AppMode.REAL and not health.database.ready:
        render_db_unavailable_guard(health, on_switch_demo)
        return

    filters = render_shared_filter_bar(repository=repository, mode=mode)

    records = repository.list_matching_for_stats(filters)
    kpis = stats_service.compute_kpis(records=records, label_policy=filters.label_policy)

    # Banner tóm tắt ngữ cảnh dữ liệu đầu trang (không ghi cứng số lượng mẫu)
    origin_desc = (
        "Tất cả nhóm dữ liệu (Hỗn hợp)"
        if filters.select_all_origins
        else ", ".join(DATA_ORIGIN_LABELS.get(o, o) for o in filters.data_origins)
    )
    if filters.date_from and filters.date_to:
        date_desc = f"{filters.date_from.strftime('%d/%m/%Y')} → {filters.date_to.strftime('%d/%m/%Y')}"
    elif filters.date_from:
        date_desc = f"Từ {filters.date_from.strftime('%d/%m/%Y')}"
    elif filters.date_to:
        date_desc = f"Đến {filters.date_to.strftime('%d/%m/%Y')}"
    else:
        date_desc = "Toàn bộ thời gian hiện có"

    updated_desc = (
        kpis.latest_updated_at.strftime("%d/%m/%Y %H:%M")
        if kpis.latest_updated_at
        else "Chưa có bản ghi"
    )
    policy_desc = LABEL_POLICY_LABELS[filters.label_policy.value]

    strip_html = f"""
    <div class="tnu-context-strip">
        <span>Nhóm dữ liệu: <strong>{html.escape(origin_desc)}</strong></span>
        <span>Khoảng thời gian: <strong>{html.escape(date_desc)}</strong></span>
        <span>Số bản ghi đang chọn: <strong>{kpis.n_total:,}</strong> (hợp lệ để tính tỷ lệ: <strong>{kpis.n_valid:,}</strong>)</span>
        <span>Chính sách nhãn: <strong>{html.escape(policy_desc)}</strong></span>
        <span>Cập nhật gần nhất: <strong>{html.escape(updated_desc)}</strong></span>
    </div>
    """
    st.markdown(strip_html, unsafe_allow_html=True)

    if filters.select_all_origins or kpis.is_mixed_origins:
        render_mixed_origin_warning(kpis.origin_breakdown)

    if kpis.n_total == 0:
        with st.container(border=True):
            st.info(
                "Chưa có bản ghi nào khớp với bộ lọc hiện tại "
                + (
                    "(mặc định chế độ thực tế chỉ truy vấn nhóm dữ liệu khảo sát đã xác nhận TNU_REAL)."
                    if mode == AppMode.REAL
                    else "trong phiên Demo hiện tại."
                )
            )
            if mode == AppMode.REAL:
                if st.button(
                    "Chuyển sang Chế độ demo để xem giao diện minh họa",
                    key="btn_empty_to_demo",
                ):
                    on_switch_demo()
                    st.rerun()
        return

    # 4 Thẻ KPI chính
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            label="1. Tổng phản hồi (N_total)",
            value_str=f"{kpis.n_total:,}",
            sub_str=(
                f"Đã xử lý: {kpis.n_done:,} · Cần duyệt: {kpis.n_needs_review:,} · "
                f"Chờ/Lỗi: {kpis.n_pending_or_running + kpis.n_error:,}"
            ),
        )
    with k2:
        if kpis.pos_ratio_pct is None:
            render_kpi_card(
                label="2. Tỷ lệ tích cực",
                value_str="Chưa có dữ liệu",
                sub_str="N_valid = 0 theo chính sách nhãn đang chọn",
            )
        else:
            render_kpi_card(
                label="2. Tỷ lệ tích cực",
                value_str=f"{kpis.pos_ratio_pct:.1f}%",
                sub_str=f"{kpis.n_pos:,} / {kpis.n_valid:,} mẫu hợp lệ (N_pos / N_valid)",
                accent_color=COLORS["positive"],
            )
    with k3:
        if kpis.neg_ratio_pct is None:
            render_kpi_card(
                label="3. Tỷ lệ tiêu cực",
                value_str="Chưa có dữ liệu",
                sub_str="N_valid = 0 theo chính sách nhãn đang chọn",
            )
        else:
            render_kpi_card(
                label="3. Tỷ lệ tiêu cực",
                value_str=f"{kpis.neg_ratio_pct:.1f}%",
                sub_str=f"{kpis.n_neg:,} / {kpis.n_valid:,} mẫu hợp lệ (N_neg / N_valid)",
                accent_color=COLORS["negative"],
            )
    with k4:
        if kpis.nss_score is None:
            render_kpi_card(
                label="4. Chỉ số cảm xúc ròng (NSS)",
                value_str="Chưa có dữ liệu",
                sub_str="Công thức: 100 × (N_pos − N_neg) / N_valid",
            )
        else:
            nss_color = (
                COLORS["positive"]
                if kpis.nss_score > 0
                else (COLORS["negative"] if kpis.nss_score < 0 else COLORS["neutral"])
            )
            render_kpi_card(
                label="4. Chỉ số cảm xúc ròng (NSS)",
                value_str=f"{kpis.nss_score:+.1f}",
                sub_str=f"Thang −100 đến +100 · Trung tính: {kpis.n_neu:,}/{kpis.n_valid:,}",
                accent_color=nss_color,
            )

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)

    # Dải thông tin phụ: Độ tự tin trung bình & Quy tắc cảnh báo minh họa
    with st.container(border=True):
        info_c1, info_c2 = st.columns([1.15, 1.35])
        with info_c1:
            conf_str = (
                f"{kpis.avg_confidence:.2%}"
                if kpis.avg_confidence is not None
                else "Chưa có dữ liệu"
            )
            st.markdown(f"**Độ tự tin trung bình (thông tin phụ):** `{conf_str}`")
            st.caption(kpis.avg_confidence_model_note)
        with info_c2:
            rule_txt = (
                f"Quy tắc cấu hình minh họa: cảnh báo khi Tỷ lệ tiêu cực >= "
                f"{config.negative_alert_ratio_threshold * 100:.0f}% và N_valid >= {config.negative_alert_min_samples} mẫu "
                "(chỉ phục vụ theo dõi vận hành minh họa, không phải ngưỡng an toàn đã kiểm chứng)."
            )
            if kpis.alert_triggered and kpis.alert_message:
                st.warning(kpis.alert_message)
            else:
                st.markdown("**Trạng thái ngưỡng theo dõi tiêu cực:** Không kích hoạt cảnh báo.")
                st.caption(rule_txt)

    if kpis.n_valid == 0:
        st.info(
            "Hiện chưa có bản ghi nào có nhãn cảm xúc hợp lệ (N_valid = 0) theo chính sách nhãn đang chọn "
            "nên chưa hiển thị biểu đồ tỷ lệ cảm xúc. Các bản ghi PENDING/ERROR không được tính vào mẫu số."
        )
        return

    topic_lookup = {t.code: t.name for t in repository.list_topics()}
    valid_df = stats_service.build_valid_dataframe(
        records=records,
        label_policy=filters.label_policy,
        topic_lookup=topic_lookup,
    )

    # Bố cục biểu đồ 2x2 responsive
    row1_c1, row1_c2 = st.columns(2, gap="medium")

    with row1_c1:
        with st.container(border=True):
            ctrl1, ctrl2 = st.columns(2)
            with ctrl1:
                granularity = st.radio(
                    "Đơn vị thời gian",
                    options=["day", "week"],
                    format_func=lambda x: "Theo ngày" if x == "day" else "Theo tuần",
                    horizontal=True,
                    key=f"ov_gran_{mode.value}",
                )
            with ctrl2:
                metric_mode = st.radio(
                    "Đại lượng hiển thị",
                    options=["count", "ratio"],
                    format_func=lambda x: "Số lượng mẫu" if x == "count" else "Tỷ lệ (%)",
                    horizontal=True,
                    key=f"ov_metric_{mode.value}",
                )

            trend_df = stats_service.aggregate_trend(valid_df, granularity=granularity)
            y_col = "count" if metric_mode == "count" else "ratio_pct"
            y_label = "Số phản hồi hợp lệ" if metric_mode == "count" else "Tỷ lệ (%)"
            fig_trend = px.bar(
                trend_df,
                x="bucket",
                y=y_col,
                color="sentiment_vi",
                category_orders={"sentiment_vi": _SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_PLOTLY_COLOR_MAP,
                custom_data=["count", "bucket_n_valid", "ratio_pct", "sentiment_vi"],
                barmode="stack" if metric_mode == "ratio" else "group",
            )
            fig_trend.update_traces(
                hovertemplate=(
                    "<b>Mốc thời gian:</b> %{x}<br>"
                    "<b>Cảm xúc:</b> %{customdata[3]}<br>"
                    "<b>Số lượng:</b> %{customdata[0]} / %{customdata[1]} mẫu hợp lệ<br>"
                    "<b>Tỷ lệ:</b> %{customdata[2]:.1f}%<extra></extra>"
                )
            )
            apply_tnu_plotly_style(
                fig_trend,
                title=f"Xu hướng cảm xúc ({'theo ngày' if granularity == 'day' else 'theo tuần'}, N_valid = {kpis.n_valid:,})",
                height=340,
                xaxis_title="Thời gian phản hồi",
                yaxis_title=y_label,
            )
            st.plotly_chart(fig_trend, width="stretch")

    with row1_c2:
        with st.container(border=True):
            st.markdown("<div style='height:3.1rem;'></div>", unsafe_allow_html=True)
            src_df = stats_service.aggregate_by_dimension(
                valid_df, "source_code", "source_name"
            )
            fig_src = px.bar(
                src_df,
                x="ratio_pct",
                y="source_name",
                color="sentiment_vi",
                orientation="h",
                category_orders={"sentiment_vi": _SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_PLOTLY_COLOR_MAP,
                custom_data=["count", "group_n_valid", "ratio_pct", "sentiment_vi"],
                barmode="stack",
            )
            fig_src.update_traces(
                hovertemplate=(
                    "<b>Nguồn:</b> %{y}<br>"
                    "<b>Cảm xúc:</b> %{customdata[3]}<br>"
                    "<b>Tỷ lệ:</b> %{customdata[2]:.1f}% (%{customdata[0]}/%{customdata[1]} mẫu hợp lệ)<extra></extra>"
                )
            )
            apply_tnu_plotly_style(
                fig_src,
                title=f"Cơ cấu cảm xúc theo nguồn thu thập (N_valid = {kpis.n_valid:,})",
                height=340,
                xaxis_title="Tỷ lệ trên mẫu hợp lệ từng nguồn (%)",
                yaxis_title="",
            )
            st.plotly_chart(fig_src, width="stretch")

    row2_c1, row2_c2 = st.columns(2, gap="medium")

    with row2_c1:
        with st.container(border=True):
            topic_df = stats_service.aggregate_by_dimension(
                valid_df, "topic_code", "topic_name"
            )
            fig_topic = px.bar(
                topic_df,
                x="count",
                y="topic_name",
                color="sentiment_vi",
                orientation="h",
                category_orders={"sentiment_vi": _SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_PLOTLY_COLOR_MAP,
                custom_data=["count", "group_n_valid", "ratio_pct", "sentiment_vi"],
                barmode="stack",
            )
            fig_topic.update_traces(
                hovertemplate=(
                    "<b>Chủ đề:</b> %{y}<br>"
                    "<b>Cảm xúc:</b> %{customdata[3]}<br>"
                    "<b>Số lượng:</b> %{customdata[0]} / %{customdata[1]} mẫu (%{customdata[2]:.1f}%)<extra></extra>"
                )
            )
            apply_tnu_plotly_style(
                fig_topic,
                title=f"Phân bố cảm xúc theo chủ đề (N_valid = {kpis.n_valid:,})",
                height=350,
                xaxis_title="Số lượng phản hồi hợp lệ",
                yaxis_title="",
            )
            st.plotly_chart(fig_topic, width="stretch")

    with row2_c2:
        with st.container(border=True):
            unit_df = stats_service.aggregate_by_dimension(
                valid_df, "unit_code", "unit_name"
            )
            fig_unit = px.bar(
                unit_df,
                x="count",
                y="unit_code",
                color="sentiment_vi",
                orientation="h",
                category_orders={"sentiment_vi": _SENTIMENT_ORDER},
                color_discrete_map=SENTIMENT_PLOTLY_COLOR_MAP,
                custom_data=["count", "group_n_valid", "ratio_pct", "unit_name", "sentiment_vi"],
                barmode="stack",
            )
            fig_unit.update_traces(
                hovertemplate=(
                    "<b>Đơn vị:</b> %{customdata[3]} (%{y})<br>"
                    "<b>Cảm xúc:</b> %{customdata[4]}<br>"
                    "<b>Số lượng:</b> %{customdata[0]} / %{customdata[1]} mẫu (%{customdata[2]:.1f}%)<extra></extra>"
                )
            )
            apply_tnu_plotly_style(
                fig_unit,
                title=f"Phân bố cảm xúc theo đơn vị tham khảo (N_valid = {kpis.n_valid:,})",
                height=350,
                xaxis_title="Số lượng phản hồi hợp lệ",
                yaxis_title="Mã đơn vị",
            )
            st.plotly_chart(fig_unit, width="stretch")

    # Bảng "Phản hồi tiêu cực cần xem xét" — phân trang 15 dòng trực tiếp từ Repository
    st.markdown("### Phản hồi tiêu cực cần xem xét")
    with st.container(border=True):
        hdr_c1, hdr_c2 = st.columns([1.4, 1.0])
        with hdr_c1:
            st.caption(
                "Chỉ hiển thị nội dung đã che thông tin cá nhân (<EMAIL>, <PHONE>, <STUDENT_ID>). "
                "Lưu ý: Confidence là xác suất đầu ra của mô hình, KHÔNG phản ánh mức độ nghiêm trọng của sự việc."
            )
        with hdr_c2:
            sort_opts = {
                "newest": "Thời gian mới nhất",
                "confidence_asc": "Confidence thấp → cao (ưu tiên rà soát)",
                "confidence_desc": "Confidence cao → thấp",
            }
            neg_sort = st.selectbox(
                "Sắp xếp danh sách",
                options=list(sort_opts.keys()),
                format_func=lambda k: sort_opts[k],
                key=f"ov_neg_sort_{mode.value}",
            )

        page_state_key = f"ov_neg_page_{mode.value}"
        curr_page = int(st.session_state.get(page_state_key, 1))
        neg_page = repository.list_paginated(
            filters=filters,
            page=curr_page,
            page_size=config.page_size_negative_review,  # 15 dòng/trang
            sort_by=neg_sort,
            only_negative_for_review=True,
        )
        st.session_state[page_state_key] = neg_page.page

        if neg_page.total_count == 0:
            st.info("Không có phản hồi tiêu cực nào trong phạm vi bộ lọc hiện tại.")
        else:
            table_rows = [
                {
                    "ID": r.feedback_id,
                    "Thời gian": r.feedback_date.strftime("%d/%m/%Y %H:%M"),
                    "Đơn vị": r.unit_code,
                    "Chủ đề": f"{r.predicted_topic_name} ({TOPIC_SOURCE_LABELS.get(r.topic_assignment_source, r.topic_assignment_source)})",
                    "Nguồn": r.source_name,
                    "Nội dung (đã che PII)": r.masked_content,
                    "Confidence": (
                        f"{r.confidence_score:.2%}" if r.confidence_score is not None else "—"
                    ),
                    "Trạng thái duyệt": REVIEW_STATUS_LABELS.get(
                        r.review_status, r.review_status
                    ),
                }
                for r in neg_page.items
            ]
            st.dataframe(
                pd.DataFrame(table_rows),
                width="stretch",
                hide_index=True,
            )

            pg_c1, pg_c2, pg_c3 = st.columns([1, 2, 1])
            with pg_c1:
                if st.button(
                    "← Trang trước",
                    disabled=(neg_page.page <= 1),
                    key=f"ov_prev_{mode.value}",
                ):
                    st.session_state[page_state_key] = neg_page.page - 1
                    st.rerun()
            with pg_c2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:0.4rem;color:#475569;font-size:0.88rem;'>"
                    f"Trang <strong>{neg_page.page}</strong> / <strong>{neg_page.total_pages}</strong> "
                    f"(Tổng {neg_page.total_count:,} phản hồi tiêu cực · {config.page_size_negative_review} dòng/trang)</div>",
                    unsafe_allow_html=True,
                )
            with pg_c3:
                if st.button(
                    "Trang sau →",
                    disabled=(neg_page.page >= neg_page.total_pages),
                    width="stretch",
                    key=f"ov_next_{mode.value}",
                ):
                    st.session_state[page_state_key] = neg_page.page + 1
                    st.rerun()
