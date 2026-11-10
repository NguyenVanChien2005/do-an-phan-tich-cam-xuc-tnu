"""Thiết lập CSS tối giản an toàn, helper escape HTML và Plotly template đồng nhất bảng màu."""

from __future__ import annotations

import html
from typing import Dict

import plotly.graph_objects as go
import streamlit as st

from tnu_sentiment.constants import COLORS, ReviewStatus, SENTIMENT_DISPLAY_MAP

SENTIMENT_PLOTLY_COLOR_MAP: Dict[str, str] = {
    "Tiêu cực": COLORS["negative"],
    "Trung tính": COLORS["neutral"],
    "Tích cực": COLORS["positive"],
    "Tiêu cực (0)": COLORS["negative"],
    "Trung tính (1)": COLORS["neutral"],
    "Tích cực (2)": COLORS["positive"],
    "Negative": COLORS["negative"],
    "Neutral": COLORS["neutral"],
    "Positive": COLORS["positive"],
}


def inject_academic_css() -> None:
    """Áp dụng lớp CSS gọn nhẹ phục vụ phân cấp thị giác học thuật và responsive (1440/1280/768px).
    Tránh phụ thuộc selector DOM nội bộ dễ vỡ của Streamlit.
    """
    css = f"""
    <style>
        .block-container {{
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }}
        .tnu-header-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.9rem;
        }}
        .tnu-header-top {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
        }}
        .tnu-title {{
            font-size: 1.45rem;
            font-weight: 700;
            color: {COLORS['brand_primary']};
            margin: 0;
            line-height: 1.25;
        }}
        .tnu-subtitle {{
            font-size: 0.92rem;
            color: {COLORS['text_secondary']};
            margin: 0.15rem 0 0 0;
        }}
        .tnu-badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
        }}
        .tnu-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.26rem 0.68rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid transparent;
            white-space: nowrap;
        }}
        .tnu-banner-demo {{
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-left: 4px solid {COLORS['brand_primary']};
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.85rem;
            color: {COLORS['text_primary']};
            font-size: 0.9rem;
        }}
        .tnu-banner-warn {{
            background: {COLORS['needs_review_bg']};
            border: 1px solid #FDE68A;
            border-left: 4px solid {COLORS['needs_review']};
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.85rem;
            color: {COLORS['needs_review_fg']};
            font-size: 0.9rem;
        }}
        .tnu-context-strip {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 0.65rem 1rem;
            margin-bottom: 0.9rem;
            font-size: 0.86rem;
            color: {COLORS['text_secondary']};
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem 1.4rem;
        }}
        .tnu-context-strip strong {{
            color: {COLORS['text_primary']};
        }}
        .tnu-kpi-card {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
            height: 100%;
        }}
        .tnu-kpi-label {{
            font-size: 0.83rem;
            font-weight: 600;
            color: {COLORS['text_secondary']};
            margin-bottom: 0.3rem;
        }}
        .tnu-kpi-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            line-height: 1.2;
        }}
        .tnu-kpi-sub {{
            font-size: 0.79rem;
            color: {COLORS['text_secondary']};
            margin-top: 0.35rem;
        }}
        .tnu-footer {{
            margin-top: 2rem;
            padding-top: 0.85rem;
            border-top: 1px solid {COLORS['border']};
            font-size: 0.82rem;
            color: {COLORS['text_secondary']};
        }}
        @media (max-width: 1280px) {{
            .block-container {{
                padding-left: 1.15rem;
                padding-right: 1.15rem;
            }}
            .tnu-kpi-value {{
                font-size: 1.45rem;
            }}
            .tnu-context-strip {{
                gap: 0.45rem 1rem;
                font-size: 0.83rem;
            }}
        }}
        @media (max-width: 768px) {{
            .block-container {{
                padding-top: 1rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
                overflow-x: clip;
            }}
            .tnu-header-top {{
                flex-direction: column;
                align-items: flex-start;
            }}
            .tnu-title {{
                font-size: 1.25rem;
            }}
            .tnu-kpi-card {{
                padding: 0.8rem 0.9rem;
                margin-bottom: 0.5rem;
            }}
            .tnu-kpi-value {{
                font-size: 1.3rem;
            }}
            .tnu-badge {{
                white-space: normal;
            }}
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_pill_html(label: str, bg_color: str, fg_color: str, border_color: str = "") -> str:
    """Tạo HTML badge tĩnh đã escape hoàn toàn nhãn đầu vào."""
    safe_label = html.escape(str(label), quote=True)
    safe_bg = html.escape(str(bg_color), quote=True)
    safe_fg = html.escape(str(fg_color), quote=True)
    b_col = html.escape(str(border_color or bg_color), quote=True)
    return (
        f'<span class="tnu-badge" style="background:{safe_bg};color:{safe_fg};border-color:{b_col};">'
        f"{safe_label}</span>"
    )


def sentiment_badge_html(label_id: int | None) -> str:
    if label_id not in SENTIMENT_DISPLAY_MAP:
        return render_pill_html("Chưa có nhãn", COLORS["neutral_bg"], COLORS["neutral_fg"])
    info = SENTIMENT_DISPLAY_MAP[label_id]
    return render_pill_html(info["full_label"], info["bg"], info["fg"])


def review_badge_html(review_status: str) -> str:
    if review_status == ReviewStatus.NEEDS_REVIEW.value:
        return render_pill_html(
            "Cần kiểm duyệt", COLORS["needs_review_bg"], COLORS["needs_review_fg"]
        )
    if review_status == ReviewStatus.VERIFIED.value:
        return render_pill_html(
            "Đã xác nhận", COLORS["positive_bg"], COLORS["positive_fg"]
        )
    return render_pill_html("Chưa kiểm duyệt", COLORS["neutral_bg"], COLORS["neutral_fg"])


def apply_tnu_plotly_style(
    fig: go.Figure,
    title: str = "",
    height: int = 340,
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    """Áp dụng template trực quan hóa học thuật thống nhất cho tất cả biểu đồ Plotly."""
    fig.update_layout(
        title=dict(
            text=html.escape(title) if title else "",
            font=dict(size=15, color=COLORS["text_primary"], family="Segoe UI, Arial, sans-serif"),
            x=0.0,
            xanchor="left",
        ),
        height=height,
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(
            family="Segoe UI, Roboto, Arial, sans-serif",
            size=12,
            color=COLORS["text_primary"],
        ),
        margin=dict(l=16, r=16, t=48 if title else 24, b=28),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            title_text="",
            font=dict(size=11, color=COLORS["text_secondary"]),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=COLORS["border"],
            font=dict(size=12, color=COLORS["text_primary"]),
        ),
    )
    fig.update_xaxes(
        title_text=xaxis_title,
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_secondary"]),
        title_font=dict(size=12, color=COLORS["text_secondary"]),
    )
    fig.update_yaxes(
        title_text=yaxis_title,
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor=COLORS["border"],
        tickfont=dict(color=COLORS["text_secondary"]),
        title_font=dict(size=12, color=COLORS["text_secondary"]),
    )
    return fig
