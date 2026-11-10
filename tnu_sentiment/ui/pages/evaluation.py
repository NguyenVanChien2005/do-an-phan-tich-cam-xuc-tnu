"""Phân hệ 4: Đánh giá mô hình — Đọc artifact thực nghiệm, tách Validation/Test, Confusion Matrix và Ca lỗi."""

from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tnu_sentiment.constants import AppMode, COLORS
from tnu_sentiment.domain.models import BenchmarkRun
from tnu_sentiment.repositories.base import EvaluationRepository
from tnu_sentiment.services.benchmark_service import BenchmarkService
from tnu_sentiment.ui.components import render_demo_banner
from tnu_sentiment.ui.theme import apply_tnu_plotly_style


def render_evaluation_page(
    mode: AppMode,
    eval_repository: EvaluationRepository,
    benchmark_service: BenchmarkService,
) -> None:
    st.subheader("Đánh giá thực nghiệm mô hình")

    if mode == AppMode.DEMO:
        render_demo_banner()

    runs = eval_repository.load_runs(include_demo_fixtures=(mode == AppMode.DEMO))
    if not runs:
        with st.container(border=True):
            st.warning(
                "Chưa tìm thấy tệp artifact đánh giá thực nghiệm trong thư mục `artifacts/evaluation/`. "
                "Hệ thống không ghi cứng số liệu giả để làm đẹp biểu đồ."
            )
            st.caption(
                "Để sinh artifact đánh giá xác thực từ checkpoint và tập dữ liệu hiện có, chạy lệnh: "
                "`python scripts/evaluate_checkpoint.py`"
            )
        return

    tasks_available = sorted({r.task for r in runs})
    task_labels = {
        "SENTIMENT": "Phân tích cảm xúc (SENTIMENT — 3 lớp)",
        "TOPIC": "Phân loại chủ đề (TOPIC — Các lớp chủ đề)",
    }

    with st.container(border=True):
        sel_c1, sel_c2 = st.columns(2)
        with sel_c1:
            selected_task = st.selectbox(
                "1. Chọn tác vụ đánh giá",
                options=tasks_available,
                format_func=lambda t: task_labels.get(t, t),
                key=f"eval_task_{mode.value}",
            )

        task_runs = [r for r in runs if r.task == selected_task]
        # Tách biệt nghiêm ngặt Validation và Test thành hai nhóm riêng
        splits_available = [
            sp
            for sp in ["Validation", "Test"]
            if any(r.split.lower() == sp.lower() for r in task_runs)
        ] + [
            r.split
            for r in task_runs
            if r.split.lower() not in ("validation", "test")
        ]
        splits_unique = list(dict.fromkeys(splits_available))

        with sel_c2:
            selected_split = st.selectbox(
                "2. Chọn phân tập đánh giá (Validation và Test luôn tách riêng)",
                options=splits_unique,
                format_func=lambda s: f"Tập {s}",
                key=f"eval_split_{mode.value}_{selected_task}",
            )

    split_runs: List[BenchmarkRun] = [
        r for r in task_runs if r.split.lower() == selected_split.lower()
    ]
    if not split_runs:
        st.info("Chưa có lượt chạy đánh giá nào cho phân tập này.")
        return

    if any(r.result_source == "DEMO_FIXTURE" for r in split_runs):
        st.warning(
            "Danh sách hiện có chứa lượt chạy từ **Fixture minh họa Demo (`DEMO_FIXTURE`)** — "
            "chỉ dùng để minh họa bố cục giao diện khi chưa có mô hình tương ứng."
        )

    # Bảng tổng hợp siêu dữ liệu các Run trong nhóm đang chọn
    st.markdown(f"#### Danh sách lượt đánh giá trên tập {selected_split}")
    meta_rows = [
        {
            "Mã Run (run_id)": r.run_id,
            "Mô hình": r.model_name,
            "Checkpoint / Phiên bản": r.checkpoint_version,
            "Phiên bản tập dữ liệu": r.dataset_version,
            "Phân tập": r.split,
            "Định danh tập mẫu (evaluation_set_id)": r.evaluation_set_id,
            "Số mẫu (n_samples)": r.n_samples,
            "Accuracy": round(r.accuracy, 4),
            "Macro-F1": round(r.macro_f1, 4),
            "Weighted-F1": round(r.weighted_f1, 4),
            "Thời điểm đánh giá": r.evaluated_at,
            "Nguồn kết quả": (
                "Artifact thực nghiệm"
                if r.result_source == "ARTIFACT_VERIFIED"
                else "Fixture minh họa Demo"
            ),
        }
        for r in split_runs
    ]
    st.dataframe(pd.DataFrame(meta_rows), width="stretch", hide_index=True)

    # Chọn Run chính để kiểm tra tương thích đối sánh và xem chi tiết
    anchor_run = st.selectbox(
        "Chọn lượt chạy (Run) làm mốc phân tích chi tiết & đối sánh",
        options=split_runs,
        format_func=lambda r: (
            f"{r.model_name} [{r.run_id}] — n={r.n_samples:,} mẫu ({r.result_source})"
        ),
        key=f"eval_anchor_{mode.value}_{selected_task}_{selected_split}",
    )

    comparable_runs, excluded_runs = benchmark_service.filter_comparable_group(
        anchor_run, split_runs
    )

    if excluded_runs:
        for ex_run, reason in excluded_runs:
            st.warning(
                f"Đã loại `{ex_run.run_id}` khỏi biểu đồ đối sánh trực tiếp với `{anchor_run.run_id}`: {reason}"
            )

    if len(comparable_runs) >= 2:
        with st.container(border=True):
            comp_records = []
            for r in comparable_runs:
                comp_records.extend(
                    [
                        {"Mô hình": r.model_name, "Chỉ số": "Accuracy", "Giá trị": round(r.accuracy * 100, 2)},
                        {"Mô hình": r.model_name, "Chỉ số": "Macro Precision", "Giá trị": round(r.macro_precision * 100, 2)},
                        {"Mô hình": r.model_name, "Chỉ số": "Macro Recall", "Giá trị": round(r.macro_recall * 100, 2)},
                        {"Mô hình": r.model_name, "Chỉ số": "Macro-F1", "Giá trị": round(r.macro_f1 * 100, 2)},
                        {"Mô hình": r.model_name, "Chỉ số": "Weighted-F1", "Giá trị": round(r.weighted_f1 * 100, 2)},
                    ]
                )
            comp_df = pd.DataFrame(comp_records)
            fig_comp = px.bar(
                comp_df,
                x="Chỉ số",
                y="Giá trị",
                color="Mô hình",
                barmode="group",
                text="Giá trị",
                color_discrete_sequence=[COLORS["brand_primary"], COLORS["neutral"], COLORS["positive"]],
            )
            fig_comp.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.2f}%<extra></extra>",
            )
            fig_comp.update_yaxes(range=[0, 112])
            apply_tnu_plotly_style(
                fig_comp,
                title=(
                    f"Kết quả đối sánh quan sát trên cùng tập {anchor_run.split} "
                    f"(n = {anchor_run.n_samples:,}, mã tập: {anchor_run.evaluation_set_id})"
                ),
                height=360,
                xaxis_title="Thước đo đánh giá",
                yaxis_title="Điểm số (%)",
            )
            st.plotly_chart(fig_comp, width="stretch")
            st.caption(
                "Ghi chú học thuật: Biểu đồ phản ánh kết quả thực nghiệm quan sát được trên tập đánh giá xác định ở trên; "
                "chỉ các run có cùng tác vụ, cùng evaluation_set_id và cùng định nghĩa metric mới được hiển thị chung."
            )

    # Chi tiết 1 Run: KPI tổng thể + Bảng từng lớp + Ma trận nhầm lẫn
    st.markdown(f"#### Phân tích chi tiết lượt chạy: `{anchor_run.run_id}`")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{anchor_run.accuracy:.2%}")
    m2.metric("Macro Precision", f"{anchor_run.macro_precision:.2%}")
    m3.metric("Macro Recall", f"{anchor_run.macro_recall:.2%}")
    m4.metric("Macro-F1", f"{anchor_run.macro_f1:.2%}")
    m5.metric("Weighted-F1", f"{anchor_run.weighted_f1:.2%}")

    c_left, c_right = st.columns([1.08, 0.92], gap="medium")

    with c_left:
        with st.container(border=True):
            cls_header = (
                "Kết quả chi tiết theo từng lớp cảm xúc"
                if anchor_run.task == "SENTIMENT"
                else "Kết quả chi tiết theo từng lớp chủ đề"
            )
            st.markdown(f"**{cls_header} (`n = {anchor_run.n_samples:,}` mẫu)**")
            cls_df = pd.DataFrame(
                [
                    {
                        "Mã lớp": cm.class_id,
                        "Tên lớp": cm.class_name,
                        "Precision": round(cm.precision, 4),
                        "Recall": round(cm.recall, 4),
                        "F1-Score": round(cm.f1, 4),
                        "Support (Số mẫu)": cm.support,
                    }
                    for cm in anchor_run.per_class_metrics
                ]
            )
            st.dataframe(cls_df, width="stretch", hide_index=True)

            fig_cls = px.bar(
                cls_df.melt(
                    id_vars=["Tên lớp", "Support (Số mẫu)"],
                    value_vars=["Precision", "Recall", "F1-Score"],
                    var_name="Thước đo",
                    value_name="Giá trị",
                ),
                x="Tên lớp",
                y="Giá trị",
                color="Thước đo",
                barmode="group",
                color_discrete_sequence=[COLORS["brand_primary"], COLORS["positive"], "#0EA5E9"],
            )
            fig_cls.update_yaxes(range=[0, 1.08], tickformat=".0%")
            apply_tnu_plotly_style(
                fig_cls,
                title=f"Precision / Recall / F1 từng lớp ({anchor_run.run_id})",
                height=290,
                xaxis_title="Lớp phân loại",
                yaxis_title="Điểm số",
            )
            st.plotly_chart(fig_cls, width="stretch")

    with c_right:
        with st.container(border=True):
            cm_labels = anchor_run.confusion_matrix_labels
            cm_matrix = anchor_run.confusion_matrix
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm_matrix,
                    x=cm_labels,
                    y=cm_labels,
                    text=cm_matrix,
                    texttemplate="%{text}",
                    textfont={"size": 13},
                    colorscale=[
                        [0.0, "#F8FAFC"],
                        [0.35, "#DBEAFE"],
                        [0.7, "#60A5FA"],
                        [1.0, COLORS["brand_primary"]],
                    ],
                    hovertemplate=(
                        "<b>Nhãn thật (Ground Truth):</b> %{y}<br>"
                        "<b>Nhãn dự đoán (Predicted):</b> %{x}<br>"
                        "<b>Số lượng mẫu:</b> %{z}<extra></extra>"
                    ),
                    showscale=True,
                )
            )
            fig_cm.update_yaxes(autorange="reversed")
            apply_tnu_plotly_style(
                fig_cm,
                title=f"Ma trận nhầm lẫn — Run: {anchor_run.run_id} ({anchor_run.split})",
                height=410,
                xaxis_title="Nhãn dự đoán bởi mô hình (Predicted Label)",
                yaxis_title="Nhãn chuẩn thực tế (True Label)",
            )
            st.plotly_chart(fig_cm, width="stretch")

    # Bảng phân tích các ca dự đoán sai trên các mẫu có nhãn chuẩn (đã che PII)
    st.markdown(f"#### Bảng phân tích ca lỗi có nhãn chuẩn (`{anchor_run.run_id}`)")
    with st.container(border=True):
        if not anchor_run.error_cases:
            st.info("Lượt chạy này chưa đính kèm danh sách mẫu ca lỗi chi tiết trong tệp artifact.")
        else:
            categories = ["Tất cả nhóm lỗi"] + sorted(
                {ec.error_category for ec in anchor_run.error_cases}
            )
            chosen_cat = st.selectbox(
                "Lọc theo nhóm nguyên nhân lỗi",
                options=categories,
                key=f"err_cat_{anchor_run.run_id}",
            )
            filtered_errs = [
                ec
                for ec in anchor_run.error_cases
                if chosen_cat == "Tất cả nhóm lỗi" or ec.error_category == chosen_cat
            ]
            err_table = [
                {
                    "Mã mẫu": ec.sample_id,
                    "Nội dung (đã che PII)": ec.masked_text,
                    "Nhãn thật": ec.true_label,
                    "Nhãn dự đoán": ec.predicted_label,
                    "Confidence": (
                        f"{ec.confidence:.2%}" if ec.confidence is not None else "—"
                    ),
                    "Nhóm lỗi": ec.error_category,
                    "Ghi chú phân tích": ec.analysis_note,
                }
                for ec in filtered_errs
            ]
            st.dataframe(pd.DataFrame(err_table), width="stretch", hide_index=True)
            st.caption(
                "Chỉ sử dụng các mẫu thuộc tập đánh giá có nhãn chuẩn (Ground Truth) và đã được kiểm tra che PII."
            )
