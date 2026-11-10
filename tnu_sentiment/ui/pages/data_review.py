"""Phân hệ 3: Dữ liệu và kiểm duyệt — Kiểm duyệt theo ID, Import CSV/XLSX 8 bước và Lịch sử thay đổi."""

from __future__ import annotations

import io
from typing import Callable, Dict, List

import openpyxl
import pandas as pd
import streamlit as st

from tnu_sentiment.config import AppConfig
from tnu_sentiment.constants import (
    AppMode,
    DATA_ORIGIN_LABELS,
    DataOrigin,
    ReviewStatus,
    SENTIMENT_ID_TO_VI,
)
from tnu_sentiment.domain.models import ImportExecutionSummary, SystemHealthStatus
from tnu_sentiment.repositories.base import FeedbackRepository
from tnu_sentiment.services.import_service import ImportService, sanitize_csv_cell
from tnu_sentiment.services.review_service import ReviewService
from tnu_sentiment.ui.components import (
    render_db_unavailable_guard,
    render_demo_banner,
    render_mixed_origin_warning,
    render_shared_filter_bar,
)

_SENTIMENT_EDIT_CHOICES = {
    "NONE": "Chưa chọn",
    "0": "0 — Tiêu cực",
    "1": "1 — Trung tính",
    "2": "2 — Tích cực",
}
_CHOICE_TO_INT = {
    "Chưa chọn": None,
    "0 — Tiêu cực": 0,
    "1 — Trung tính": 1,
    "2 — Tích cực": 2,
}


def _build_template_xlsx_bytes() -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ImportTemplate"
    ws.append(
        ["sentence", "unit_code", "source_name", "feedback_date", "source_record_id", "topic_code"]
    )
    ws.append(
        [
            "Giảng viên hướng dẫn thực hành rất tận tình, dễ hiểu.",
            "ICTU",
            "SURVEY_PORTAL",
            "2026-10-15 08:30:00",
            "XLSX-REC-001",
            "LECTURER",
        ]
    )
    ws.append(
        [
            "Phòng máy tầng 3 có hai máy tính bị hỏng chuột cần bảo trì.",
            "TNUT",
            "QA_BOX",
            "2026-10-16 14:15:00",
            "XLSX-REC-002",
            "FACILITY",
        ]
    )
    buf = io.BytesIO()
    wb.save(buf)
    wb.close()
    return buf.getvalue()


def render_data_review_page(
    config: AppConfig,
    mode: AppMode,
    health: SystemHealthStatus,
    repository: FeedbackRepository,
    import_service: ImportService,
    review_service: ReviewService,
    on_switch_demo: Callable[[], None],
) -> None:
    st.subheader("Quản lý dữ liệu & Kiểm duyệt phản hồi")

    if mode == AppMode.DEMO:
        render_demo_banner()

    if mode == AppMode.REAL and not health.database.ready:
        render_db_unavailable_guard(health, on_switch_demo)
        return

    filters = render_shared_filter_bar(
        repository=repository, mode=mode, include_search=True
    )

    # Hiển thị cảnh báo khi chọn hỗn hợp nguồn gốc dữ liệu
    if filters.select_all_origins or (
        len(filters.data_origins) == 1
        and filters.data_origins[0] not in (DataOrigin.TNU_REAL.value, DataOrigin.DEMO.value)
    ):
        stats_records = repository.list_matching_for_stats(filters)
        origin_counts: Dict[str, int] = {}
        for r in stats_records:
            origin_counts[r.data_origin] = origin_counts.get(r.data_origin, 0) + 1
        render_mixed_origin_warning(origin_counts)

    tab_review, tab_import, tab_history = st.tabs(
        [
            "1. Dữ liệu & Kiểm duyệt",
            "2. Import CSV / XLSX",
            "3. Lịch sử kiểm duyệt",
        ]
    )

    # ==========================================================================
    # TAB 1: TRA CỨU PHÂN TRANG, CHỈNH SỬA KIỂM DUYỆT THEO ID & XUẤT CSV
    # ==========================================================================
    with tab_review:
        with st.container(border=True):
            st.info(
                "**Lưu ý định danh người duyệt:** Hệ thống hiện vận hành như bản triển khai một người dùng "
                "(chưa tích hợp SSO/xác thực tài khoản). Trường `Người duyệt` bên dưới do người thao tác tự nhập "
                "để lưu vết kiểm duyệt minh họa, **không phải danh tính đã được xác thực**."
            )

            top_r1, top_r2, top_r3 = st.columns([1.2, 1.1, 0.9])
            with top_r1:
                reviewer_name = st.text_input(
                    "Người duyệt (tự khai báo)",
                    value=st.session_state.get("reviewer_name_input", ""),
                    placeholder="Ví dụ: nguyenvanchien_reviewer",
                    key=f"rev_user_{mode.value}",
                )
                st.session_state["reviewer_name_input"] = reviewer_name
            with top_r2:
                sort_options = {
                    "newest": "Mới nhất trước",
                    "oldest": "Cũ nhất trước",
                    "confidence_asc": "Confidence thấp → cao",
                    "confidence_desc": "Confidence cao → thấp",
                }
                data_sort = st.selectbox(
                    "Sắp xếp bảng dữ liệu",
                    options=list(sort_options.keys()),
                    format_func=lambda k: sort_options[k],
                    key=f"dr_sort_{mode.value}",
                )
            with top_r3:
                export_rows = repository.export_records(filters=filters, limit=5000)
                if export_rows:
                    safe_export_rows = [
                        {k: sanitize_csv_cell(v) for k, v in r.items()}
                        for r in export_rows
                    ]
                    export_bytes = (
                        pd.DataFrame(safe_export_rows)
                        .to_csv(index=False)
                        .encode("utf-8-sig")
                    )
                else:
                    export_bytes = b""
                st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
                st.download_button(
                    label=f"Xuất CSV đã che PII ({len(export_rows):,} dòng)",
                    data=export_bytes,
                    file_name=f"tnu_feedbacks_{mode.value.lower()}_export.csv",
                    mime="text/csv",
                    disabled=(len(export_rows) == 0),
                    width="stretch",
                    key=f"btn_export_csv_{mode.value}",
                )
                if len(export_rows) == 0:
                    st.caption("Không có bản ghi để xuất.")

            page_key = f"dr_page_{mode.value}"
            curr_page = int(st.session_state.get(page_key, 1))
            paged = repository.list_paginated(
                filters=filters,
                page=curr_page,
                page_size=config.page_size_data_table,  # 15 dòng/trang
                sort_by=data_sort,
                only_negative_for_review=False,
            )
            st.session_state[page_key] = paged.page

            if paged.total_count == 0:
                st.info("Chưa có dữ liệu khớp bộ lọc hiện tại.")
            else:
                topic_codes_valid = [t.code for t in repository.list_topics()]
                editor_data: List[Dict[str, object]] = []
                for r in paged.items:
                    rev_sent_disp = (
                        _SENTIMENT_EDIT_CHOICES[str(r.reviewed_sentiment_label)]
                        if r.reviewed_sentiment_label in (0, 1, 2)
                        else "Chưa chọn"
                    )
                    editor_data.append(
                        {
                            "feedback_id": r.feedback_id,
                            "record_version": r.record_version,
                            "feedback_date": r.feedback_date.strftime("%d/%m/%Y %H:%M"),
                            "data_origin": r.data_origin,
                            "unit_code": r.unit_code,
                            "masked_content": r.masked_content,
                            "processing_status": r.processing_status,
                            "predicted_sentiment": (
                                f"{r.predicted_sentiment_vi} ({r.predicted_sentiment_label})"
                                if r.predicted_sentiment_label is not None
                                else "Chưa có"
                            ),
                            "confidence": (
                                round(float(r.confidence_score), 4)
                                if r.confidence_score is not None
                                else None
                            ),
                            "predicted_topic": r.predicted_topic_code,
                            "review_status": r.review_status,
                            "reviewed_sentiment_choice": rev_sent_disp,
                            "reviewed_topic_code": r.reviewed_topic_code or "Chưa chọn",
                            "review_note": r.review_note or "",
                        }
                    )

                st.caption(
                    "Các cột ID, Nội dung đã che PII và Dự đoán gốc của mô hình được khóa chỉ đọc. "
                    "Việc lưu kiểm duyệt đối chiếu trực tiếp theo `feedback_id` và `record_version`, "
                    "tuyệt đối không ghi đè dự đoán gốc."
                )

                edited_df = st.data_editor(
                    pd.DataFrame(editor_data),
                    hide_index=True,
                    width="stretch",
                    disabled=[
                        "feedback_id",
                        "record_version",
                        "feedback_date",
                        "data_origin",
                        "unit_code",
                        "masked_content",
                        "processing_status",
                        "predicted_sentiment",
                        "confidence",
                        "predicted_topic",
                    ],
                    column_config={
                        "feedback_id": st.column_config.NumberColumn("ID", width="small"),
                        "record_version": st.column_config.NumberColumn("Ver", width="small"),
                        "feedback_date": st.column_config.TextColumn("Thời gian"),
                        "data_origin": st.column_config.TextColumn("Nguồn gốc"),
                        "unit_code": st.column_config.TextColumn("Đơn vị", width="small"),
                        "masked_content": st.column_config.TextColumn(
                            "Nội dung (đã che PII)", width="large"
                        ),
                        "processing_status": st.column_config.TextColumn("Xử lý", width="small"),
                        "predicted_sentiment": st.column_config.TextColumn("Dự đoán gốc (Khóa)"),
                        "confidence": st.column_config.NumberColumn(
                            "Confidence", format="%.4f", width="small"
                        ),
                        "predicted_topic": st.column_config.TextColumn("Chủ đề gốc (Khóa)"),
                        "review_status": st.column_config.SelectboxColumn(
                            "Trạng thái duyệt ✎",
                            options=[s.value for s in ReviewStatus],
                            required=True,
                        ),
                        "reviewed_sentiment_choice": st.column_config.SelectboxColumn(
                            "Cảm xúc duyệt ✎",
                            options=list(_CHOICE_TO_INT.keys()),
                            required=True,
                        ),
                        "reviewed_topic_code": st.column_config.SelectboxColumn(
                            "Chủ đề duyệt ✎",
                            options=["Chưa chọn"] + topic_codes_valid,
                            required=True,
                        ),
                        "review_note": st.column_config.TextColumn("Ghi chú duyệt ✎"),
                    },
                    key=f"editor_review_{mode.value}_p{paged.page}_{data_sort}",
                )

                normalized_edited: List[Dict[str, object]] = []
                for _, row in edited_df.iterrows():
                    sent_choice = str(row["reviewed_sentiment_choice"])
                    topic_choice = str(row["reviewed_topic_code"])
                    normalized_edited.append(
                        {
                            "feedback_id": int(row["feedback_id"]),
                            "review_status": str(row["review_status"]),
                            "reviewed_sentiment_label": _CHOICE_TO_INT.get(sent_choice),
                            "reviewed_topic_code": (
                                None if topic_choice == "Chưa chọn" else topic_choice
                            ),
                            "review_note": str(row["review_note"] or "").strip(),
                        }
                    )

                diff_reqs, diff_errs = review_service.build_diff_requests(
                    original_records=paged.items,
                    edited_rows=normalized_edited,
                    reviewer_name=reviewer_name,
                )

                act_c1, act_c2 = st.columns([1.0, 2.0])
                with act_c1:
                    save_rev_disabled = len(diff_reqs) == 0 or len(diff_errs) > 0
                    if st.button(
                        f"Lưu kiểm duyệt ({len(diff_reqs)} thay đổi)",
                        type="primary",
                        disabled=save_rev_disabled,
                        width="stretch",
                        key=f"btn_save_review_{mode.value}",
                    ):
                        applied, apply_errs = review_service.save_reviews(
                            repository=repository, updates=diff_reqs
                        )
                        if applied > 0:
                            st.success(f"Đã lưu thành công {applied} bản ghi kiểm duyệt.")
                        for e in apply_errs:
                            st.error(e)
                        if applied > 0 and not apply_errs:
                            st.rerun()
                with act_c2:
                    if diff_errs:
                        for err in diff_errs:
                            st.warning(err)
                    elif len(diff_reqs) == 0:
                        st.caption(
                            "Chưa có dòng nào thay đổi ở các cột kiểm duyệt (✎) nên nút lưu đang tạm khóa."
                        )

                # Phân trang bảng kiểm duyệt
                p1, p2, p3 = st.columns([1, 2, 1])
                with p1:
                    if st.button(
                        "← Trang trước",
                        disabled=(paged.page <= 1),
                        key=f"dr_prev_{mode.value}",
                    ):
                        st.session_state[page_key] = paged.page - 1
                        st.rerun()
                with p2:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:0.4rem;color:#475569;font-size:0.88rem;'>"
                        f"Trang <strong>{paged.page}</strong> / <strong>{paged.total_pages}</strong> "
                        f"(Tổng {paged.total_count:,} bản ghi · {config.page_size_data_table} dòng/trang)</div>",
                        unsafe_allow_html=True,
                    )
                with p3:
                    if st.button(
                        "Trang sau →",
                        disabled=(paged.page >= paged.total_pages),
                        width="stretch",
                        key=f"dr_next_{mode.value}",
                    ):
                        st.session_state[page_key] = paged.page + 1
                        st.rerun()

    # ==========================================================================
    # TAB 2: IMPORT CSV / XLSX 8 BƯỚC AN TOÀN
    # ==========================================================================
    with tab_import:
        with st.container(border=True):
            st.markdown("#### Tải tệp mẫu chuẩn")
            st.caption(
                f"Cột bắt buộc: `sentence`, `unit_code`, `source_name`, `feedback_date` "
                f"(tùy chọn: `source_record_id` để chống import lặp, `topic_code`). "
                f"Giới hạn cấu hình: tối đa {config.import_max_file_mb} MB, {config.import_max_rows:,} dòng. "
                "Tệp `.xlsx` chỉ đọc dữ liệu thuần, tự động chặn thực thi công thức và macro."
            )
            dl_c1, dl_c2, _ = st.columns([1, 1, 1.5])
            with dl_c1:
                st.download_button(
                    "Tải tệp mẫu CSV",
                    data=import_service.build_template_csv_bytes(),
                    file_name="tnu_import_template.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with dl_c2:
                st.download_button(
                    "Tải tệp mẫu XLSX",
                    data=_build_template_xlsx_bytes(),
                    file_name="tnu_import_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )

        with st.container(border=True):
            st.markdown("#### Bước 1 — Khai báo nguồn gốc đợt Import")
            if mode == AppMode.DEMO:
                declared_origin = DataOrigin.DEMO.value
                origin_confirmed = True
                batch_note = "Import thử nghiệm trong phiên Demo"
                st.info(
                    "Trong **Chế độ demo**, mọi đợt import được gắn nhãn `DEMO` và chỉ lưu tạm trong phiên."
                )
            else:
                orig_col1, orig_col2 = st.columns([1.2, 1.8])
                with orig_col1:
                    declared_origin = st.selectbox(
                        "Phân loại nguồn gốc của tệp tải lên",
                        options=[o.value for o in DataOrigin],
                        index=3,  # Mặc định DEMO cho tới khi người dùng chủ động chọn loại khác
                        format_func=lambda k: DATA_ORIGIN_LABELS[k],
                        key="imp_declared_origin",
                    )
                with orig_col2:
                    batch_note = st.text_input(
                        "Mô tả nguồn / văn bản xác nhận đợt thu thập",
                        placeholder="Ví dụ: Kết quả khảo sát học phần HK1 năm học 2026-2027",
                        key="imp_origin_note",
                    )
                origin_confirmed = st.checkbox(
                    f"Tôi xác nhận toàn bộ bản ghi trong tệp thuộc đúng nhóm nguồn gốc `{declared_origin}` đã chọn",
                    value=False,
                    key="imp_origin_confirm_chk",
                    help="Hệ thống không mặc định coi mọi tệp tải lên là dữ liệu khảo sát thật ĐHTN (TNU_REAL).",
                )

            st.markdown("#### Bước 2 & 3 — Tải tệp và Kiểm tra cấu trúc / dữ liệu")
            uploaded = st.file_uploader(
                "Chọn tệp dữ liệu (.csv hoặc .xlsx)",
                type=["csv", "xlsx"],
                key=f"imp_uploader_{mode.value}",
            )

            if uploaded is not None:
                try:
                    raw_bytes = uploaded.getvalue()
                    headers, rows = import_service.read_tabular_bytes(
                        uploaded.name, raw_bytes
                    )
                    schema_errs, previews = import_service.validate_and_preview(
                        headers=headers, rows=rows, repository=repository
                    )
                except Exception as read_exc:
                    st.error(f"Lỗi đọc tệp đầu vào: {read_exc}")
                    schema_errs = [str(read_exc)]
                    previews = []

                if schema_errs:
                    for se in schema_errs:
                        st.error(se)
                elif previews:
                    valid_previews = [p for p in previews if p.is_valid]
                    invalid_previews = [p for p in previews if not p.is_valid]

                    st.markdown(
                        f"#### Bước 4 — Xem trước dữ liệu đã che PII "
                        f"(Tổng đọc: **{len(previews)}** dòng · Hợp lệ: **{len(valid_previews)}** · Lỗi đầu vào: **{len(invalid_previews)}**)"
                    )
                    preview_df = pd.DataFrame(
                        [
                            {
                                "Dòng": p.row_number,
                                "Trạng thái dòng": "Hợp lệ" if p.is_valid else "Lỗi đầu vào",
                                "Nội dung (đã che PII)": p.masked_sentence,
                                "Đơn vị": p.unit_code,
                                "Nguồn": p.source_code,
                                "Ngày phản hồi": (
                                    p.feedback_date.strftime("%Y-%m-%d %H:%M")
                                    if p.feedback_date
                                    else "Sai định dạng"
                                ),
                                "Mã bản ghi nguồn": p.source_record_id or "—",
                                "Chủ đề": p.topic_code,
                            }
                            for p in previews[:50]
                        ]
                    )
                    st.dataframe(preview_df, width="stretch", hide_index=True)
                    if len(previews) > 50:
                        st.caption(f"Đang hiển thị 50/{len(previews)} dòng đầu tiên để xem trước.")

                    if invalid_previews:
                        st.markdown("#### Bước 5 — Chi tiết lỗi đầu vào theo dòng")
                        err_df = pd.DataFrame(
                            [
                                {
                                    "Dòng số": p.row_number,
                                    "Nội dung (đã che PII)": p.masked_sentence,
                                    "Chi tiết lỗi": "; ".join(p.errors),
                                }
                                for p in invalid_previews
                            ]
                        )
                        st.dataframe(err_df, width="stretch", hide_index=True)

                    st.markdown("#### Bước 6 & 7 — Xác nhận Import và Tiến độ thực tế")
                    can_run_import = (
                        len(previews) > 0
                        and origin_confirmed
                        and health.sentiment_model.ready
                        and health.database.ready
                    )
                    imp_reason = ""
                    if not origin_confirmed:
                        imp_reason = "Vui lòng tích ô xác nhận nguồn gốc dữ liệu đợt Import ở Bước 1."
                    elif not health.sentiment_model.ready:
                        imp_reason = "Mô hình cảm xúc chưa sẵn sàng nên chưa thể suy luận tự động cho lô Import."
                    elif not health.database.ready:
                        imp_reason = "Kết nối CSDL chưa sẵn sàng."

                    if st.button(
                        f"Xác nhận thực thi Import ({len(previews)} dòng)",
                        type="primary",
                        disabled=not can_run_import,
                        key=f"btn_exec_import_{mode.value}",
                    ):
                        prog_bar = st.progress(0.0, text="Đang khởi tạo tiến trình Import...")

                        def _on_progress(curr: int, tot: int) -> None:
                            frac = curr / max(1, tot)
                            prog_bar.progress(
                                frac,
                                text=f"Đang xử lý dòng {curr}/{tot} ({frac * 100:.0f}%)...",
                            )

                        summary = import_service.execute_import(
                            filename=uploaded.name,
                            declared_origin=declared_origin,
                            origin_note=batch_note or f"Import từ {uploaded.name}",
                            previews=previews,
                            repository=repository,
                            progress_callback=_on_progress,
                        )
                        prog_bar.progress(1.0, text="Đã hoàn tất xử lý lô Import.")
                        st.session_state[f"last_import_summary_{mode.value}"] = summary

                    if imp_reason:
                        st.caption(f"Lý do tạm khóa nút xác nhận Import: {imp_reason}")

            # Bước 8: Hiển thị tổng kết đợt Import gần nhất
            last_summary: ImportExecutionSummary | None = st.session_state.get(
                f"last_import_summary_{mode.value}"
            )
            if last_summary is not None:
                st.markdown("---")
                st.markdown(
                    f"#### Bước 8 — Tổng kết đợt Import (`{last_summary.batch_id}` · Nguồn gốc: `{last_summary.declared_origin}`)"
                )
                s1, s2, s3, s4, s5, s6 = st.columns(6)
                s1.metric("Đã lưu thành công", last_summary.saved_count)
                s2.metric("Bỏ qua do trùng", last_summary.duplicate_skipped_count)
                s3.metric("Lỗi đầu vào", last_summary.validation_error_count)
                s4.metric("Lỗi suy luận", last_summary.inference_error_count)
                s5.metric("Lỗi ghi dữ liệu", last_summary.db_error_count)
                s6.metric("Chưa xử lý (gián đoạn)", last_summary.unprocessed_count)

                st.caption(
                    f"Kiểm chứng đối soát không chồng lặp: "
                    f"{last_summary.saved_count} + {last_summary.duplicate_skipped_count} + "
                    f"{last_summary.validation_error_count} + {last_summary.inference_error_count} + "
                    f"{last_summary.db_error_count} + {last_summary.unprocessed_count} = "
                    f"**{last_summary.accounted_rows}** / **{last_summary.total_rows_read}** tổng số dòng đã đọc."
                )

                if last_summary.row_errors:
                    err_csv_bytes = import_service.export_error_report_csv(last_summary)
                    st.download_button(
                        label=f"Tải danh sách dòng lỗi / bỏ qua ({len(last_summary.row_errors)} dòng - CSV)",
                        data=err_csv_bytes,
                        file_name=f"import_errors_{last_summary.batch_id}.csv",
                        mime="text/csv",
                        key=f"dl_imp_err_{last_summary.batch_id}",
                    )

    # ==========================================================================
    # TAB 3: LỊCH SỬ THAY ĐỔI KIỂM DUYỆT (AUDIT LOG)
    # ==========================================================================
    with tab_history:
        with st.container(border=True):
            st.markdown("#### Lịch sử kiểm duyệt bản ghi")
            st.caption(
                "Ghi nhận mọi thao tác duyệt theo ID bản ghi và phiên bản (record_version). "
                "Tên người duyệt được ghi nhận là thông tin khai báo thủ công (chưa xác thực tài khoản)."
            )
            flt_fid_raw = st.text_input(
                "Lọc theo ID phản hồi (để trống để xem 50 thay đổi gần nhất)",
                placeholder="Ví dụ: 27",
                key=f"hist_fid_{mode.value}",
            ).strip()
            flt_fid = int(flt_fid_raw) if flt_fid_raw.isdigit() else None

            logs = review_service.get_recent_logs(
                repository=repository, feedback_id=flt_fid, limit=50
            )
            if not logs:
                st.info("Chưa có lịch sử thay đổi kiểm duyệt nào được ghi nhận.")
            else:
                log_rows = [
                    {
                        "Mã log": lg.log_id,
                        "ID phản hồi": lg.feedback_id,
                        "Thời điểm": lg.changed_at.strftime("%d/%m/%Y %H:%M:%S"),
                        "Trạng thái": f"{lg.previous_review_status or '—'} → {lg.new_review_status}",
                        "Nhãn cảm xúc duyệt": (
                            f"{SENTIMENT_ID_TO_VI.get(lg.previous_sentiment_label, 'Chưa có')} → "
                            f"{SENTIMENT_ID_TO_VI.get(lg.new_sentiment_label, 'Chưa có')}"
                        ),
                        "Chủ đề duyệt": f"{lg.previous_topic_code or '—'} → {lg.new_topic_code or '—'}",
                        "Người duyệt (chưa xác thực)": lg.reviewed_by,
                        "Phiên bản": f"v{lg.record_version_before} → v{lg.record_version_after}",
                        "Ghi chú": lg.review_note or "",
                    }
                    for lg in logs
                ]
                st.dataframe(
                    pd.DataFrame(log_rows), width="stretch", hide_index=True
                )
