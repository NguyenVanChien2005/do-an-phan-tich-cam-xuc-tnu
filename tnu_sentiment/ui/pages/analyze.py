"""Phân hệ 2: Phân tích phản hồi — Nhập liệu, 6 ca kiểm thử mẫu, suy luận thực/minh họa và giải thích xử lý."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from tnu_sentiment.config import AppConfig
from tnu_sentiment.constants import (
    AppMode,
    COLORS,
    SENTIMENT_DISPLAY_MAP,
    TOPIC_SOURCE_LABELS,
)
from tnu_sentiment.domain.models import AnalysisOutcome, SystemHealthStatus
from tnu_sentiment.models.llm_helper import OptionalLLMHelper
from tnu_sentiment.repositories.base import FeedbackRepository
from tnu_sentiment.services.analysis_service import AnalysisService
from tnu_sentiment.ui.components import render_demo_banner
from tnu_sentiment.ui.theme import (
    SENTIMENT_PLOTLY_COLOR_MAP,
    apply_tnu_plotly_style,
    render_pill_html,
    review_badge_html,
    sentiment_badge_html,
)


def render_analyze_page(
    config: AppConfig,
    mode: AppMode,
    health: SystemHealthStatus,
    repository: FeedbackRepository,
    analysis_service: AnalysisService,
    llm_helper: OptionalLLMHelper,
) -> None:
    st.subheader("Phân tích cảm xúc phản hồi")

    if mode == AppMode.DEMO:
        render_demo_banner()

    samples = analysis_service.list_sample_cases()
    sample_map = {s["sample_id"]: s for s in samples}
    sample_options = ["CUSTOM_INPUT"] + [s["sample_id"] for s in samples]

    units = repository.list_units()
    sources = repository.list_sources()
    topics = repository.list_topics()

    unit_dict = {u.code: f"{u.code} — {u.name}" for u in units}
    source_dict = {s.code: s.name for s in sources}
    topic_dict = {t.code: t.name for t in topics}

    outcome_key = f"current_analysis_outcome_{mode.value}"
    saved_key = f"current_analysis_saved_{mode.value}"
    text_key = f"analyze_input_text_{mode.value}"
    unit_key = f"analyze_unit_{mode.value}"
    source_key = f"analyze_source_{mode.value}"
    topic_key = f"analyze_topic_{mode.value}"

    if text_key not in st.session_state:
        st.session_state[text_key] = ""

    def _on_sample_change() -> None:
        chosen_id = st.session_state.get(f"sample_selector_{mode.value}", "CUSTOM_INPUT")
        if chosen_id in sample_map:
            sc = sample_map[chosen_id]
            st.session_state[text_key] = sc["text"]
            if sc.get("suggested_unit") in unit_dict:
                st.session_state[unit_key] = sc["suggested_unit"]
            if sc.get("suggested_source") in source_dict:
                st.session_state[source_key] = sc["suggested_source"]
            if sc.get("suggested_topic") in topic_dict:
                st.session_state[topic_key] = sc["suggested_topic"]

    # Desktop chia hai vùng tỷ lệ khoảng 45% và 55%
    col_left, col_right = st.columns([0.45, 0.55], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("#### 1. Nhập nội dung & Siêu dữ liệu")

            chosen_sample_id = st.selectbox(
                "Chọn câu mẫu kiểm thử (6 nhóm tình huống)",
                options=sample_options,
                format_func=lambda sid: (
                    "— Tự nhập văn bản phản hồi mới —"
                    if sid == "CUSTOM_INPUT"
                    else sample_map[sid]["group_label"]
                ),
                key=f"sample_selector_{mode.value}",
                on_change=_on_sample_change,
            )

            if chosen_sample_id in sample_map:
                st.caption(f"Mục đích ca kiểm thử: {sample_map[chosen_sample_id]['note']}")

            input_text = st.text_area(
                "Nội dung ý kiến phản hồi (Tiếng Việt)",
                height=145,
                max_chars=config.max_input_chars,
                placeholder="Nhập câu phản hồi cần phân tích cảm xúc...",
                key=text_key,
            )
            char_len = len(input_text.strip())
            st.caption(f"Bộ đếm ký tự: **{char_len:,}** / {config.max_input_chars:,} ký tự.")

            meta_c1, meta_c2 = st.columns(2)
            with meta_c1:
                selected_unit = st.selectbox(
                    "Đơn vị liên quan",
                    options=list(unit_dict.keys()),
                    format_func=lambda k: unit_dict.get(k, k),
                    key=unit_key,
                )
            with meta_c2:
                selected_source = st.selectbox(
                    "Nguồn phản hồi",
                    options=list(source_dict.keys()),
                    format_func=lambda k: source_dict.get(k, k),
                    key=source_key,
                )

            selected_topic = st.selectbox(
                "Chủ đề (gán thủ công — chưa có mô hình chủ đề)",
                options=list(topic_dict.keys()),
                format_func=lambda k: topic_dict.get(k, k),
                key=topic_key,
                help="Do repository chưa có checkpoint phân loại chủ đề riêng, người dùng chọn thủ công theo mã ổn định, không tạo chủ đề ngẫu nhiên.",
            )

            matched_sample = analysis_service.find_matching_sample_case(input_text)
            confirmed_real = False
            origin_note = ""

            if mode == AppMode.DEMO or matched_sample is not None:
                st.info(
                    "Nguồn gốc dữ liệu khi lưu: **DEMO**. "
                    "Các câu mẫu có sẵn hoặc thao tác trong chế độ Demo luôn được gắn nhãn `DEMO`, "
                    "tuyệt đối không đưa vào thống kê thực tế ĐHTN."
                )
            else:
                confirmed_real = st.checkbox(
                    "Xác nhận đây là phản hồi thu thập từ khảo sát thực tế đã xác minh thuộc ĐHTN (TNU_REAL)",
                    value=False,
                    key=f"chk_confirm_real_{mode.value}",
                    help="Việc chọn đơn vị như ICTU không tự động chứng minh câu nhập tay thuộc khảo sát thực tế.",
                )
                if confirmed_real:
                    origin_note = st.text_input(
                        "Ghi chú đợt khảo sát / nguồn xác minh",
                        placeholder="Ví dụ: Phiếu khảo sát HK1 2026-2027 khoa CNTT",
                        key=f"txt_origin_note_{mode.value}",
                    )
                else:
                    st.caption(
                        "Nếu không tích xác nhận nguồn thực tế, bản ghi khi lưu sẽ được phân loại vào nhóm `DEMO`."
                    )

            # Kiểm tra điều kiện bật/tắt nút Phân tích và Lưu phản hồi
            can_analyze = char_len > 0 and (
                health.sentiment_model.ready or mode == AppMode.DEMO
            )
            analyze_disabled_reason = ""
            if char_len == 0:
                analyze_disabled_reason = "Vui lòng nhập nội dung phản hồi hoặc chọn một câu mẫu."
            elif mode == AppMode.REAL and not health.sentiment_model.ready:
                analyze_disabled_reason = (
                    "Chưa cấu hình checkpoint mô hình cảm xúc hợp lệ nên suy luận thật đã bị vô hiệu hóa."
                )

            outcome_key = f"current_analysis_outcome_{mode.value}"
            saved_key = f"current_analysis_saved_{mode.value}"
            flash_key = f"current_analysis_flash_{mode.value}"

            current_outcome: AnalysisOutcome | None = st.session_state.get(outcome_key)
            is_stale = analysis_service.is_outcome_stale(input_text, current_outcome)
            already_saved = bool(st.session_state.get(saved_key, False))

            can_save = (
                current_outcome is not None
                and current_outcome.prediction is not None
                and not current_outcome.error_message
                and not is_stale
                and not already_saved
                and health.database.ready
            )

            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                analyze_clicked = st.button(
                    "Phân tích",
                    type="primary",
                    width="stretch",
                    disabled=not can_analyze,
                    key=f"btn_do_analyze_{mode.value}",
                )
            with btn_c2:
                save_clicked = st.button(
                    "Lưu phản hồi",
                    width="stretch",
                    disabled=not can_save,
                    key=f"btn_do_save_{mode.value}",
                )

            flash_msg = st.session_state.pop(flash_key, None)
            if flash_msg is not None:
                lvl, msg_txt = flash_msg
                if lvl == "success":
                    st.success(msg_txt)
                else:
                    st.warning(msg_txt)

            if analyze_disabled_reason:
                st.caption(f"Lý do khóa nút Phân tích: {analyze_disabled_reason}")
            elif not can_save:
                if current_outcome is None:
                    st.caption("Nút 'Lưu phản hồi' sẽ mở sau khi bạn bấm 'Phân tích' và có kết quả hợp lệ.")
                elif is_stale:
                    st.caption("Đã khóa 'Lưu phản hồi' vì nội dung ô nhập đã thay đổi sau lần phân tích trước.")
                elif already_saved:
                    st.caption("Kết quả của lượt phân tích này đã được lưu (chống ghi lặp cùng thao tác).")
                elif not health.database.ready:
                    st.caption("Đã khóa 'Lưu phản hồi' vì kết nối CSDL MySQL chưa sẵn sàng.")

            if analyze_clicked:
                try:
                    new_outcome = analysis_service.analyze(
                        raw_text=input_text,
                        mode=mode,
                        unit_code=selected_unit,
                        source_code=selected_source,
                        manual_topic_code=selected_topic,
                        confirmed_tnu_real=confirmed_real,
                        origin_note=origin_note,
                    )
                    st.session_state[outcome_key] = new_outcome
                    st.session_state[saved_key] = False
                    st.session_state.pop(flash_key, None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

            if save_clicked and current_outcome is not None:
                try:
                    save_res = analysis_service.save_outcome(
                        repository=repository,
                        current_raw_text=input_text,
                        outcome=current_outcome,
                        override_topic_code=selected_topic,
                        override_unit_code=selected_unit,
                        override_source_code=selected_source,
                        override_confirmed_tnu_real=confirmed_real,
                        override_origin_note=origin_note,
                        mode=mode,
                    )
                    st.session_state[saved_key] = True
                    st.session_state[flash_key] = (
                        "success" if save_res.saved else "warning",
                        save_res.message,
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Không thể lưu phản hồi: {exc}")

    with col_right:
        with st.container(border=True):
            st.markdown("#### 2. Kết quả phân tích & Giải thích xử lý")

            outcome: AnalysisOutcome | None = st.session_state.get(outcome_key)
            if outcome is None:
                st.info(
                    "Chọn một câu mẫu bên trái hoặc nhập phản hồi tiếng Việt, sau đó bấm **Phân tích** "
                    "để xem nhãn cảm xúc, phân bố xác suất 3 lớp và quá trình tiền xử lý."
                )
                return

            if analysis_service.is_outcome_stale(input_text, outcome):
                st.warning(
                    "Nội dung văn bản hiện tại đã thay đổi so với lúc phân tích. "
                    "Kết quả bên dưới không còn khớp với ô nhập — vui lòng bấm **Phân tích** lại."
                )

            if outcome.error_message or outcome.prediction is None:
                st.error(outcome.error_message or "Chưa thể phân tích phản hồi này.")
                with st.expander("Xem quá trình xử lý sơ bộ (Che PII)", expanded=True):
                    st.markdown("**Văn bản sau khi che thông tin cá nhân (PII):**")
                    st.code(outcome.preprocess.masked_content, language=None)
                return

            pred = outcome.prediction

            if pred.is_illustrative_demo:
                st.warning(
                    "**Kết quả minh họa (cố định từ bộ mẫu Demo):** Do môi trường chưa nạp checkpoint PhoBERT, "
                    "kết quả dưới đây là dữ liệu mẫu cố định để minh họa giao diện, không phải suy luận thời gian thực."
                )

            # Hàng hiển thị nhãn chính
            sent_html = sentiment_badge_html(pred.label_id)
            rev_html = review_badge_html(outcome.review_status)
            topic_name_disp = topic_dict.get(selected_topic, selected_topic)
            topic_src_disp = TOPIC_SOURCE_LABELS.get(
                outcome.topic_source, outcome.topic_source
            )
            topic_pill = render_pill_html(
                f"Chủ đề: {topic_name_disp} ({topic_src_disp})",
                COLORS["neutral_bg"],
                COLORS["neutral_fg"],
                COLORS["border"],
            )
            st.markdown(
                f'<div class="tnu-badge-row" style="margin-bottom:0.85rem;">{sent_html}{rev_html}{topic_pill}</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(
                    "Nhãn cảm xúc",
                    SENTIMENT_DISPLAY_MAP[pred.label_id]["full_label"],
                )
            with m2:
                conf_disp = (
                    f"{pred.confidence:.2%}" if pred.confidence is not None else "Không có"
                )
                st.metric("Độ tự tin (Confidence)", conf_disp)
            with m3:
                lat_disp = (
                    f"{pred.inference_time_ms:.1f} ms"
                    if pred.inference_time_ms is not None
                    else "Minh họa cố định"
                )
                st.metric("Thời gian suy luận", lat_disp)

            st.caption(f"Phạm vi phép đo thời gian: {pred.measurement_scope}.")

            # Cảnh báo ngưỡng confidence thử nghiệm / đa ý / cắt token
            if (
                pred.confidence is not None
                and pred.confidence < config.confidence_review_threshold
            ):
                st.warning(
                    f"Độ tự tin ({pred.confidence:.2%}) thấp hơn ngưỡng thử nghiệm "
                    f"{config.confidence_review_threshold:.0%} → Hệ thống gắn trạng thái **Cần kiểm duyệt (NEEDS_REVIEW)**. "
                    "(Lưu ý: ngưỡng 0,65 là cấu hình thử nghiệm, chưa phải ngưỡng tối ưu đã kiểm chứng)."
                )
            else:
                st.caption(
                    f"Ngưỡng gắn cờ kiểm duyệt thử nghiệm hiện tại: {config.confidence_review_threshold:.0%}. "
                    "Độ tự tin cao phản ánh mức tập trung xác suất của mô hình, không phải bảo đảm kết quả luôn đúng."
                )

            if pred.was_truncated:
                st.warning(
                    f"Câu đầu vào có {pred.token_count} token, vượt giới hạn tối đa {pred.max_length} token của PhoBERT "
                    "nên phần cuối đã bị cắt bớt khi suy luận."
                )

            if outcome.preprocess.is_multi_aspect:
                st.info(
                    "**Lưu ý câu nhiều ý / tương phản:** Mô hình hiện tại thực hiện phân loại cảm xúc tổng thể "
                    "trên toàn câu (sentence-level), chưa tách nhãn cảm xúc riêng cho từng khía cạnh (aspect-based)."
                )

            # Biểu đồ xác suất 3 lớp (đúng ánh xạ 0: Negative, 1: Neutral, 2: Positive)
            prob_df = pd.DataFrame(
                [
                    {
                        "Nhãn": "Tiêu cực (0)",
                        "Xác suất (%)": round(pred.probabilities.get(0, 0.0) * 100.0, 2),
                    },
                    {
                        "Nhãn": "Trung tính (1)",
                        "Xác suất (%)": round(pred.probabilities.get(1, 0.0) * 100.0, 2),
                    },
                    {
                        "Nhãn": "Tích cực (2)",
                        "Xác suất (%)": round(pred.probabilities.get(2, 0.0) * 100.0, 2),
                    },
                ]
            )
            fig_prob = px.bar(
                prob_df,
                x="Xác suất (%)",
                y="Nhãn",
                color="Nhãn",
                orientation="h",
                text="Xác suất (%)",
                color_discrete_map=SENTIMENT_PLOTLY_COLOR_MAP,
            )
            fig_prob.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
            )
            fig_prob.update_layout(showlegend=False)
            fig_prob.update_xaxes(range=[0, 112])
            apply_tnu_plotly_style(
                fig_prob,
                title="Phân bố xác suất dự đoán 3 lớp cảm xúc",
                height=235,
                xaxis_title="Xác suất (%)",
                yaxis_title="",
            )
            st.plotly_chart(fig_prob, width="stretch")

            # Expander "Xem quá trình xử lý" mặc định đóng
            with st.expander("Xem quá trình xử lý", expanded=False):
                st.markdown("**1. Văn bản sau khi che thông tin định danh cá nhân (PII):**")
                st.code(outcome.preprocess.masked_content, language=None)
                if outcome.preprocess.pii_types_detected:
                    st.caption(
                        f"Đã che các loại PII: `{', '.join(outcome.preprocess.pii_types_detected)}`"
                    )

                st.markdown("**2. Các bước chuẩn hóa thực tế được pipeline áp dụng:**")
                for step in outcome.preprocess.steps:
                    st.markdown(f"- **{step.step_name}** — _{step.description}_")
                    st.code(step.output_text, language=None)

                st.markdown("**3. Văn bản cuối đưa vào mô hình:**")
                st.code(outcome.preprocess.processed_content, language=None)

                if outcome.preprocess.compatibility_notes:
                    for cnote in outcome.preprocess.compatibility_notes:
                        st.caption(f"Ghi chú tương thích: {cnote}")

                st.markdown("**4. Chi tiết kỹ thuật phiên bản:**")
                st.write(
                    f"- **Phiên bản pipeline:** `{outcome.preprocess.pipeline_version}` "
                    f"(Bộ tách từ: `{outcome.preprocess.segmenter_name}`)"
                )
                st.write(
                    f"- **Phiên bản mô hình:** `{pred.model_version}` "
                    f"(Số token: `{pred.token_count}/{pred.max_length}`)"
                )

            # Chức năng LLM hỗ trợ tùy chọn (không tự động gọi API)
            if config.llm_enabled:
                with st.expander("Trợ lý LLM hỗ trợ rà soát (Tùy chọn — không tự xác nhận)", expanded=False):
                    if not llm_helper.is_enabled_and_configured:
                        st.caption(llm_helper.health().technical_detail)
                    else:
                        st.caption(
                            "Văn bản luôn được che PII trước khi gửi. Nhãn gợi ý từ LLM chỉ mang tính tham khảo, "
                            "không tự động trở thành nhãn đã được con người xác nhận (VERIFIED)."
                        )
                        if st.button("Yêu cầu LLM gợi ý tham khảo", key=f"btn_ask_llm_{mode.value}"):
                            try:
                                sug = llm_helper.request_suggestion(
                                    outcome.preprocess.masked_content
                                )
                                outcome.llm_suggestion = sug
                                st.session_state[outcome_key] = outcome
                            except Exception as llm_exc:
                                st.error(f"Không thể lấy gợi ý từ LLM: {llm_exc}")

                        if outcome.llm_suggestion is not None:
                            s_lbl = outcome.llm_suggestion.suggested_sentiment_label
                            s_txt = (
                                SENTIMENT_DISPLAY_MAP[s_lbl]["full_label"]
                                if s_lbl in SENTIMENT_DISPLAY_MAP
                                else "Không rõ"
                            )
                            st.info(
                                f"**Gợi ý tham khảo ({outcome.llm_suggestion.model_name}):** "
                                f"Cảm xúc `{s_txt}` · Chủ đề `{outcome.llm_suggestion.suggested_topic_code}`\n\n"
                                f"Giải thích: {outcome.llm_suggestion.explanation}"
                            )
