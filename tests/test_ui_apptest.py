"""Kiểm thử tích hợp giao diện tất cả 4 phân hệ bằng Streamlit AppTest (headless UI testing)."""

from __future__ import annotations

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_demo_mode_without_mysql_and_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kiểm chứng điều kiện nghiệm thu 1, 2, 3, 5, 6, 11, 13 khi KHÔNG có MySQL và KHÔNG có checkpoint."""
    monkeypatch.setenv("SENTIMENT_CHECKPOINT_DIR", "")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("MYSQL_HOST", "")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, f"Uncaught exception on Overview: {at.exception}"

    full_md_overview = "\n".join(m.value for m in at.markdown)
    assert "Dữ liệu mô phỏng — chỉ phục vụ minh họa giao diện" in full_md_overview
    assert "sẽ mất khi phiên làm việc kết thúc" in full_md_overview
    assert "Chế độ demo" in full_md_overview
    assert "Chưa cấu hình mô hình" in full_md_overview
    assert "Chủ đề được gán thủ công" in full_md_overview

    # Giữ trạng thái bộ lọc khi chuyển từ Tổng quan sang Dữ liệu và kiểm duyệt
    at.multiselect(key="flt_DEMO_0_units").select("ICTU")
    at.run()
    assert not at.exception

    # Chuyển sang trang 3: Dữ liệu và kiểm duyệt -> bộ lọc đơn vị 'ICTU' phải được giữ nguyên
    at.sidebar.radio(key="sidebar_main_nav").set_value("data_review")
    at.run()
    assert not at.exception, f"Uncaught exception on Data Review: {at.exception}"
    assert at.multiselect(key="flt_DEMO_0_units").value == ["ICTU"]
    full_md_dr = "\n".join(m.value for m in at.markdown)
    assert "Dữ liệu mô phỏng — chỉ phục vụ minh họa giao diện" in full_md_dr

    # Chuyển sang trang 2: Phân tích phản hồi
    at.sidebar.radio(key="sidebar_main_nav").set_value("analyze")
    at.run()
    assert not at.exception, f"Uncaught exception on Analyze: {at.exception}"
    full_md_analyze = "\n".join(m.value for m in at.markdown)
    assert "Dữ liệu mô phỏng — chỉ phục vụ minh họa giao diện" in full_md_analyze

    # Văn bản tự nhập mới khi KHÔNG có mô hình -> tuyệt đối không tạo nhãn giả, báo rõ chưa thể phân tích
    at.text_area(key="analyze_input_text_DEMO").input(
        "Một câu phản hồi hoàn toàn mới không nằm trong danh sách câu mẫu cố định."
    )
    at.run()
    at.button(key="btn_do_analyze_DEMO").click()
    at.run()
    assert not at.exception
    assert len(at.error) >= 1
    assert "Chưa thể phân tích văn bản này" in at.error[0].value
    assert at.button(key="btn_do_save_DEMO").disabled is True

    # Chọn câu mẫu có sẵn trong chế độ Demo -> cho phép hiển thị 'Kết quả minh họa'
    at.text_area(key="analyze_input_text_DEMO").input(
        "Giảng viên dạy rất nhiệt tình nhưng máy chiếu phòng học mờ quá."
    )
    at.run()
    at.button(key="btn_do_analyze_DEMO").click()
    at.run()
    assert not at.exception, f"Sample analyze failed: {at.exception}"
    assert any("Kết quả minh họa" in w.value for w in at.warning)
    assert at.button(key="btn_do_save_DEMO").disabled is False

    # Sửa văn bản sau khi đã phân tích -> nút Lưu phản hồi phải bị khóa ngay!
    at.text_area(key="analyze_input_text_DEMO").input(
        "Giảng viên dạy rất nhiệt tình nhưng máy chiếu phòng học đã sửa tốt rồi."
    )
    at.run()
    assert not at.exception
    assert at.button(key="btn_do_save_DEMO").disabled is True
    assert any("đã thay đổi so với lúc phân tích" in w.value for w in at.warning)

    # Chuyển sang trang 4: Đánh giá mô hình
    at.sidebar.radio(key="sidebar_main_nav").set_value("evaluation")
    at.run()
    assert not at.exception, f"Uncaught exception on Evaluation: {at.exception}"
    full_md_eval = "\n".join(m.value for m in at.markdown)
    assert "Dữ liệu mô phỏng — chỉ phục vụ minh họa giao diện" in full_md_eval

    # Chuyển sang Chế độ kết nối dữ liệu thật (khi chưa bật MySQL):
    # Phải giữ nguyên chế độ REAL, hiển thị lỗi kết nối và KHÔNG tự nhảy về dữ liệu giả!
    at.sidebar.radio(key="sidebar_main_nav").set_value("overview")
    at.sidebar.radio[1].set_value("REAL")
    at.run()
    assert not at.exception, f"Uncaught exception on REAL guard: {at.exception}"
    assert len(at.error) >= 1
    assert "KHÔNG tự động điền dữ liệu mô phỏng" in at.error[0].value

    # Bấm nút xác nhận chuyển sang Demo -> chuyển về Demo thành công
    at.button(key="btn_guard_to_demo").click()
    at.run()
    assert not at.exception
    assert any("Dữ liệu mô phỏng" in m.value for m in at.markdown)


def test_analyze_and_save_with_real_phobert_checkpoint() -> None:
    """Kiểm chứng luồng nhập -> suy luận PhoBERT thực tế -> lưu không trùng lặp khi có checkpoint."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    header_md = "\n".join(m.value for m in at.markdown)
    assert "Mô hình cảm xúc sẵn sàng" in header_md

    at.sidebar.radio(key="sidebar_main_nav").set_value("analyze")
    at.run()
    assert not at.exception

    at.text_area(key="analyze_input_text_DEMO").input(
        "Thầy cô khoa CNTT hỗ trợ sinh viên làm đồ án rất tận tâm và chi tiết."
    )
    at.run()
    at.button(key="btn_do_analyze_DEMO").click()
    at.run()
    assert not at.exception
    assert at.button(key="btn_do_save_DEMO").disabled is False

    # Bấm Lưu phản hồi -> lưu thành công và khóa nút Lưu ngay để chống ghi lặp cùng thao tác
    at.button(key="btn_do_save_DEMO").click()
    at.run()
    assert not at.exception
    assert len(at.success) >= 1
    assert at.button(key="btn_do_save_DEMO").disabled is True
