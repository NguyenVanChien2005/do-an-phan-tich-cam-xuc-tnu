"""Các hằng số giao diện, bảng màu, nhãn chuẩn và danh mục khởi tạo tham khảo."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple

# ------------------------------------------------------------------------------
# Thông tin đồ án (chỉ hiển thị gọn ở mục Thông tin đồ án / Footer)
# ------------------------------------------------------------------------------
PROJECT_META = {
    "short_name": "TNU Sentiment",
    "subtitle": "Phân tích cảm xúc từ phản hồi cộng đồng",
    "thesis_title": "Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên",
    "student_name": "Nguyễn Văn Chiến",
    "student_id": "DTC235200079",
    "class_name": "CN CNTT K22A",
    "institution": "Trường Đại học Công nghệ Thông tin và Truyền thông — ICTU",
    "advisor_name": "TS. Hà Thị Thanh",
}

# ------------------------------------------------------------------------------
# Bảng màu cố định theo đặc tả thiết kế
# ------------------------------------------------------------------------------
COLORS = {
    "brand_primary": "#1E3A8A",
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "border": "#E2E8F0",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    # Màu cảm xúc đồng nhất giữa badge, bảng và Plotly
    "positive": "#1A73E8",
    "positive_bg": "#E8F0FE",
    "positive_fg": "#174EA6",
    "neutral": "#64748B",
    "neutral_bg": "#F1F5F9",
    "neutral_fg": "#334155",
    "negative": "#D93025",
    "negative_bg": "#FCE8E6",
    "negative_fg": "#A50E0E",
    "needs_review": "#F59E0B",
    "needs_review_bg": "#FEF3C7",
    "needs_review_fg": "#92400E",
}

# ------------------------------------------------------------------------------
# Quy ước nhãn cảm xúc của checkpoint PhoBERT (0: Negative, 1: Neutral, 2: Positive)
# ------------------------------------------------------------------------------
SENTIMENT_ID_TO_CODE: Dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

SENTIMENT_CODE_TO_ID: Dict[str, int] = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

SENTIMENT_ID_TO_VI: Dict[int, str] = {
    0: "Tiêu cực",
    1: "Trung tính",
    2: "Tích cực",
}

SENTIMENT_DISPLAY_MAP: Dict[int, Dict[str, str]] = {
    0: {
        "code": "Negative",
        "label_vi": "Tiêu cực",
        "full_label": "Tiêu cực (0)",
        "color": COLORS["negative"],
        "bg": COLORS["negative_bg"],
        "fg": COLORS["negative_fg"],
    },
    1: {
        "code": "Neutral",
        "label_vi": "Trung tính",
        "full_label": "Trung tính (1)",
        "color": COLORS["neutral"],
        "bg": COLORS["neutral_bg"],
        "fg": COLORS["neutral_fg"],
    },
    2: {
        "code": "Positive",
        "label_vi": "Tích cực",
        "full_label": "Tích cực (2)",
        "color": COLORS["positive"],
        "bg": COLORS["positive_bg"],
        "fg": COLORS["positive_fg"],
    },
}


class AppMode(str, Enum):
    REAL = "REAL"
    DEMO = "DEMO"


class DataOrigin(str, Enum):
    TNU_REAL = "TNU_REAL"
    PUBLIC_BENCHMARK = "PUBLIC_BENCHMARK"
    SYNTHETIC = "SYNTHETIC"
    DEMO = "DEMO"


DATA_ORIGIN_LABELS: Dict[str, str] = {
    DataOrigin.TNU_REAL.value: "Phản hồi thực tế ĐHTN (TNU_REAL)",
    DataOrigin.PUBLIC_BENCHMARK.value: "Benchmark công khai (PUBLIC_BENCHMARK)",
    DataOrigin.SYNTHETIC.value: "Dữ liệu sinh bổ sung (SYNTHETIC)",
    DataOrigin.DEMO.value: "Dữ liệu minh họa giao diện (DEMO)",
}


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


PROCESSING_STATUS_LABELS: Dict[str, str] = {
    ProcessingStatus.PENDING.value: "Chờ xử lý (PENDING)",
    ProcessingStatus.RUNNING.value: "Đang xử lý (RUNNING)",
    ProcessingStatus.DONE.value: "Đã xử lý (DONE)",
    ProcessingStatus.ERROR.value: "Lỗi xử lý (ERROR)",
}


class ReviewStatus(str, Enum):
    UNREVIEWED = "UNREVIEWED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    VERIFIED = "VERIFIED"


REVIEW_STATUS_LABELS: Dict[str, str] = {
    ReviewStatus.UNREVIEWED.value: "Chưa kiểm duyệt",
    ReviewStatus.NEEDS_REVIEW.value: "Cần kiểm duyệt",
    ReviewStatus.VERIFIED.value: "Đã xác nhận",
}


class LabelPolicy(str, Enum):
    VERIFIED_FIRST = "VERIFIED_FIRST"
    VERIFIED_ONLY = "VERIFIED_ONLY"


LABEL_POLICY_LABELS: Dict[str, str] = {
    LabelPolicy.VERIFIED_FIRST.value: "Dùng nhãn đã duyệt khi có, nếu chưa có dùng dự đoán",
    LabelPolicy.VERIFIED_ONLY.value: "Chỉ nhãn đã duyệt",
}


class TopicAssignmentSource(str, Enum):
    MANUAL = "MANUAL"
    MODEL = "MODEL"
    RULE = "RULE"
    IMPORT = "IMPORT"
    REVIEWER = "REVIEWER"


TOPIC_SOURCE_LABELS: Dict[str, str] = {
    TopicAssignmentSource.MANUAL.value: "Gán thủ công",
    TopicAssignmentSource.MODEL.value: "Mô hình dự đoán",
    TopicAssignmentSource.RULE.value: "Quy tắc từ khóa",
    TopicAssignmentSource.IMPORT.value: "Theo tệp Import",
    TopicAssignmentSource.REVIEWER.value: "Người duyệt xác nhận",
}

# ------------------------------------------------------------------------------
# Danh mục khởi tạo tham khảo (không đóng cứng cơ cấu toàn ĐHTN; repository nạp động)
# ------------------------------------------------------------------------------
INITIAL_UNITS: List[Tuple[str, str]] = [
    ("UNSPECIFIED", "Chưa xác định"),
    ("TNU_COMMON", "Khối chung ĐHTN"),
    ("ICTU", "Trường ĐH Công nghệ Thông tin và Truyền thông (ICTU)"),
    ("TNUT", "Trường ĐH Kỹ thuật Công nghiệp (TNUT)"),
    ("TUEBA", "Trường ĐH Kinh tế và Quản trị Kinh doanh (TUEBA)"),
    ("TNUE", "Trường ĐH Sư phạm (TNUE)"),
    ("TUAF", "Trường ĐH Nông Lâm (TUAF)"),
    ("TUMP", "Trường ĐH Y - Dược (TUMP)"),
    ("TNUS", "Trường ĐH Khoa học (TNUS)"),
    ("SFL", "Trường Ngoại ngữ (SFL)"),
    ("ISTNU", "Khoa Quốc tế (ISTNU)"),
]

INITIAL_SOURCES: List[Tuple[str, str]] = [
    ("UNSPECIFIED", "Chưa xác định"),
    ("SURVEY_PORTAL", "Khảo sát ý kiến người học"),
    ("LMS_FEEDBACK", "Hệ thống học tập trực tuyến (LMS)"),
    ("STUDENT_FORUM", "Kênh phản hồi sinh viên"),
    ("QA_BOX", "Hòm thư góp ý đơn vị"),
    ("CSV_IMPORT", "Nhập từ tệp CSV/XLSX"),
]

# Mapping chủ đề bằng mã ổn định (topic_code), KHÔNG dựa vào argmax + 1
INITIAL_TOPICS: List[Tuple[str, str, int | None]] = [
    ("UNCLASSIFIED", "Chưa phân loại", None),
    ("LECTURER", "Giảng viên & Phương pháp giảng dạy", 0),
    ("CURRICULUM", "Chương trình đào tạo & Học phần", 1),
    ("FACILITY", "Cơ sở vật chất & Hạ tầng CNTT", 2),
    ("OTHER_SUPPORT", "Công tác sinh viên & Dịch vụ khác", 3),
]

CORPUS_IDX_TO_TOPIC_CODE: Dict[int, str] = {
    idx: code for code, _name, idx in INITIAL_TOPICS if idx is not None
}

# Nhãn trạng thái hướng tới người dùng trên Header
USER_STATUS_LABELS = {
    "db_connected": "Đã kết nối dữ liệu",
    "demo_mode": "Chế độ demo",
    "db_disconnected": "Chưa kết nối dữ liệu",
    "sentiment_ready": "Mô hình cảm xúc sẵn sàng",
    "sentiment_missing": "Chưa cấu hình mô hình",
    "topic_manual": "Chủ đề được gán thủ công",
    "topic_ready": "Mô hình chủ đề sẵn sàng",
}

PIPELINE_VERSION = "tnu-preprocess-v1.1-pyvi"

