"""Định nghĩa kiểu dữ liệu cho phản hồi, dự đoán, bộ lọc, kiểm duyệt và kết quả thực nghiệm."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from tnu_sentiment.constants import (
    DataOrigin,
    LabelPolicy,
    ProcessingStatus,
    ReviewStatus,
    SENTIMENT_ID_TO_VI,
    TopicAssignmentSource,
)


@dataclass(frozen=True)
class CatalogItem:
    code: str
    name: str
    extra: Optional[str] = None


@dataclass(frozen=True)
class PreprocessStepTrace:
    step_name: str
    description: str
    output_text: str


@dataclass(frozen=True)
class PreprocessResult:
    masked_content: str
    processed_content: str
    pii_types_detected: List[str]
    is_multi_aspect: bool
    steps: List[PreprocessStepTrace]
    pipeline_version: str
    segmenter_name: str
    compatibility_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SentimentPrediction:
    label_id: int
    label_code: str
    label_vi: str
    confidence: Optional[float]
    probabilities: Dict[int, float]  # {0: p_neg, 1: p_neu, 2: p_pos}
    model_version: str
    was_truncated: bool
    token_count: int
    max_length: int
    inference_time_ms: Optional[float]
    measurement_scope: str
    is_illustrative_demo: bool = False


@dataclass(frozen=True)
class LLMSuggestion:
    suggested_sentiment_label: Optional[int]
    suggested_topic_code: Optional[str]
    explanation: str
    model_name: str


@dataclass
class AnalysisOutcome:
    """Kết quả đầy đủ sau khi người dùng bấm 'Phân tích' tại phân hệ Phân tích phản hồi."""

    idempotency_key: str
    analyzed_raw_length: int
    analyzed_text_signature: str
    preprocess: PreprocessResult
    prediction: Optional[SentimentPrediction]
    selected_topic_code: str
    topic_source: str
    unit_code: str
    source_code: str
    data_origin: str
    origin_note: Optional[str]
    review_status: str
    review_reason: Optional[str]
    total_elapsed_ms: Optional[float]
    error_message: Optional[str] = None
    llm_suggestion: Optional[LLMSuggestion] = None


@dataclass
class FeedbackRecord:
    """Thực thể phản hồi chuẩn hóa dùng chung cho cả MySQLRepository và DemoRepository."""

    feedback_id: int
    data_origin: str
    origin_note: Optional[str]
    unit_code: str
    unit_name: str
    source_code: str
    source_name: str
    masked_content: str
    processed_content: Optional[str]
    pii_types_detected: Optional[str]
    was_truncated: bool
    is_multi_aspect: bool
    is_duplicate_candidate: bool
    processing_status: str
    processing_error: Optional[str]

    predicted_sentiment_label: Optional[int]
    predicted_sentiment_text: Optional[str]
    confidence_score: Optional[float]
    prob_negative: Optional[float]
    prob_neutral: Optional[float]
    prob_positive: Optional[float]

    predicted_topic_code: str
    predicted_topic_name: str
    topic_assignment_source: str

    model_version: Optional[str]
    pipeline_version: Optional[str]
    inference_time_ms: Optional[float]

    review_status: str
    review_reason: Optional[str]
    reviewed_sentiment_label: Optional[int]
    reviewed_topic_code: Optional[str]
    reviewed_topic_name: Optional[str]
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]
    review_note: Optional[str]
    record_version: int

    feedback_date: datetime
    created_at: datetime
    updated_at: datetime

    idempotency_key: Optional[str] = None
    source_record_id: Optional[str] = None
    source_url: Optional[str] = None
    post_id: Optional[str] = None
    parent_comment_id: Optional[str] = None
    collected_at: Optional[datetime] = None
    content_hash: str = ""

    def effective_sentiment_label(self, policy: LabelPolicy) -> Optional[int]:
        """Xác định nhãn cảm xúc hợp lệ theo chính sách nhãn đang áp dụng.
        Lưu ý: Bản ghi chưa xử lý thành công (PENDING/RUNNING/ERROR) không có nhãn hợp lệ.
        """
        if self.processing_status != ProcessingStatus.DONE.value:
            return None
        if (
            self.review_status == ReviewStatus.VERIFIED.value
            and self.reviewed_sentiment_label in (0, 1, 2)
        ):
            return self.reviewed_sentiment_label
        if policy == LabelPolicy.VERIFIED_ONLY:
            return None
        if self.predicted_sentiment_label in (0, 1, 2):
            return self.predicted_sentiment_label
        return None

    def effective_topic_code(self, policy: LabelPolicy) -> str:
        if (
            self.review_status == ReviewStatus.VERIFIED.value
            and self.reviewed_topic_code
        ):
            return self.reviewed_topic_code
        if policy == LabelPolicy.VERIFIED_ONLY:
            return "UNCLASSIFIED"
        return self.predicted_topic_code or "UNCLASSIFIED"

    @property
    def predicted_sentiment_vi(self) -> str:
        if self.predicted_sentiment_label in SENTIMENT_ID_TO_VI:
            return SENTIMENT_ID_TO_VI[self.predicted_sentiment_label]
        return "Chưa có"

    @property
    def reviewed_sentiment_vi(self) -> str:
        if self.reviewed_sentiment_label in SENTIMENT_ID_TO_VI:
            return SENTIMENT_ID_TO_VI[self.reviewed_sentiment_label]
        return "Chưa duyệt"


@dataclass
class FilterCriteria:
    """Bộ lọc dùng chung và đồng bộ trạng thái giữa trang Tổng quan và Dữ liệu."""

    data_origins: List[str] = field(default_factory=lambda: [DataOrigin.TNU_REAL.value])
    select_all_origins: bool = False
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    unit_codes: List[str] = field(default_factory=list)
    source_codes: List[str] = field(default_factory=list)
    topic_codes: List[str] = field(default_factory=list)
    processing_statuses: List[str] = field(default_factory=list)
    review_statuses: List[str] = field(default_factory=list)
    sentiment_filter: List[int] = field(default_factory=list)
    search_query: str = ""
    label_policy: LabelPolicy = LabelPolicy.VERIFIED_FIRST


@dataclass(frozen=True)
class PaginatedRecords:
    items: List[FeedbackRecord]
    total_count: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        if self.total_count <= 0:
            return 1
        return (self.total_count + self.page_size - 1) // self.page_size


@dataclass(frozen=True)
class KPISummary:
    n_total: int
    n_done: int
    n_pending_or_running: int
    n_error: int
    n_needs_review: int
    n_verified: int
    n_valid: int
    n_pos: int
    n_neu: int
    n_neg: int
    pos_ratio_pct: Optional[float]
    neu_ratio_pct: Optional[float]
    neg_ratio_pct: Optional[float]
    nss_score: Optional[float]
    avg_confidence: Optional[float]
    avg_confidence_n: int
    avg_confidence_model_note: str
    origin_breakdown: Dict[str, int]
    is_mixed_origins: bool
    latest_updated_at: Optional[datetime]
    alert_triggered: bool
    alert_message: Optional[str]


@dataclass(frozen=True)
class SaveRecordResult:
    saved: bool
    was_idempotent_duplicate: bool
    was_source_duplicate: bool
    is_content_duplicate_candidate: bool
    feedback_id: Optional[int]
    message: str


@dataclass(frozen=True)
class ReviewUpdateRequest:
    feedback_id: int
    expected_version: int
    new_review_status: str
    reviewed_sentiment_label: Optional[int]
    reviewed_topic_code: Optional[str]
    reviewed_by: str
    review_note: Optional[str]


@dataclass(frozen=True)
class ReviewLogEntry:
    log_id: int
    feedback_id: int
    previous_review_status: Optional[str]
    new_review_status: str
    previous_sentiment_label: Optional[int]
    new_sentiment_label: Optional[int]
    previous_topic_code: Optional[str]
    new_topic_code: Optional[str]
    reviewed_by: str
    is_authenticated_user: bool
    review_note: Optional[str]
    record_version_before: int
    record_version_after: int
    changed_at: datetime


@dataclass(frozen=True)
class ImportRowPreview:
    row_number: int  # 1-based data row index in file
    masked_sentence: str
    unit_code: str
    source_code: str
    feedback_date: Optional[datetime]
    source_record_id: Optional[str]
    topic_code: str
    is_valid: bool
    errors: List[str]
    source_url: Optional[str] = None
    post_id: Optional[str] = None
    parent_comment_id: Optional[str] = None
    collected_at: Optional[datetime] = None
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class ImportExecutionSummary:
    batch_id: str
    filename: str
    declared_origin: str
    total_rows_read: int
    saved_count: int
    duplicate_skipped_count: int
    validation_error_count: int
    inference_error_count: int
    db_error_count: int
    unprocessed_count: int
    row_errors: List[Dict[str, object]]

    @property
    def accounted_rows(self) -> int:
        return (
            self.saved_count
            + self.duplicate_skipped_count
            + self.validation_error_count
            + self.inference_error_count
            + self.db_error_count
            + self.unprocessed_count
        )


@dataclass(frozen=True)
class ClassMetric:
    class_id: str
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ErrorCaseSample:
    sample_id: str
    masked_text: str
    true_label: str
    predicted_label: str
    confidence: Optional[float]
    error_category: str
    analysis_note: str


@dataclass(frozen=True)
class BenchmarkRun:
    run_id: str
    task: str  # "SENTIMENT" | "TOPIC"
    model_name: str
    checkpoint_version: str
    dataset_version: str
    split: str  # "Validation" | "Test"
    evaluation_set_id: str
    metric_definition_version: str
    n_samples: int
    evaluated_at: str
    result_source: str  # "ARTIFACT_VERIFIED" | "DEMO_FIXTURE"
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: List[ClassMetric]
    confusion_matrix_labels: List[str]
    confusion_matrix: List[List[int]]
    error_cases: List[ErrorCaseSample] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass(frozen=True)
class ComponentHealth:
    ready: bool
    user_badge: str
    technical_detail: str


@dataclass(frozen=True)
class SystemHealthStatus:
    mode: str
    database: ComponentHealth
    sentiment_model: ComponentHealth
    topic_model: ComponentHealth
    llm_helper: ComponentHealth

