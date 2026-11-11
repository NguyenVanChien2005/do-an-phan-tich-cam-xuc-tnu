"""Interface trừu tượng cho tầng truy cập dữ liệu phản hồi và kết quả đánh giá mô hình."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

from tnu_sentiment.domain.models import (
    BenchmarkRun,
    CatalogItem,
    ComponentHealth,
    FeedbackRecord,
    FilterCriteria,
    ImportExecutionSummary,
    PaginatedRecords,
    ReviewLogEntry,
    ReviewUpdateRequest,
    SaveRecordResult,
)


@dataclass
class NewFeedbackPayload:
    data_origin: str
    origin_note: Optional[str]
    unit_code: str
    source_code: str
    masked_content: str
    processed_content: Optional[str]
    pii_types_detected: Optional[str]
    was_truncated: bool
    is_multi_aspect: bool
    processing_status: str
    processing_error: Optional[str]
    predicted_sentiment_label: Optional[int]
    predicted_sentiment_text: Optional[str]
    confidence_score: Optional[float]
    prob_negative: Optional[float]
    prob_neutral: Optional[float]
    prob_positive: Optional[float]
    predicted_topic_code: str
    topic_assignment_source: str
    model_version: Optional[str]
    pipeline_version: Optional[str]
    inference_time_ms: Optional[float]
    review_status: str
    review_reason: Optional[str]
    feedback_date: datetime
    idempotency_key: Optional[str] = None
    source_record_id: Optional[str] = None
    source_url: Optional[str] = None
    post_id: Optional[str] = None
    parent_comment_id: Optional[str] = None
    collected_at: Optional[datetime] = None
    content_hash: str = ""
    author_hash: Optional[str] = None
    batch_id: Optional[str] = None
    llm_suggested_sentiment: Optional[int] = None
    llm_suggested_topic_code: Optional[str] = None
    llm_note: Optional[str] = None


def compute_half_open_date_bounds(
    date_from: Optional[date],
    date_to: Optional[date],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Chuyển khoảng lọc ngày sang khoảng nửa mở [start_dt, end_exclusive_dt)
    theo giờ địa phương Asia/Ho_Chi_Minh đã thống nhất lưu trữ.
    """
    start_dt = datetime.combine(date_from, time.min) if date_from is not None else None
    end_excl = (
        datetime.combine(date_to + timedelta(days=1), time.min)
        if date_to is not None
        else None
    )
    return start_dt, end_excl


class FeedbackRepository(ABC):
    """Hợp đồng truy xuất dữ liệu dùng chung cho cả MySQL và Demo trong phiên."""

    @property
    @abstractmethod
    def repository_kind(self) -> str: ...

    @abstractmethod
    def health(self) -> ComponentHealth: ...

    @abstractmethod
    def list_units(self) -> List[CatalogItem]: ...

    @abstractmethod
    def list_sources(self) -> List[CatalogItem]: ...

    @abstractmethod
    def list_topics(self) -> List[CatalogItem]: ...

    @abstractmethod
    def list_paginated(
        self,
        filters: FilterCriteria,
        page: int = 1,
        page_size: int = 15,
        sort_by: str = "newest",
        only_negative_for_review: bool = False,
    ) -> PaginatedRecords: ...

    @abstractmethod
    def list_matching_for_stats(
        self,
        filters: FilterCriteria,
    ) -> List[FeedbackRecord]: ...

    @abstractmethod
    def get_by_id(self, feedback_id: int) -> Optional[FeedbackRecord]: ...

    @abstractmethod
    def check_source_record_exists(
        self, source_code: str, source_record_id: str
    ) -> bool: ...

    @abstractmethod
    def save_feedback(self, payload: NewFeedbackPayload) -> SaveRecordResult: ...

    @abstractmethod
    def apply_review_updates(
        self, updates: List[ReviewUpdateRequest]
    ) -> Tuple[int, List[str]]: ...

    @abstractmethod
    def list_review_logs(
        self, feedback_id: Optional[int] = None, limit: int = 50
    ) -> List[ReviewLogEntry]: ...

    @abstractmethod
    def export_records(
        self, filters: FilterCriteria, limit: int = 5000
    ) -> List[Dict[str, object]]: ...

    def save_import_batch(
        self, summary: ImportExecutionSummary, origin_note: Optional[str] = None
    ) -> None:
        """Lưu lịch sử lô import vào bảng import_batches (mặc định no-op nếu lớp con không ghi đè)."""
        return None

    def list_import_batches(self, limit: int = 20) -> List[Dict[str, object]]:
        """Liệt kê lịch sử các lô import từ bảng import_batches."""
        return []


class EvaluationRepository(ABC):
    @abstractmethod
    def load_runs(self, include_demo_fixtures: bool = False) -> List[BenchmarkRun]: ...

