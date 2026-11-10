"""Dịch vụ tính toán KPI thống kê và dữ liệu biểu đồ cho trang Tổng quan."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from tnu_sentiment.config import AppConfig
from tnu_sentiment.constants import (
    DataOrigin,
    LabelPolicy,
    ProcessingStatus,
    ReviewStatus,
    SENTIMENT_ID_TO_VI,
)
from tnu_sentiment.domain.models import FeedbackRecord, KPISummary


class StatsService:
    """Tính KPI trung thực theo đúng định nghĩa mẫu số N_valid và chính sách nhãn."""

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config

    def compute_kpis(
        self,
        records: List[FeedbackRecord],
        label_policy: LabelPolicy,
    ) -> KPISummary:
        n_total = len(records)
        n_done = 0
        n_pending_or_running = 0
        n_error = 0
        n_needs_review = 0
        n_verified = 0

        n_valid = 0
        n_pos = 0
        n_neu = 0
        n_neg = 0

        origin_counts: Counter[str] = Counter()
        conf_by_model: Dict[str, List[float]] = {}
        latest_updated: Optional[datetime] = None

        for r in records:
            origin_counts[r.data_origin] += 1

            if latest_updated is None or r.updated_at > latest_updated:
                latest_updated = r.updated_at

            if r.processing_status == ProcessingStatus.DONE.value:
                n_done += 1
            elif r.processing_status == ProcessingStatus.ERROR.value:
                n_error += 1
            else:
                n_pending_or_running += 1

            if r.review_status == ReviewStatus.NEEDS_REVIEW.value:
                n_needs_review += 1
            elif r.review_status == ReviewStatus.VERIFIED.value:
                n_verified += 1

            eff_label = r.effective_sentiment_label(label_policy)
            if eff_label in (0, 1, 2):
                n_valid += 1
                if eff_label == 2:
                    n_pos += 1
                elif eff_label == 1:
                    n_neu += 1
                elif eff_label == 0:
                    n_neg += 1

            # Độ tự tin trung bình: chỉ tính trên dự đoán DONE có confidence hợp lệ [0,1] và nguồn model tương thích
            if (
                r.processing_status == ProcessingStatus.DONE.value
                and r.confidence_score is not None
                and 0.0 <= float(r.confidence_score) <= 1.0
                and r.model_version
                and not str(r.model_version).startswith("DEMO_STATIC")
            ):
                conf_by_model.setdefault(str(r.model_version), []).append(
                    float(r.confidence_score)
                )

        if n_valid > 0:
            pos_pct: Optional[float] = round(100.0 * n_pos / n_valid, 2)
            neu_pct: Optional[float] = round(100.0 * n_neu / n_valid, 2)
            neg_pct: Optional[float] = round(100.0 * n_neg / n_valid, 2)
            nss: Optional[float] = round(100.0 * (n_pos - n_neg) / n_valid, 2)
        else:
            # Không có nhãn hợp lệ -> trả None để UI hiển thị "Chưa có dữ liệu", tuyệt đối không giả định 0%
            pos_pct = None
            neu_pct = None
            neg_pct = None
            nss = None

        avg_conf: Optional[float] = None
        avg_conf_n = 0
        avg_conf_note = "Chưa có mẫu dự đoán từ mô hình thực tế có độ tự tin tương thích."
        if conf_by_model:
            # Chỉ tính trung bình trên phiên bản mô hình chiếm đa số để không trộn thang xác suất khác mô hình
            dominant_model, vals = max(conf_by_model.items(), key=lambda kv: len(kv[1]))
            avg_conf = round(sum(vals) / len(vals), 4)
            avg_conf_n = len(vals)
            extra = (
                f" (bỏ qua {len(conf_by_model) - 1} nguồn model khác phiên bản)"
                if len(conf_by_model) > 1
                else ""
            )
            avg_conf_note = f"Tính trên {avg_conf_n} dự đoán của {dominant_model}{extra} — không phải Accuracy."

        non_zero_origins = [k for k, v in origin_counts.items() if v > 0]
        is_mixed = len(non_zero_origins) > 1 or (
            len(non_zero_origins) == 1
            and non_zero_origins[0] not in (DataOrigin.TNU_REAL.value, DataOrigin.DEMO.value)
        )

        alert_triggered = False
        alert_msg: Optional[str] = None
        min_samples = self._cfg.negative_alert_min_samples
        ratio_thresh = self._cfg.negative_alert_ratio_threshold
        if n_valid >= min_samples and (n_neg / n_valid) >= ratio_thresh:
            alert_triggered = True
            alert_msg = (
                f"Tỷ lệ phản hồi tiêu cực hiện ở mức {neg_pct:.1f}% ({n_neg}/{n_valid} mẫu hợp lệ), "
                f"vượt ngưỡng cấu hình minh họa {ratio_thresh * 100:.0f}% (áp dụng khi N_valid >= {min_samples}). "
                "Lưu ý: đây chỉ là quy tắc cảnh báo cấu hình minh họa phục vụ theo dõi quản trị, "
                "không phải ngưỡng an toàn đã được kiểm chứng thực nghiệm."
            )

        return KPISummary(
            n_total=n_total,
            n_done=n_done,
            n_pending_or_running=n_pending_or_running,
            n_error=n_error,
            n_needs_review=n_needs_review,
            n_verified=n_verified,
            n_valid=n_valid,
            n_pos=n_pos,
            n_neu=n_neu,
            n_neg=n_neg,
            pos_ratio_pct=pos_pct,
            neu_ratio_pct=neu_pct,
            neg_ratio_pct=neg_pct,
            nss_score=nss,
            avg_confidence=avg_conf,
            avg_confidence_n=avg_conf_n,
            avg_confidence_model_note=avg_conf_note,
            origin_breakdown=dict(origin_counts),
            is_mixed_origins=is_mixed,
            latest_updated_at=latest_updated,
            alert_triggered=alert_triggered,
            alert_message=alert_msg,
        )

    @staticmethod
    def build_valid_dataframe(
        records: List[FeedbackRecord],
        label_policy: LabelPolicy,
        topic_lookup: Dict[str, str],
    ) -> pd.DataFrame:
        rows = []
        for r in records:
            eff_lbl = r.effective_sentiment_label(label_policy)
            if eff_lbl not in (0, 1, 2):
                continue
            eff_topic = r.effective_topic_code(label_policy)
            rows.append(
                {
                    "feedback_id": r.feedback_id,
                    "feedback_date": r.feedback_date,
                    "sentiment_id": eff_lbl,
                    "sentiment_vi": SENTIMENT_ID_TO_VI[eff_lbl],
                    "unit_code": r.unit_code,
                    "unit_name": r.unit_name,
                    "source_code": r.source_code,
                    "source_name": r.source_name,
                    "topic_code": eff_topic,
                    "topic_name": topic_lookup.get(eff_topic, eff_topic),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "feedback_id",
                    "feedback_date",
                    "sentiment_id",
                    "sentiment_vi",
                    "unit_code",
                    "unit_name",
                    "source_code",
                    "source_name",
                    "topic_code",
                    "topic_name",
                ]
            )
        return pd.DataFrame(rows)

    @staticmethod
    def aggregate_trend(valid_df: pd.DataFrame, granularity: str = "day") -> pd.DataFrame:
        """Tổng hợp xu hướng theo ngày ('day') hoặc tuần ('week'), kèm mẫu số N_valid từng mốc."""
        if valid_df.empty:
            return pd.DataFrame()
        df = valid_df.copy()
        dt_series = pd.to_datetime(df["feedback_date"])
        if granularity == "week":
            df["bucket"] = dt_series.dt.to_period("W-MON").apply(lambda p: p.start_time.strftime("%Y-%m-%d"))
        else:
            df["bucket"] = dt_series.dt.strftime("%Y-%m-%d")

        grouped = (
            df.groupby(["bucket", "sentiment_id", "sentiment_vi"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        bucket_totals = (
            grouped.groupby("bucket", as_index=False)["count"]
            .sum()
            .rename(columns={"count": "bucket_n_valid"})
        )
        merged = grouped.merge(bucket_totals, on="bucket", how="left")
        merged["ratio_pct"] = (100.0 * merged["count"] / merged["bucket_n_valid"]).round(2)
        return merged.sort_values(["bucket", "sentiment_id"])

    @staticmethod
    def aggregate_by_dimension(
        valid_df: pd.DataFrame,
        code_col: str,
        name_col: str,
    ) -> pd.DataFrame:
        if valid_df.empty:
            return pd.DataFrame()
        grouped = (
            valid_df.groupby([code_col, name_col, "sentiment_id", "sentiment_vi"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        dim_totals = (
            grouped.groupby([code_col, name_col], as_index=False)["count"]
            .sum()
            .rename(columns={"count": "group_n_valid"})
        )
        merged = grouped.merge(dim_totals, on=[code_col, name_col], how="left")
        merged["ratio_pct"] = (100.0 * merged["count"] / merged["group_n_valid"]).round(2)
        return merged.sort_values(["group_n_valid", "sentiment_id"], ascending=[False, True])

