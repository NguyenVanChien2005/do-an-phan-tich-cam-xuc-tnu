"""Kiểm thử tính toán KPI, mẫu số N_valid, tập rỗng và chính sách nhãn."""

from __future__ import annotations

from datetime import datetime

from tnu_sentiment.config import load_app_config
from tnu_sentiment.constants import LabelPolicy
from tnu_sentiment.domain.models import FeedbackRecord
from tnu_sentiment.services.stats_service import StatsService


def _make_rec(
    fid: int,
    proc_status: str = "DONE",
    pred_lbl: int | None = 2,
    rev_status: str = "UNREVIEWED",
    rev_lbl: int | None = None,
    conf: float | None = 0.85,
    origin: str = "TNU_REAL",
) -> FeedbackRecord:
    now = datetime(2026, 10, 20, 9, 0, 0)
    return FeedbackRecord(
        feedback_id=fid,
        data_origin=origin,
        origin_note=None,
        unit_code="ICTU",
        unit_name="ICTU",
        source_code="SURVEY_PORTAL",
        source_name="Khảo sát",
        masked_content=f"Mẫu {fid}",
        processed_content=f"mẫu {fid}",
        pii_types_detected=None,
        was_truncated=False,
        is_multi_aspect=False,
        is_duplicate_candidate=False,
        processing_status=proc_status,
        processing_error=None,
        predicted_sentiment_label=pred_lbl,
        predicted_sentiment_text=None,
        confidence_score=conf,
        prob_negative=None,
        prob_neutral=None,
        prob_positive=None,
        predicted_topic_code="LECTURER",
        predicted_topic_name="Giảng viên",
        topic_assignment_source="MANUAL",
        model_version="PhoBERT-v1" if proc_status == "DONE" else None,
        pipeline_version="v1",
        inference_time_ms=20.0,
        review_status=rev_status,
        review_reason=None,
        reviewed_sentiment_label=rev_lbl,
        reviewed_topic_code=None,
        reviewed_topic_name=None,
        reviewed_by="tester" if rev_status == "VERIFIED" else None,
        reviewed_at=now if rev_status == "VERIFIED" else None,
        review_note=None,
        record_version=1,
        feedback_date=now,
        created_at=now,
        updated_at=now,
    )


def test_hand_calculated_kpis_and_pending_error_exclusion() -> None:
    cfg = load_app_config()
    svc = StatsService(cfg)

    # 6 bản ghi tổng:
    # - #1: DONE, pred=2 (Pos)
    # - #2: DONE, pred=2 (Pos)
    # - #3: DONE, pred=0 (Neg), VERIFIED thành 1 (Neu)! -> Với VERIFIED_FIRST tính là 1 (Neu)
    # - #4: DONE, pred=0 (Neg)
    # - #5: PENDING -> loại khỏi N_valid
    # - #6: ERROR -> loại khỏi N_valid
    records = [
        _make_rec(1, "DONE", 2),
        _make_rec(2, "DONE", 2),
        _make_rec(3, "DONE", 0, rev_status="VERIFIED", rev_lbl=1),
        _make_rec(4, "DONE", 0, rev_status="NEEDS_REVIEW"),
        _make_rec(5, "PENDING", None, conf=None),
        _make_rec(6, "ERROR", None, conf=None),
    ]

    kpi_first = svc.compute_kpis(records, LabelPolicy.VERIFIED_FIRST)
    assert kpi_first.n_total == 6
    assert kpi_first.n_done == 4
    assert kpi_first.n_pending_or_running == 1
    assert kpi_first.n_error == 1
    assert kpi_first.n_valid == 4  # Chỉ 4 bản ghi DONE hợp lệ
    assert kpi_first.n_pos == 2
    assert kpi_first.n_neu == 1
    assert kpi_first.n_neg == 1
    assert kpi_first.pos_ratio_pct == 50.0  # 2 / 4
    assert kpi_first.neg_ratio_pct == 25.0  # 1 / 4
    assert kpi_first.nss_score == 25.0      # 100 * (2 - 1) / 4

    # Khi đổi sang CHỈ NHÃN ĐÃ DUYỆT (VERIFIED_ONLY): chỉ bản ghi #3 có nhãn duyệt = 1 (Neu)
    kpi_vo = svc.compute_kpis(records, LabelPolicy.VERIFIED_ONLY)
    assert kpi_vo.n_total == 6
    assert kpi_vo.n_valid == 1
    assert kpi_vo.n_pos == 0
    assert kpi_vo.n_neu == 1
    assert kpi_vo.n_neg == 0
    assert kpi_vo.pos_ratio_pct == 0.0
    assert kpi_vo.nss_score == 0.0


def test_empty_or_no_valid_labels_never_fakes_zero_percent() -> None:
    cfg = load_app_config()
    svc = StatsService(cfg)

    kpi_empty = svc.compute_kpis([], LabelPolicy.VERIFIED_FIRST)
    assert kpi_empty.n_total == 0
    assert kpi_empty.n_valid == 0
    assert kpi_empty.pos_ratio_pct is None
    assert kpi_empty.neg_ratio_pct is None
    assert kpi_empty.nss_score is None

    only_pending = [_make_rec(1, "PENDING", None), _make_rec(2, "ERROR", None)]
    kpi_no_valid = svc.compute_kpis(only_pending, LabelPolicy.VERIFIED_FIRST)
    assert kpi_no_valid.n_total == 2
    assert kpi_no_valid.n_valid == 0
    assert kpi_no_valid.pos_ratio_pct is None
    assert kpi_no_valid.neg_ratio_pct is None
    assert kpi_no_valid.nss_score is None


def test_negative_alert_requires_minimum_30_valid_samples() -> None:
    cfg = load_app_config()
    svc = StatsService(cfg)

    # 10 mẫu đều tiêu cực (100% > 35%), nhưng N_valid = 10 < 30 -> KHÔNG báo động
    small_neg = [_make_rec(i, "DONE", 0) for i in range(1, 11)]
    assert svc.compute_kpis(small_neg, LabelPolicy.VERIFIED_FIRST).alert_triggered is False

    # 30 mẫu hợp lệ trong đó 12 mẫu tiêu cực (40% >= 35%) -> kích hoạt cảnh báo minh họa
    enough_neg = [_make_rec(i, "DONE", 0) for i in range(1, 13)] + [
        _make_rec(i, "DONE", 2) for i in range(13, 31)
    ]
    kpi_30 = svc.compute_kpis(enough_neg, LabelPolicy.VERIFIED_FIRST)
    assert kpi_30.alert_triggered is True
    assert "minh họa" in (kpi_30.alert_message or "")

