"""Kiểm thử Import CSV/XLSX: sai cột, sai ngày, chặn công thức Excel, trùng bản ghi nguồn, lỗi DB và tổng đối soát."""

from __future__ import annotations

import io
from pathlib import Path

import openpyxl

from tnu_sentiment.config import load_app_config
from tnu_sentiment.domain.models import ComponentHealth, SaveRecordResult, SentimentPrediction
from tnu_sentiment.models.sentiment_adapter import BaseSentimentAdapter
from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor
from tnu_sentiment.repositories.base import NewFeedbackPayload
from tnu_sentiment.repositories.demo_repo import DemoFeedbackRepository
from tnu_sentiment.services.import_service import ImportService


class _StubAdapter(BaseSentimentAdapter):
    @property
    def is_ready(self) -> bool:
        return True

    @property
    def model_version(self) -> str:
        return "stub-v1"

    def health(self) -> ComponentHealth:
        return ComponentHealth(True, "Sẵn sàng", "stub")

    def predict(self, processed_text: str) -> SentimentPrediction:
        return SentimentPrediction(
            label_id=0,
            label_code="Negative",
            label_vi="Tiêu cực",
            confidence=0.88,
            probabilities={0: 0.88, 1: 0.06, 2: 0.06},
            model_version="stub-v1",
            was_truncated=False,
            token_count=8,
            max_length=256,
            inference_time_ms=5.0,
            measurement_scope="stub",
        )


class _FailingDBRepo(DemoFeedbackRepository):
    def save_feedback(self, payload: NewFeedbackPayload) -> SaveRecordResult:
        if "gây lỗi db" in payload.masked_content.lower():
            raise RuntimeError("Simulated MySQL write failure")
        return super().save_feedback(payload)


def test_import_missing_columns_reported_clearly(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = DemoFeedbackRepository(tmp_path / "empty.json")
    svc = ImportService(cfg, _StubAdapter(), VietnamesePreprocessor())

    bad_csv = "noi_dung,don_vi\nXin chào,ICTU\n".encode("utf-8")
    headers, rows = svc.read_tabular_bytes("bad.csv", bad_csv)
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert len(schema_errs) == 1
    assert "Thiếu cột bắt buộc" in schema_errs[0]
    assert previews == []


def test_import_summary_counts_are_disjoint_and_sum_to_total_rows(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = _FailingDBRepo(tmp_path / "empty.json")
    svc = ImportService(cfg, _StubAdapter(), VietnamesePreprocessor())

    # 5 dòng kiểm thử đủ các nhánh:
    # 1. Hợp lệ + chứa PII -> che PII & lưu thành công (saved=1)
    # 2. Trùng source_record_id với dòng 1 -> bỏ qua do trùng (duplicate_skipped=1)
    # 3. Sai ngày tháng -> lỗi đầu vào (validation_error=1)
    # 4. Sai mã đơn vị -> lỗi đầu vào (validation_error=2)
    # 5. Lỗi ghi DB mô phỏng -> lỗi ghi dữ liệu (db_error=1)
    csv_content = (
        "sentence,unit_code,source_name,feedback_date,source_record_id\n"
        "\"Em sđt 0987654321 phản ánh máy chiếu hỏng.\",ICTU,QA_BOX,2026-10-20 08:00:00,SRC-100\n"
        "\"Bản ghi trùng mã nguồn.\",ICTU,QA_BOX,2026-10-20 09:00:00,SRC-100\n"
        "\"Phản hồi sai ngày tháng.\",TNUT,QA_BOX,2026-99-99,SRC-101\n"
        "\"Phản hồi sai mã đơn vị.\",DON_VI_LA,QA_BOX,2026-10-21,SRC-102\n"
        "\"Phản hồi gây lỗi db khi ghi.\",ICTU,QA_BOX,2026-10-22,SRC-103\n"
    ).encode("utf-8")

    headers, rows = svc.read_tabular_bytes("batch.csv", csv_content)
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert schema_errs == []
    assert len(previews) == 5
    # PII đã được che ngay từ bước preview!
    assert "0987654321" not in previews[0].masked_sentence
    assert "<PHONE>" in previews[0].masked_sentence

    summary = svc.execute_import(
        filename="batch.csv",
        declared_origin="DEMO",
        origin_note="Test import",
        previews=previews,
        repository=repo,
    )
    assert summary.total_rows_read == 5
    assert summary.saved_count == 1
    assert summary.duplicate_skipped_count == 1
    assert summary.validation_error_count == 2
    assert summary.inference_error_count == 0
    assert summary.db_error_count == 1
    assert summary.unprocessed_count == 0
    assert summary.accounted_rows == summary.total_rows_read


def test_xlsx_formula_cells_are_blocked(tmp_path: Path) -> None:
    cfg = load_app_config()
    repo = DemoFeedbackRepository(tmp_path / "empty.json")
    svc = ImportService(cfg, _StubAdapter(), VietnamesePreprocessor())

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["sentence", "unit_code", "source_name", "feedback_date"])
    ws.append(["=HYPERLINK(\"http://evil.example\",\"click\")", "ICTU", "QA_BOX", "2026-10-20"])
    buf = io.BytesIO()
    wb.save(buf)
    wb.close()

    headers, rows = svc.read_tabular_bytes("formula.xlsx", buf.getvalue())
    schema_errs, previews = svc.validate_and_preview(headers, rows, repo)
    assert schema_errs == []
    assert len(previews) == 1
    assert previews[0].is_valid is False
    assert any("công thức" in e.lower() for e in previews[0].errors)

