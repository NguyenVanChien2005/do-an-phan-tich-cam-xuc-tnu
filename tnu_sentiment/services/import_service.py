"""Dịch vụ Import lô CSV/XLSX an toàn, kiểm tra dòng, che PII và tổng kết không chồng lặp."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import openpyxl
import pandas as pd

from tnu_sentiment.config import AppConfig
from tnu_sentiment.constants import (
    ProcessingStatus,
    ReviewStatus,
    TopicAssignmentSource,
)
from tnu_sentiment.domain.models import (
    ImportExecutionSummary,
    ImportRowPreview,
)
from tnu_sentiment.models.sentiment_adapter import BaseSentimentAdapter
from tnu_sentiment.nlp.pii_masker import compute_author_hash, compute_content_hash, mask_pii
from tnu_sentiment.nlp.preprocessor import VietnamesePreprocessor
from tnu_sentiment.repositories.base import FeedbackRepository, NewFeedbackPayload

REQUIRED_IMPORT_COLUMNS = ("sentence", "unit_code", "source_name", "feedback_date")
OPTIONAL_IMPORT_COLUMNS = ("source_record_id", "topic_code")

_SUPPORTED_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
)


def sanitize_csv_cell(value: object) -> str:
    """Ngăn chặn CSV Formula Injection khi xuất tệp lỗi / tệp kết quả."""
    txt = "" if value is None else str(value)
    if txt and txt[0] in ("=", "+", "-", "@"):
        return "'" + txt
    return txt


def parse_feedback_datetime(raw_val: object) -> Optional[datetime]:
    if raw_val is None:
        return None
    if isinstance(raw_val, datetime):
        return raw_val.replace(tzinfo=None)
    s = str(raw_val).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return None
    for fmt in _SUPPORTED_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None)
    except ValueError:
        return None


class ImportService:
    """Thực hiện kiểm tra tệp CSV/XLSX (không chạy công thức/macro), xem trước đã che PII và thực thi import."""

    def __init__(
        self,
        config: AppConfig,
        sentiment_adapter: BaseSentimentAdapter,
        preprocessor: VietnamesePreprocessor,
    ) -> None:
        self._cfg = config
        self._sentiment = sentiment_adapter
        self._preprocessor = preprocessor

    @staticmethod
    def build_template_csv_bytes() -> bytes:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["sentence", "unit_code", "source_name", "feedback_date", "source_record_id", "topic_code"]
        )
        writer.writerow(
            [
                "Giảng viên hướng dẫn thực hành rất tận tình, dễ hiểu.",
                "ICTU",
                "SURVEY_PORTAL",
                "2026-10-15 08:30:00",
                "SAMPLE-REC-001",
                "LECTURER",
            ]
        )
        writer.writerow(
            [
                "Phòng máy tầng 3 có hai máy tính bị hỏng chuột cần bảo trì.",
                "TNUT",
                "QA_BOX",
                "2026-10-16 14:15:00",
                "SAMPLE-REC-002",
                "FACILITY",
            ]
        )
        return buf.getvalue().encode("utf-8-sig")

    def read_tabular_bytes(
        self, filename: str, file_bytes: bytes
    ) -> Tuple[List[str], List[Dict[str, object]]]:
        """Đọc an toàn tệp CSV hoặc XLSX:
        - Kiểm tra phần mở rộng, dung lượng và số dòng tối đa.
        - Với XLSX: chỉ đọc dữ liệu thuần (read_only=True, data_only=True, keep_vba=False), từ chối ô công thức.
        """
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext not in ("csv", "xlsx"):
            raise ValueError(
                f"Định dạng tệp '.{ext}' không được hỗ trợ. Chỉ chấp nhận tệp .csv hoặc .xlsx."
            )

        max_bytes = self._cfg.import_max_file_mb * 1024 * 1024
        if len(file_bytes) == 0:
            raise ValueError("Tệp tải lên rỗng (0 byte).")
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"Kích thước tệp vượt quá giới hạn cấu hình ({self._cfg.import_max_file_mb} MB)."
            )

        if ext == "csv":
            decoded: Optional[str] = None
            for enc in ("utf-8-sig", "utf-8", "cp1258"):
                try:
                    decoded = file_bytes.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if decoded is None:
                raise ValueError("Không thể giải mã bảng mã ký tự của tệp CSV (yêu cầu UTF-8).")

            reader = csv.DictReader(io.StringIO(decoded))
            if not reader.fieldnames:
                raise ValueError("Tệp CSV không có dòng tiêu đề cột (header).")
            headers = [str(h).strip() for h in reader.fieldnames if h is not None]
            rows: List[Dict[str, object]] = []
            for r in reader:
                rows.append({str(k).strip(): v for k, v in r.items() if k is not None})
                if len(rows) > self._cfg.import_max_rows:
                    raise ValueError(
                        f"Số dòng dữ liệu vượt quá giới hạn cấu hình ({self._cfg.import_max_rows} dòng)."
                    )
            return headers, rows

        # Đọc XLSX an toàn: không thực thi macro/VBA, không tính toán công thức
        wb_formula_check = openpyxl.load_workbook(
            io.BytesIO(file_bytes), read_only=True, data_only=False
        )
        ws_f = wb_formula_check.active
        wb_val = openpyxl.load_workbook(
            io.BytesIO(file_bytes), read_only=True, data_only=True
        )
        ws_v = wb_val.active
        if ws_v is None or ws_f is None:
            wb_formula_check.close()
            wb_val.close()
            raise ValueError("Tệp XLSX không có trang tính hợp lệ.")

        try:
            iter_f = ws_f.iter_rows(values_only=True)
            iter_v = ws_v.iter_rows(values_only=True)
            header_tuple = next(iter_v, None)
            _ = next(iter_f, None)
            if not header_tuple:
                raise ValueError("Tệp XLSX trống, thiếu dòng tiêu đề cột.")

            headers = [str(h).strip() if h is not None else "" for h in header_tuple]
            rows = []
            for row_f, row_v in zip(iter_f, iter_v):
                if all(cell is None or str(cell).strip() == "" for cell in row_v) and all(
                    cell is None or str(cell).strip() == "" for cell in row_f
                ):
                    continue
                row_dict: Dict[str, object] = {}
                for idx, col_name in enumerate(headers):
                    if not col_name:
                        continue
                    raw_f = row_f[idx] if idx < len(row_f) else None
                    val_v = row_v[idx] if idx < len(row_v) else None
                    if isinstance(raw_f, str) and raw_f.strip().startswith("="):
                        row_dict[col_name] = "__FORMULA_BLOCKED__"
                    else:
                        row_dict[col_name] = val_v
                rows.append(row_dict)
                if len(rows) > self._cfg.import_max_rows:
                    raise ValueError(
                        f"Số dòng dữ liệu vượt quá giới hạn cấu hình ({self._cfg.import_max_rows} dòng)."
                    )
            return [h for h in headers if h], rows
        finally:
            wb_formula_check.close()
            wb_val.close()

    def validate_and_preview(
        self,
        headers: List[str],
        rows: List[Dict[str, object]],
        repository: FeedbackRepository,
    ) -> Tuple[List[str], List[ImportRowPreview]]:
        """Kiểm tra cấu trúc cột, dữ liệu từng dòng và che PII ngay trên dữ liệu xem trước."""
        header_set = {h.strip() for h in headers}
        missing_cols = [c for c in REQUIRED_IMPORT_COLUMNS if c not in header_set]
        if missing_cols:
            return (
                [
                    f"Thiếu cột bắt buộc: {', '.join(missing_cols)}. "
                    f"Yêu cầu đủ các cột: {', '.join(REQUIRED_IMPORT_COLUMNS)}."
                ],
                [],
            )

        if not rows:
            return (["Tệp không có dòng dữ liệu nào để import."], [])

        unit_map = {u.code.upper(): u.code for u in repository.list_units()}
        source_by_code = {s.code.upper(): s.code for s in repository.list_sources()}
        source_by_name = {s.name.strip().lower(): s.code for s in repository.list_sources()}
        topic_map = {t.code.upper(): t.code for t in repository.list_topics()}

        previews: List[ImportRowPreview] = []
        for idx, r in enumerate(rows, start=1):
            errs: List[str] = []

            for k, v in r.items():
                if v == "__FORMULA_BLOCKED__":
                    errs.append(f"Cột '{k}' chứa công thức Excel không được phép thực thi.")

            raw_sent = "" if r.get("sentence") is None else str(r.get("sentence")).strip()
            masked_sent, _ = mask_pii(raw_sent)
            if not raw_sent or raw_sent == "__FORMULA_BLOCKED__":
                errs.append("Cột 'sentence' rỗng.")
            elif len(raw_sent) > self._cfg.max_input_chars:
                errs.append(
                    f"Độ dài câu ({len(raw_sent)}) vượt quá tối đa {self._cfg.max_input_chars} ký tự."
                )

            raw_unit = "" if r.get("unit_code") is None else str(r.get("unit_code")).strip().upper()
            if not raw_unit:
                resolved_unit = "UNSPECIFIED"
            elif raw_unit in unit_map:
                resolved_unit = unit_map[raw_unit]
            else:
                resolved_unit = raw_unit
                errs.append(
                    f"Mã đơn vị '{raw_unit}' không nằm trong danh mục hiện có của hệ thống."
                )

            raw_src = "" if r.get("source_name") is None else str(r.get("source_name")).strip()
            if not raw_src:
                resolved_src = "UNSPECIFIED"
            elif raw_src.upper() in source_by_code:
                resolved_src = source_by_code[raw_src.upper()]
            elif raw_src.lower() in source_by_name:
                resolved_src = source_by_name[raw_src.lower()]
            else:
                resolved_src = raw_src
                errs.append(
                    f"Nguồn '{raw_src}' không khớp mã hoặc tên nguồn trong danh mục hệ thống."
                )

            fb_dt = parse_feedback_datetime(r.get("feedback_date"))
            if fb_dt is None:
                errs.append(
                    f"Giá trị ngày tháng '{r.get('feedback_date')}' sai định dạng (hỗ trợ YYYY-MM-DD hoặc DD/MM/YYYY)."
                )

            raw_srid = r.get("source_record_id")
            srid = (
                str(raw_srid).strip()
                if raw_srid is not None and str(raw_srid).strip()
                else None
            )

            raw_topic = (
                str(r.get("topic_code")).strip().upper()
                if r.get("topic_code") is not None and str(r.get("topic_code")).strip()
                else "UNCLASSIFIED"
            )
            if raw_topic not in topic_map:
                errs.append(f"Mã chủ đề '{raw_topic}' không hợp lệ.")
                resolved_topic = "UNCLASSIFIED"
            else:
                resolved_topic = topic_map[raw_topic]

            previews.append(
                ImportRowPreview(
                    row_number=idx,
                    masked_sentence=masked_sent,
                    unit_code=resolved_unit,
                    source_code=resolved_src,
                    feedback_date=fb_dt,
                    source_record_id=srid,
                    topic_code=resolved_topic,
                    is_valid=(len(errs) == 0),
                    errors=errs,
                )
            )

        return [], previews

    def execute_import(
        self,
        filename: str,
        declared_origin: str,
        origin_note: str,
        previews: List[ImportRowPreview],
        repository: FeedbackRepository,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ImportExecutionSummary:
        """Thực thi import từng bản ghi hợp lệ và phân loại kết quả không chồng lặp,
        đảm bảo tổng 6 nhóm kết quả bằng chính xác tổng số dòng đã đọc.
        """
        batch_id = f"batch-{uuid.uuid4().hex[:12]}"
        total_rows = len(previews)

        saved_count = 0
        duplicate_skipped_count = 0
        validation_error_count = 0
        inference_error_count = 0
        db_error_count = 0
        row_errors: List[Dict[str, object]] = []

        seen_source_keys_in_file: Set[Tuple[str, str]] = set()
        processed_so_far = 0

        try:
            for item in previews:
                processed_so_far += 1
                if progress_callback is not None:
                    progress_callback(processed_so_far, total_rows)

                # Nhóm 1: Lỗi đầu vào (validation error)
                if not item.is_valid or item.feedback_date is None:
                    validation_error_count += 1
                    row_errors.append(
                        {
                            "row_number": item.row_number,
                            "category": "VALIDATION_ERROR",
                            "masked_sentence": sanitize_csv_cell(item.masked_sentence),
                            "detail": "; ".join(item.errors),
                        }
                    )
                    continue

                # Nhóm 2: Trùng bản ghi nguồn (source_record_id) trong file hoặc trong Repository
                if item.source_record_id:
                    s_key = (item.source_code, item.source_record_id)
                    if s_key in seen_source_keys_in_file or repository.check_source_record_exists(
                        item.source_code, item.source_record_id
                    ):
                        duplicate_skipped_count += 1
                        row_errors.append(
                            {
                                "row_number": item.row_number,
                                "category": "DUPLICATE_SKIPPED",
                                "masked_sentence": sanitize_csv_cell(item.masked_sentence),
                                "detail": f"Bỏ qua do trùng khóa bản ghi nguồn ({item.source_code}, {item.source_record_id}).",
                            }
                        )
                        continue
                    seen_source_keys_in_file.add(s_key)

                # Nhóm 3: Tiền xử lý & Suy luận mô hình
                if not self._sentiment.is_ready:
                    inference_error_count += 1
                    row_errors.append(
                        {
                            "row_number": item.row_number,
                            "category": "INFERENCE_ERROR",
                            "masked_sentence": sanitize_csv_cell(item.masked_sentence),
                            "detail": "Mô hình cảm xúc chưa sẵn sàng, không thể suy luận cho lô import.",
                        }
                    )
                    continue

                try:
                    prep = self._preprocessor.run(item.masked_sentence)
                    pred = self._sentiment.predict(prep.processed_content)
                except Exception as inf_exc:
                    inference_error_count += 1
                    safe_err, _ = mask_pii(str(inf_exc))
                    row_errors.append(
                        {
                            "row_number": item.row_number,
                            "category": "INFERENCE_ERROR",
                            "masked_sentence": sanitize_csv_cell(item.masked_sentence),
                            "detail": f"Lỗi tiền xử lý/suy luận: {safe_err}",
                        }
                    )
                    continue

                rev_status = ReviewStatus.UNREVIEWED.value
                rev_reason: Optional[str] = None
                if (
                    pred.confidence is not None
                    and pred.confidence < self._cfg.confidence_review_threshold
                ):
                    rev_status = ReviewStatus.NEEDS_REVIEW.value
                    rev_reason = (
                        f"Confidence ({pred.confidence:.2%}) < ngưỡng thử nghiệm "
                        f"{self._cfg.confidence_review_threshold:.0%}"
                    )
                elif prep.is_multi_aspect:
                    rev_status = ReviewStatus.NEEDS_REVIEW.value
                    rev_reason = "Câu nhiều ý tương phản khi import"

                topic_src = (
                    TopicAssignmentSource.IMPORT.value
                    if item.topic_code != "UNCLASSIFIED"
                    else TopicAssignmentSource.MANUAL.value
                )

                payload = NewFeedbackPayload(
                    data_origin=declared_origin,
                    origin_note=origin_note,
                    unit_code=item.unit_code,
                    source_code=item.source_code,
                    masked_content=prep.masked_content,
                    processed_content=prep.processed_content,
                    pii_types_detected=(
                        ",".join(prep.pii_types_detected)
                        if prep.pii_types_detected
                        else None
                    ),
                    was_truncated=pred.was_truncated,
                    is_multi_aspect=prep.is_multi_aspect,
                    processing_status=ProcessingStatus.DONE.value,
                    processing_error=None,
                    predicted_sentiment_label=pred.label_id,
                    predicted_sentiment_text=pred.label_code,
                    confidence_score=pred.confidence,
                    prob_negative=pred.probabilities.get(0),
                    prob_neutral=pred.probabilities.get(1),
                    prob_positive=pred.probabilities.get(2),
                    predicted_topic_code=item.topic_code,
                    topic_assignment_source=topic_src,
                    model_version=pred.model_version,
                    pipeline_version=prep.pipeline_version,
                    inference_time_ms=pred.inference_time_ms,
                    review_status=rev_status,
                    review_reason=rev_reason,
                    feedback_date=item.feedback_date,
                    idempotency_key=f"{batch_id}-row-{item.row_number}",
                    source_record_id=item.source_record_id,
                    content_hash=compute_content_hash(prep.masked_content),
                    author_hash=compute_author_hash(
                        f"{item.source_code}:{item.unit_code}", self._cfg.secret_salt
                    ),
                    batch_id=batch_id,
                )

                # Nhóm 4 & 5: Ghi vào Repository
                try:
                    save_res = repository.save_feedback(payload)
                    if save_res.saved:
                        saved_count += 1
                    elif save_res.was_source_duplicate or save_res.was_idempotent_duplicate:
                        duplicate_skipped_count += 1
                        row_errors.append(
                            {
                                "row_number": item.row_number,
                                "category": "DUPLICATE_SKIPPED",
                                "masked_sentence": sanitize_csv_cell(item.masked_sentence),
                                "detail": save_res.message,
                            }
                        )
                    else:
                        db_error_count += 1
                        row_errors.append(
                            {
                                "row_number": item.row_number,
                                "category": "DB_ERROR",
                                "masked_sentence": sanitize_csv_cell(item.masked_sentence),
                                "detail": save_res.message,
                            }
                        )
                except Exception as db_exc:
                    db_error_count += 1
                    safe_db_err, _ = mask_pii(str(db_exc))
                    row_errors.append(
                        {
                            "row_number": item.row_number,
                            "category": "DB_ERROR",
                            "masked_sentence": sanitize_csv_cell(item.masked_sentence),
                            "detail": f"Lỗi ghi dữ liệu: {safe_db_err}",
                        }
                    )
        except Exception as fatal_exc:
            safe_fatal, _ = mask_pii(str(fatal_exc))
            row_errors.append(
                {
                    "row_number": processed_so_far,
                    "category": "INTERRUPTED",
                    "masked_sentence": "",
                    "detail": f"Tiến trình bị gián đoạn: {safe_fatal}",
                }
            )

        accounted = (
            saved_count
            + duplicate_skipped_count
            + validation_error_count
            + inference_error_count
            + db_error_count
        )
        unprocessed_count = max(0, total_rows - accounted)

        return ImportExecutionSummary(
            batch_id=batch_id,
            filename=filename,
            declared_origin=declared_origin,
            total_rows_read=total_rows,
            saved_count=saved_count,
            duplicate_skipped_count=duplicate_skipped_count,
            validation_error_count=validation_error_count,
            inference_error_count=inference_error_count,
            db_error_count=db_error_count,
            unprocessed_count=unprocessed_count,
            row_errors=row_errors,
        )

    @staticmethod
    def export_error_report_csv(summary: ImportExecutionSummary) -> bytes:
        if not summary.row_errors:
            return b""
        df = pd.DataFrame(summary.row_errors)
        return df.to_csv(index=False).encode("utf-8-sig")

