"""Repository đọc artifact kết quả đánh giá mô hình thực nghiệm và fixture demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tnu_sentiment.domain.models import BenchmarkRun, ClassMetric, ErrorCaseSample
from tnu_sentiment.nlp.pii_masker import mask_pii
from tnu_sentiment.repositories.base import EvaluationRepository


class FileEvaluationRepository(EvaluationRepository):
    """Đọc kết quả đánh giá từ thư mục artifacts/evaluation/*.json (chế độ thật)
    hoặc kết hợp fixture minh họa khi ở chế độ Demo.
    Không hard-code điểm số vào mã nguồn giao diện.
    """

    def __init__(self, artifacts_dir: Path, demo_fixture_file: Path) -> None:
        self._artifacts_dir = artifacts_dir
        self._demo_fixture_file = demo_fixture_file

    @staticmethod
    def _parse_run_dict(raw: Dict[str, Any], forced_source: str | None = None) -> BenchmarkRun:
        per_class = [
            ClassMetric(
                class_id=str(c["class_id"]),
                class_name=str(c["class_name"]),
                precision=float(c["precision"]),
                recall=float(c["recall"]),
                f1=float(c["f1"]),
                support=int(c["support"]),
            )
            for c in (raw.get("per_class_metrics") or [])
        ]
        err_cases: List[ErrorCaseSample] = []
        for e in raw.get("error_cases") or []:
            safe_txt, _ = mask_pii(str(e.get("masked_text", "")))
            err_cases.append(
                ErrorCaseSample(
                    sample_id=str(e.get("sample_id", "")),
                    masked_text=safe_txt,
                    true_label=str(e.get("true_label", "")),
                    predicted_label=str(e.get("predicted_label", "")),
                    confidence=(
                        float(e["confidence"]) if e.get("confidence") is not None else None
                    ),
                    error_category=str(e.get("error_category", "")),
                    analysis_note=str(e.get("analysis_note", "")),
                )
            )

        return BenchmarkRun(
            run_id=str(raw["run_id"]),
            task=str(raw["task"]).upper(),
            model_name=str(raw["model_name"]),
            checkpoint_version=str(raw["checkpoint_version"]),
            dataset_version=str(raw["dataset_version"]),
            split=str(raw["split"]),
            evaluation_set_id=str(raw["evaluation_set_id"]),
            metric_definition_version=str(
                raw.get("metric_definition_version", "sklearn-multiclass-v1")
            ),
            n_samples=int(raw["n_samples"]),
            evaluated_at=str(raw["evaluated_at"]),
            result_source=forced_source or str(raw.get("result_source", "ARTIFACT_VERIFIED")),
            accuracy=float(raw["accuracy"]),
            macro_precision=float(raw["macro_precision"]),
            macro_recall=float(raw["macro_recall"]),
            macro_f1=float(raw["macro_f1"]),
            weighted_f1=float(raw["weighted_f1"]),
            per_class_metrics=per_class,
            confusion_matrix_labels=[str(lbl) for lbl in raw["confusion_matrix_labels"]],
            confusion_matrix=[[int(v) for v in row] for row in raw["confusion_matrix"]],
            error_cases=err_cases,
            notes=raw.get("notes"),
        )

    def load_runs(self, include_demo_fixtures: bool = False) -> List[BenchmarkRun]:
        runs: List[BenchmarkRun] = []

        if self._artifacts_dir.exists() and self._artifacts_dir.is_dir():
            for fp in sorted(self._artifacts_dir.glob("*.json")):
                try:
                    with fp.open("r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    items = payload if isinstance(payload, list) else [payload]
                    for item in items:
                        runs.append(self._parse_run_dict(item))
                except Exception:
                    continue

        if include_demo_fixtures and self._demo_fixture_file.exists():
            try:
                with self._demo_fixture_file.open("r", encoding="utf-8") as fh:
                    demo_items = json.load(fh)
                items = demo_items if isinstance(demo_items, list) else [demo_items]
                existing_ids = {r.run_id for r in runs}
                for item in items:
                    if str(item.get("run_id")) not in existing_ids:
                        runs.append(
                            self._parse_run_dict(item, forced_source="DEMO_FIXTURE")
                        )
            except Exception:
                pass

        return runs

