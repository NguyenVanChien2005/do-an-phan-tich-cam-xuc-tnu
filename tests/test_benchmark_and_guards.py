"""Kiểm thử tính tương thích đối sánh Benchmark và từ chối nạp base model thiếu classification head."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from tnu_sentiment.domain.models import BenchmarkRun
from tnu_sentiment.models.sentiment_adapter import validate_sentiment_checkpoint_dir
from tnu_sentiment.services.benchmark_service import BenchmarkService


def _make_run(
    run_id: str,
    task: str = "SENTIMENT",
    split: str = "Test",
    eval_set_id: str = "test-sha256-aaaa",
    n_samples: int = 3166,
    metric_ver: str = "sklearn-multiclass-v1",
) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=run_id,
        task=task,
        model_name=run_id,
        checkpoint_version="v1",
        dataset_version="ds-v1",
        split=split,
        evaluation_set_id=eval_set_id,
        metric_definition_version=metric_ver,
        n_samples=n_samples,
        evaluated_at="2026-11-06",
        result_source="ARTIFACT_VERIFIED",
        accuracy=0.9,
        macro_precision=0.88,
        macro_recall=0.87,
        macro_f1=0.875,
        weighted_f1=0.9,
        per_class_metrics=[],
        confusion_matrix_labels=["0", "1", "2"],
        confusion_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )


def test_benchmark_does_not_mix_incompatible_runs_even_with_equal_sample_count() -> None:
    run_a = _make_run("phobert-test", split="Test", eval_set_id="set-hash-1", n_samples=3166)
    run_b = _make_run("tfidf-test", split="Test", eval_set_id="set-hash-1", n_samples=3166)
    run_same_n_diff_set = _make_run(
        "other-test", split="Test", eval_set_id="set-hash-DIFFERENT", n_samples=3166
    )
    run_val = _make_run("phobert-val", split="Validation", eval_set_id="val-hash-1", n_samples=1709)
    run_topic = _make_run("topic-test", task="TOPIC", split="Test", eval_set_id="set-hash-1", n_samples=3166)

    ok_ab, _ = BenchmarkService.are_runs_comparable(run_a, run_b)
    assert ok_ab is True

    # Cùng n_samples=3166 nhưng khác evaluation_set_id -> TỪ CHỐI so sánh trực tiếp!
    ok_diff_set, reason_diff = BenchmarkService.are_runs_comparable(run_a, run_same_n_diff_set)
    assert ok_diff_set is False
    assert "evaluation_set_id" in reason_diff or "định danh" in reason_diff

    ok_val, _ = BenchmarkService.are_runs_comparable(run_a, run_val)
    assert ok_val is False

    ok_topic, _ = BenchmarkService.are_runs_comparable(run_a, run_topic)
    assert ok_topic is False


def test_checkpoint_validator_rejects_base_model_without_classification_head(tmp_path: Path) -> None:
    ckpt = tmp_path / "fake_base_phobert"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"dummy")
    (ckpt / "vocab.txt").write_text("<s>\n", encoding="utf-8")
    (ckpt / "bpe.codes").write_text("#version: 0.2\n", encoding="utf-8")
    # Base model RobertaModel chưa fine-tune SequenceClassification -> PHẢI TỪ CHỐI!
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["RobertaModel"],
                "id2label": {"0": "Negative", "1": "Neutral", "2": "Positive"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base model"):
        validate_sentiment_checkpoint_dir(ckpt)

