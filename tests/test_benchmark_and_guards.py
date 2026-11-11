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


def test_checkpoint_manifest_and_download_script_sha256_verification(tmp_path: Path) -> None:
    """Bước 2.2: Kiểm tra models/checkpoint_manifest.json và cơ chế kiểm tra SHA-256 của download_checkpoint.py."""
    import hashlib
    from scripts.download_checkpoint import (
        DEFAULT_MANIFEST_PATH,
        download_checkpoint,
        load_manifest,
        verify_checkpoint_sha256,
    )

    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    assert len(manifest["sha256"]) == 64
    assert "transformers" in manifest
    assert "underthesea" in manifest
    assert "pyvi" in manifest
    assert "preprocessor_version" in manifest

    # Tạo một file trọng số giả lập để kiểm thử tải và xác thực SHA-256
    fake_weights_bytes = b"fake-phobert-safetensors-payload-for-test"
    fake_sha256 = hashlib.sha256(fake_weights_bytes).hexdigest()
    source_file = tmp_path / "source_model.safetensors"
    source_file.write_bytes(fake_weights_bytes)

    custom_manifest_path = tmp_path / "checkpoint_manifest.json"
    custom_manifest_path.write_text(
        json.dumps(
            {
                "sha256": fake_sha256,
                "weights_file": "model.safetensors",
                "transformers": "5.22.1",
                "underthesea": "6.8.4",
                "pyvi": "0.1.1",
                "preprocessor_version": "tnu-preprocess-v1.1-pyvi",
            }
        ),
        encoding="utf-8",
    )

    target_dir = tmp_path / "downloaded_ckpt"
    out_path = download_checkpoint(
        manifest_path=custom_manifest_path,
        target_dir=target_dir,
        url=str(source_file),
    )
    assert out_path.exists()
    assert verify_checkpoint_sha256(out_path, fake_sha256) is True

    # Nếu SHA-256 không khớp -> phải báo lỗi ValueError và xóa file lỗi
    bad_source = tmp_path / "corrupted_model.safetensors"
    bad_source.write_bytes(b"corrupted-bytes")
    bad_target_dir = tmp_path / "bad_ckpt"
    with pytest.raises(ValueError, match="SHA-256"):
        download_checkpoint(
            manifest_path=custom_manifest_path,
            target_dir=bad_target_dir,
            url=str(bad_source),
            force=True,
        )
    assert not (bad_target_dir / "model.safetensors").exists()


@pytest.mark.requires_checkpoint
def test_real_checkpoint_matches_manifest_sha256() -> None:
    """Bước 2.1 & 2.2: Xác thực SHA-256 của model.safetensors thật khớp với models/checkpoint_manifest.json."""
    from scripts.download_checkpoint import DEFAULT_MANIFEST_PATH, ROOT_DIR, download_checkpoint, load_manifest

    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    weights_path = ROOT_DIR / manifest["checkpoint_dir"] / manifest["weights_file"]
    if not weights_path.exists():
        pytest.skip(
            "Thiếu file trọng số model.safetensors (515MB). Chạy python scripts/download_checkpoint.py để tải."
        )
    verified_path = download_checkpoint(verify_only=True)
    assert verified_path == weights_path
