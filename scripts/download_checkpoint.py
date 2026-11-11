"""Script tải trọng số PhoBERT-v2 (model.safetensors) từ HuggingFace Hub / Google Drive / URL và xác thực SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = ROOT_DIR / "models" / "checkpoint_manifest.json"


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, Any]:
    """Đọc tệp cấu hình manifest chứa SHA-256 và thông tin phiên bản môi trường."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp manifest tại: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_keys = {"sha256", "transformers", "pyvi", "preprocessor_version"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Manifest thiếu các trường bắt buộc: {sorted(missing)}")
    return data


def compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Tính mã băm SHA-256 của tệp theo luồng (streaming chunks) để tiết kiệm RAM."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checkpoint_sha256(
    weights_path: Path,
    expected_sha256: str,
    raise_on_mismatch: bool = True,
) -> bool:
    """Kiểm tra tính toàn vẹn của tệp model.safetensors bằng SHA-256."""
    if not weights_path.exists():
        if raise_on_mismatch:
            raise FileNotFoundError(f"Chưa có tệp trọng số tại: {weights_path}")
        return False
    actual = compute_file_sha256(weights_path).lower()
    expected = expected_sha256.strip().lower()
    if actual != expected:
        if raise_on_mismatch:
            raise ValueError(
                f"Sai lệch mã băm SHA-256 của '{weights_path.name}': "
                f"kỳ vọng {expected}, thực tế {actual}."
            )
        return False
    return True


def _normalize_gdrive_download_url(gdrive_url: str) -> str:
    """Chuyển liên kết chia sẻ Google Drive thành liên kết tải trực tiếp."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", gdrive_url)
    if not match:
        match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", gdrive_url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    return gdrive_url


def _download_from_url(url: str, destination: Path) -> None:
    """Tải tệp từ URL HTTP/HTTPS hoặc đường dẫn file:// về đích."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        if url.startswith("file://"):
            local_src = Path(urllib.request.url2pathname(url[7:]))
            shutil.copy2(local_src, tmp_path)
        elif Path(url).exists():
            shutil.copy2(Path(url), tmp_path)
        else:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TNU-Sentiment-Checkpoint-Downloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp, tmp_path.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=1024 * 1024)
        tmp_path.replace(destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _download_from_hf_hub(repo_id: str, filename: str, destination: Path) -> None:
    """Tải tệp từ HuggingFace Hub bằng huggingface_hub hoặc URL chuẩn."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(cached, destination)
    except ImportError:
        hf_url = f"https://huggingface.co/{repo_id.strip('/')}/resolve/main/{filename}"
        _download_from_url(hf_url, destination)


def download_checkpoint(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    target_dir: Optional[Path] = None,
    url: Optional[str] = None,
    hf_repo: Optional[str] = None,
    gdrive_url: Optional[str] = None,
    verify_only: bool = False,
    force: bool = False,
) -> Path:
    """Tải và xác thực SHA-256 cho trọng số mô hình PhoBERT-v2."""
    manifest = load_manifest(manifest_path)
    rel_ckpt_dir = manifest.get(
        "checkpoint_dir",
        "phobert_tnu_best_model-20261101T002214Z-1-001/phobert_tnu_best_model",
    )
    weights_name = manifest.get("weights_file", "model.safetensors")
    expected_sha256 = str(manifest["sha256"])

    resolved_dir = target_dir or (ROOT_DIR / rel_ckpt_dir)
    weights_path = resolved_dir / weights_name

    if verify_only:
        verify_checkpoint_sha256(weights_path, expected_sha256, raise_on_mismatch=True)
        return weights_path

    if weights_path.exists() and not force:
        if verify_checkpoint_sha256(weights_path, expected_sha256, raise_on_mismatch=False):
            return weights_path

    resolved_url = url or os.getenv("TNU_CHECKPOINT_URL", "").strip()
    resolved_gdrive = gdrive_url or os.getenv("TNU_CHECKPOINT_GDRIVE_URL", "").strip()
    resolved_hf = hf_repo or os.getenv("TNU_CHECKPOINT_HF_REPO", "").strip()

    if resolved_url:
        _download_from_url(resolved_url, weights_path)
    elif resolved_gdrive:
        direct_url = _normalize_gdrive_download_url(resolved_gdrive)
        _download_from_url(direct_url, weights_path)
    elif resolved_hf:
        _download_from_hf_hub(resolved_hf, weights_name, weights_path)
    else:
        raise RuntimeError(
            "Chưa cấu hình nguồn tải trọng số. Hãy thiết lập một trong các biến môi trường "
            "TNU_CHECKPOINT_URL, TNU_CHECKPOINT_GDRIVE_URL, hoặc TNU_CHECKPOINT_HF_REPO "
            "(hoặc truyền tham số --url / --gdrive-url / --hf-repo)."
        )

    try:
        verify_checkpoint_sha256(weights_path, expected_sha256, raise_on_mismatch=True)
    except ValueError:
        weights_path.unlink(missing_ok=True)
        raise

    return weights_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tải và kiểm tra mã băm SHA-256 cho trọng số PhoBERT-v2 (model.safetensors)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Đường dẫn tệp checkpoint_manifest.json",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Thư mục lưu model.safetensors (mặc định đọc từ manifest)",
    )
    parser.add_argument("--url", type=str, default=None, help="URL trực tiếp để tải model.safetensors")
    parser.add_argument("--hf-repo", type=str, default=None, help="HuggingFace Repo ID (vd: username/phobert-tnu)")
    parser.add_argument("--gdrive-url", type=str, default=None, help="Google Drive URL chia sẻ tệp")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Chỉ kiểm tra SHA-256 của tệp trọng số hiện có trên đĩa",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại kể cả khi tệp đã tồn tại",
    )
    args = parser.parse_args(argv)

    try:
        out_path = download_checkpoint(
            manifest_path=args.manifest,
            target_dir=args.target_dir,
            url=args.url,
            hf_repo=args.hf_repo,
            gdrive_url=args.gdrive_url,
            verify_only=args.verify_only,
            force=args.force,
        )
        print(f"[OK] Trọng số hợp lệ (SHA-256 khớp manifest): {out_path}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
