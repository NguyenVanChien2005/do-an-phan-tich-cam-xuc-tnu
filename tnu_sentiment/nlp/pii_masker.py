"""Che thông tin cá nhân (PII) thống nhất và hàm băm ẩn danh một chiều."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import List, Tuple

# 1. Email chuẩn
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    flags=re.IGNORECASE,
)

# 2. Mã sinh viên ĐHTN / mã sinh viên phổ biến (ví dụ: DTC235200079, DTK22510012, SV20230012...)
# hoặc khai báo tường minh "MSSV: ..."
_STUDENT_ID_EXPLICIT_RE = re.compile(
    r"(?P<prefix>\b(?:mssv|mã\s*sv|mã\s*số\s*sinh\s*viên)\s*[:=\-]?\s*)(?P<id>[A-Za-z0-9]{6,15})\b",
    flags=re.IGNORECASE,
)
_STUDENT_ID_CODE_RE = re.compile(
    r"\b(?:DTC|DTK|DTE|DTS|DTN|DTY|DTZ|DTF|IST|TNU|SV|B\d{2}|K\d{2})[A-Za-z0-9]{0,3}\d{5,12}\b",
    flags=re.IGNORECASE,
)

# 3. Số điện thoại Việt Nam (03x, 05x, 07x, 08x, 09x hoặc +84...)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+84|84|0)(?:[\s.\-]?\d){9,10}(?!\w)"
)


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """Thay thế thông tin định danh cá nhân bằng các token chuẩn <EMAIL>, <STUDENT_ID>, <PHONE>.

    Thứ tự ưu tiên:
    1. <EMAIL> trước để không tách nhầm phần tên tài khoản chứa mã sinh viên trong email.
    2. <STUDENT_ID> cho mã sinh viên để không nhận nhầm dãy chữ số cuối là số điện thoại.
    3. <PHONE> cho số điện thoại di động/cố định Việt Nam.
    """
    if not isinstance(text, str) or not text:
        return "", []

    detected: List[str] = []
    masked = text

    if _EMAIL_RE.search(masked):
        masked = _EMAIL_RE.sub("<EMAIL>", masked)
        detected.append("EMAIL")

    student_hit = False
    if _STUDENT_ID_EXPLICIT_RE.search(masked):
        masked = _STUDENT_ID_EXPLICIT_RE.sub(r"\g<prefix><STUDENT_ID>", masked)
        student_hit = True
    if _STUDENT_ID_CODE_RE.search(masked):
        masked = _STUDENT_ID_CODE_RE.sub("<STUDENT_ID>", masked)
        student_hit = True
    if student_hit:
        detected.append("STUDENT_ID")

    if _PHONE_RE.search(masked):
        masked = _PHONE_RE.sub("<PHONE>", masked)
        detected.append("PHONE")

    return masked, detected


def compute_content_hash(masked_text: str) -> str:
    """Tính SHA-256 trên nội dung đã che PII và chuẩn hóa khoảng trắng để dò ứng viên trùng."""
    canonical = re.sub(r"\s+", " ", (masked_text or "").strip().lower())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_author_hash(raw_identifier: str, secret_salt: str) -> str:
    """Băm HMAC-SHA256 một chiều phục vụ lưu trữ nội bộ (KHÔNG hiển thị trên UI/Export)."""
    key = (secret_salt or "default-local-salt").encode("utf-8")
    msg = (raw_identifier or "anonymous").strip().encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

