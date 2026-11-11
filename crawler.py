"""Module thu thập bình luận cộng đồng (Facebook / YouTube / News) chuẩn hóa khớp 100% Hợp đồng Batch Import."""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import datetime, timezone
from getpass import getpass
from typing import Any, Dict, Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tnu_sentiment.nlp.pii_masker import compute_content_hash, mask_pii
from tnu_sentiment.services.import_service import (
    compute_import_idempotency_key,
    parse_feedback_datetime,
)

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

CSV_COLUMNS = [
    "sentence",
    "unit_code",
    "source_name",
    "feedback_date",
    "source_record_id",
    "source_url",
    "post_id",
    "parent_comment_id",
    "collected_at",
    "topic_code",
    "idempotency_key",
]


def create_session() -> requests.Session:
    """Dùng lại kết nối và thử lại khi gặp lỗi tạm thời."""
    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def compute_crawler_idempotency_key(
    source_name: str,
    source_record_id: Optional[str] = None,
    sentence: str = "",
    unit_code: str = "UNSPECIFIED",
    feedback_date: Optional[datetime | str] = None,
) -> str:
    """Tính idempotency_key xác định cho dữ liệu thu thập từ Crawler:
    - Ưu tiên dựa trên (source_name, source_record_id) nếu có mã bình luận gốc.
    - Nếu không có source_record_id, tính từ (content_hash, unit_code, feedback_date).
    """
    src_code = (source_name or "FACEBOOK").strip().upper()
    srid = str(source_record_id).strip() if source_record_id is not None else ""
    masked_sent, _ = mask_pii(str(sentence or ""))
    c_hash = compute_content_hash(masked_sent)
    fb_dt = parse_feedback_datetime(feedback_date) or datetime.now(VN_TZ).replace(tzinfo=None)
    return compute_import_idempotency_key(
        source_code=src_code,
        source_record_id=srid or None,
        content_hash=c_hash,
        unit_code=(unit_code or "UNSPECIFIED").strip().upper(),
        feedback_date=fb_dt,
    )


def normalize_crawled_comment(
    raw_item: Dict[str, Any],
    default_unit_code: str = "UNSPECIFIED",
    default_source_name: str = "FACEBOOK",
) -> Dict[str, Any]:
    """Chuẩn hóa 1 bản ghi bình luận thu thập được sang đúng hợp đồng cột Batch Import của dự án."""
    raw_sentence = str(
        raw_item.get("sentence")
        or raw_item.get("message")
        or raw_item.get("content")
        or ""
    ).strip()
    masked_sentence, _ = mask_pii(raw_sentence)

    unit_code = (
        str(raw_item.get("unit_code") or default_unit_code).strip().upper()
        or "UNSPECIFIED"
    )
    platform_or_src = str(
        raw_item.get("source_name")
        or raw_item.get("platform")
        or default_source_name
    ).strip().upper()
    if platform_or_src in ("FB", "FACEBOOK"):
        source_name = "FACEBOOK"
    elif platform_or_src in ("YT", "YOUTUBE"):
        source_name = "YOUTUBE"
    elif platform_or_src in ("NEWS", "NEWS_COMMENT"):
        source_name = "NEWS_COMMENT"
    else:
        source_name = platform_or_src or "FACEBOOK"

    raw_fb_dt = (
        raw_item.get("feedback_date")
        or raw_item.get("created_time")
        or raw_item.get("created_at")
    )
    fb_dt = parse_feedback_datetime(raw_fb_dt) or datetime.now(VN_TZ).replace(
        tzinfo=None, microsecond=0
    )
    fb_dt_str = fb_dt.strftime("%Y-%m-%d %H:%M:%S")

    raw_coll_dt = raw_item.get("collected_at")
    coll_dt = parse_feedback_datetime(raw_coll_dt) or datetime.now(VN_TZ).replace(
        tzinfo=None, microsecond=0
    )
    coll_dt_str = coll_dt.strftime("%Y-%m-%d %H:%M:%S")

    srid = str(
        raw_item.get("source_record_id")
        or raw_item.get("comment_id")
        or raw_item.get("id")
        or ""
    ).strip()
    source_url = str(
        raw_item.get("source_url") or raw_item.get("source") or ""
    ).strip()
    post_id = str(raw_item.get("post_id") or "").strip()
    parent_cid = str(
        raw_item.get("parent_comment_id")
        or (raw_item.get("parent", {}) or {}).get("id", "")
        if isinstance(raw_item.get("parent"), dict)
        else (raw_item.get("parent_comment_id") or "")
    ).strip()
    topic_code = (
        str(raw_item.get("topic_code") or "UNCLASSIFIED").strip().upper()
        or "UNCLASSIFIED"
    )

    idem_key = str(raw_item.get("idempotency_key") or "").strip()
    if not idem_key:
        idem_key = compute_crawler_idempotency_key(
            source_name=source_name,
            source_record_id=srid or None,
            sentence=masked_sentence,
            unit_code=unit_code,
            feedback_date=fb_dt,
        )

    return {
        "sentence": masked_sentence,
        "unit_code": unit_code,
        "source_name": source_name,
        "feedback_date": fb_dt_str,
        "source_record_id": srid,
        "source_url": source_url,
        "post_id": post_id,
        "parent_comment_id": parent_cid,
        "collected_at": coll_dt_str,
        "topic_code": topic_code,
        "idempotency_key": idem_key,
    }


def crawl_facebook_comments(
    session: requests.Session,
    post_id: str,
    post_url: str,
    access_token: str,
    api_version: str,
    max_comments: int = 1000,
    unit_code: str = "UNSPECIFIED",
    source_name: str = "FACEBOOK",
) -> Iterator[Dict[str, Any]]:
    """Lấy tối đa max_comments bình luận cấp đầu của một bài và xuất đúng hợp đồng Batch Import."""
    endpoint = f"https://graph.facebook.com/{api_version}/{post_id}/comments"

    headers = {"Authorization": f"Bearer {access_token}"}

    params: Dict[str, Any] = {
        "fields": "id,message,created_time,like_count,parent",
        "filter": "toplevel",
        "limit": 100,
    }

    seen_ids = set()
    seen_cursors = set()
    page_number = 1

    while True:
        print(f"Đang lấy trang bình luận {page_number}...")

        response = session.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=(10, 30),
        )

        payload = response.json()

        # Facebook thường trả chi tiết lỗi trong trường "error".
        if "error" in payload:
            error = payload["error"]
            raise RuntimeError(
                f"Facebook API báo lỗi {error.get('code')}, "
                f"mã phụ {error.get('error_subcode')}. "
                "Kiểm tra token, quyền đọc và Post ID trong Graph API Explorer."
            )

        response.raise_for_status()

        if not isinstance(payload.get("data"), list):
            raise RuntimeError("API trả dữ liệu sai định dạng mong đợi.")

        for comment in payload["data"]:
            comment_id = str(comment.get("id", "")).strip()

            if comment_id and comment_id in seen_ids:
                continue

            if comment_id:
                seen_ids.add(comment_id)

            raw_entry = {
                "sentence": comment.get("message", ""),
                "unit_code": unit_code,
                "source_name": source_name,
                "feedback_date": comment.get("created_time", ""),
                "source_record_id": comment_id,
                "source_url": post_url,
                "post_id": post_id,
                "parent_comment_id": (comment.get("parent") or {}).get("id", "")
                if isinstance(comment.get("parent"), dict)
                else "",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "topic_code": "UNCLASSIFIED",
            }
            yield normalize_crawled_comment(
                raw_entry,
                default_unit_code=unit_code,
                default_source_name=source_name,
            )

            if len(seen_ids) >= max_comments:
                print(f"Đã đạt giới hạn {max_comments} bình luận.")
                return

        paging = payload.get("paging", {})

        if not paging.get("next"):
            return

        next_cursor = paging.get("cursors", {}).get("after")

        if not next_cursor or next_cursor in seen_cursors:
            raise RuntimeError("Cursor phân trang bị thiếu hoặc lặp lại.")

        seen_cursors.add(next_cursor)
        params["after"] = next_cursor
        page_number += 1


def comments_to_csv_bytes(comments: Iterable[Dict[str, Any]]) -> bytes:
    """Chuyển danh sách bình luận đã chuẩn hóa thành nội dung tệp CSV (UTF-8 with BOM) khớp Batch Import."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for comment in comments:
        normalized = normalize_crawled_comment(comment)
        writer.writerow({col: normalized.get(col, "") for col in CSV_COLUMNS})
    return buf.getvalue().encode("utf-8-sig")


def save_to_csv(comments: Iterable[Dict[str, Any]], filename: str) -> int:
    """Ghi từng bình luận theo đúng cột hợp đồng Batch Import để giữ phần đã lấy được nếu API lỗi giữa chừng."""
    count = 0

    with open(filename, mode="w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        try:
            for comment in comments:
                normalized = normalize_crawled_comment(comment)
                writer.writerow({col: normalized.get(col, "") for col in CSV_COLUMNS})
                count += 1

        except (requests.RequestException, RuntimeError, ValueError):
            print(
                f"Thu thập bị gián đoạn. File {filename} "
                f"đã giữ được {count} bình luận."
            )
            raise

    print(f"Đã lưu {count} bình luận vào {filename}")
    return count


def main() -> None:
    post_id = input("Nhập Post ID từ Graph API: ").strip()
    post_url = input("Dán đường dẫn bài đăng Facebook: ").strip()
    api_version = input("Nhập phiên bản Graph API của app (dạng vXX.0): ").strip()
    unit_code = input("Nhập mã đơn vị ĐHTN (mặc định UNSPECIFIED): ").strip() or "UNSPECIFIED"

    # Khi nhập token, Terminal sẽ ẩn ký tự.
    access_token = getpass("Nhập Page Access Token: ").strip()

    if not all([post_id, post_url, api_version, access_token]):
        print("Bạn cần nhập đủ Post ID, URL, phiên bản API và token.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"binh_luan_facebook_{timestamp}.csv"

    try:
        with create_session() as session:
            comments = crawl_facebook_comments(
                session=session,
                post_id=post_id,
                post_url=post_url,
                access_token=access_token,
                api_version=api_version,
                max_comments=1000,
                unit_code=unit_code,
                source_name="FACEBOOK",
            )

            save_to_csv(comments, filename)

    except (requests.RequestException, RuntimeError, ValueError) as error:
        print(f"Lỗi thu thập: {error}")

    except OSError as error:
        print(f"Lỗi ghi file: {error}")


if __name__ == "__main__":
    main()
