import csv
from datetime import datetime, timezone
from getpass import getpass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CSV_COLUMNS = [
    "platform",
    "post_id",
    "comment_id",
    "content",
    "source",
    "created_at",
    "collected_at",
    "like_count",
]


def create_session():
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


def crawl_facebook_comments(
    session,
    post_id,
    post_url,
    access_token,
    api_version,
    max_comments=1000,
):
    """Lấy tối đa max_comments bình luận cấp đầu của một bài."""
    endpoint = (
        f"https://graph.facebook.com/{api_version}/{post_id}/comments"
    )

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    params = {
        "fields": "id,message,created_time,like_count",
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
            comment_id = comment["id"]

            if comment_id in seen_ids:
                continue

            seen_ids.add(comment_id)

            yield {
                "platform": "facebook",
                "post_id": post_id,
                "comment_id": comment_id,
                "content": comment.get("message", ""),
                "source": post_url,
                "created_at": comment.get("created_time", ""),
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "like_count": comment.get("like_count", ""),
            }

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


def save_to_csv(comments, filename):
    """Ghi từng bình luận để giữ phần đã lấy được nếu API lỗi giữa chừng."""
    count = 0

    with open(filename, mode="w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        try:
            for comment in comments:
                writer.writerow(comment)
                count += 1

        except (requests.RequestException, RuntimeError, ValueError):
            print(
                f"Thu thập bị gián đoạn. File {filename} "
                f"đã giữ được {count} bình luận."
            )
            raise

    print(f"Đã lưu {count} bình luận vào {filename}")


def main():
    post_id = input("Nhập Post ID từ Graph API: ").strip()
    post_url = input("Dán đường dẫn bài đăng Facebook: ").strip()
    api_version = input("Nhập phiên bản Graph API của app (dạng vXX.0): ").strip()

    # Khi nhập token, Terminal sẽ ẩn ký tự.
    access_token = getpass("Nhập Page Access Token: ").strip()

    if not all([post_id, post_url, api_version, access_token]):
        print("Bạn cần nhập đủ Post ID, URL, phiên bản API và token.")
        return

    # Mỗi lần chạy tạo một file riêng.
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
            )

            save_to_csv(comments, filename)

    except (requests.RequestException, RuntimeError, ValueError) as error:
        print(f"Lỗi thu thập: {error}")

    except OSError as error:
        print(f"Lỗi ghi file: {error}")


if __name__ == "__main__":
    main()