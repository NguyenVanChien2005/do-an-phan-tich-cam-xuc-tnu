-- ==============================================================================
-- Migration 001: Nâng cấp từ cấu trúc bảng seed cũ (15 cột tnu_mysql_seed_data.csv)
-- sang kiến trúc tách biệt nguồn gốc dữ liệu, trạng thái xử lý và kiểm duyệt.
--
-- Giải thích lý do thay đổi:
-- 1. Tệp tnu_mysql_seed_data.csv cũ dùng khóa số tự tăng (unit_id, source_id, topic_id = topic + 1)
--    và cờ đơn is_verified (0/1), thiếu phân biệt nguồn gốc dữ liệu (data_origin),
--    thiếu tách biệt dự đoán gốc (predicted_*) và kết quả kiểm duyệt (reviewed_*),
--    thiếu idempotency_key / source_record_id chống ghi lặp và bảng lịch sử kiểm duyệt.
-- 2. Script này KHÔNG DROP bảng, KHÔNG xóa dữ liệu hiện hữu; chỉ bổ sung cột/bảng mới
--    và chuyển đổi an toàn. Các bản ghi từ seed cũ được gắn PUBLIC_BENCHMARK / SYNTHETIC
--    thay vì mặc định coi là TNU_REAL.
--
-- Chỉ chạy thủ công bởi quản trị viên sau khi sao lưu CSDL (mysqldump).
-- ==============================================================================

SET NAMES utf8mb4;
SET time_zone = '+07:00';

-- 1. Đảm bảo các bảng danh mục và bảng lịch sử tồn tại (tương thích schema_init.sql)
SOURCE ../schema_init.sql;

-- 2. Nếu hệ thống cũ có bảng legacy `feedback_seed_legacy`, chuyển đổi sang bảng `feedbacks`
-- mà không ghi đè dữ liệu thật hoặc giả định sai nguồn gốc dữ liệu:
-- (Ví dụ minh họa câu lệnh chuyển đổi an toàn khi tồn tại bảng legacy `feedback_records_v0`)

-- ALTER TABLE feedback_records_v0
--     ADD COLUMN IF NOT EXISTS data_origin VARCHAR(32) NOT NULL DEFAULT 'PUBLIC_BENCHMARK',
--     ADD COLUMN IF NOT EXISTS processing_status VARCHAR(24) NOT NULL DEFAULT 'DONE',
--     ADD COLUMN IF NOT EXISTS review_status VARCHAR(24) NOT NULL DEFAULT 'UNREVIEWED',
--     ADD COLUMN IF NOT EXISTS reviewed_sentiment_label TINYINT NULL,
--     ADD COLUMN IF NOT EXISTS reviewed_topic_code VARCHAR(64) NULL,
--     ADD COLUMN IF NOT EXISTS reviewed_by VARCHAR(128) NULL,
--     ADD COLUMN IF NOT EXISTS reviewed_at DATETIME NULL,
--     ADD COLUMN IF NOT EXISTS review_note VARCHAR(500) NULL,
--     ADD COLUMN IF NOT EXISTS record_version INT NOT NULL DEFAULT 1,
--     ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128) NULL UNIQUE,
--     ADD COLUMN IF NOT EXISTS source_record_id VARCHAR(191) NULL,
--     ADD COLUMN IF NOT EXISTS content_hash CHAR(64) NULL;

-- Lưu ý quan trọng khi nạp tnu_mysql_seed_data.csv:
-- - Không gắn nhãn data_origin = 'TNU_REAL' cho tệp seed 3.500 dòng này vì tệp được tổng hợp
--   từ tập UIT-VSFC (PUBLIC_BENCHMARK) và các câu sinh bổ sung (SYNTHETIC), đồng thời các xác suất
--   lớp phụ trong tệp gốc được chia đều (1 - confidence_score) / 2.
-- - Ánh xạ topic_id cũ sang topic_code ổn định:
--   1 -> 'LECTURER', 2 -> 'CURRICULUM', 3 -> 'FACILITY', 4 -> 'OTHER_SUPPORT'.

