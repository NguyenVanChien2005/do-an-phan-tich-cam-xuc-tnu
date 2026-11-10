-- ==============================================================================
-- TNU Sentiment — Khởi tạo Schema MySQL 8.0+ cho cài đặt mới (schema_init.sql)
-- Nguyên tắc an toàn: KHÔNG sử dụng DROP TABLE hoặc xóa dữ liệu hiện có.
-- Múi giờ quy ước lưu trữ DATETIME: Asia/Ho_Chi_Minh (+07:00).
-- ==============================================================================

SET NAMES utf8mb4;
SET time_zone = '+07:00';

CREATE TABLE IF NOT EXISTS units (
    unit_id INT AUTO_INCREMENT PRIMARY KEY,
    unit_code VARCHAR(32) NOT NULL UNIQUE,
    unit_name VARCHAR(255) NOT NULL,
    description VARCHAR(500) NULL,
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS sources (
    source_id INT AUTO_INCREMENT PRIMARY KEY,
    source_code VARCHAR(64) NOT NULL UNIQUE,
    source_name VARCHAR(255) NOT NULL,
    description VARCHAR(500) NULL,
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS topics (
    topic_id INT AUTO_INCREMENT PRIMARY KEY,
    topic_code VARCHAR(64) NOT NULL UNIQUE,
    topic_name VARCHAR(255) NOT NULL,
    corpus_label_idx INT NULL COMMENT 'Ánh xạ ổn định với nhãn corpus (0..3), NULL nếu chưa phân loại',
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS import_batches (
    batch_id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    declared_origin VARCHAR(32) NOT NULL,
    origin_confirmation_note VARCHAR(500) NULL,
    total_rows INT NOT NULL DEFAULT 0,
    saved_count INT NOT NULL DEFAULT 0,
    duplicate_skipped_count INT NOT NULL DEFAULT 0,
    validation_error_count INT NOT NULL DEFAULT 0,
    inference_error_count INT NOT NULL DEFAULT 0,
    db_error_count INT NOT NULL DEFAULT 0,
    unprocessed_count INT NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS feedbacks (
    feedback_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    idempotency_key VARCHAR(128) NULL UNIQUE COMMENT 'Chống ghi lặp khi bấm Lưu nhiều lần cùng một thao tác',
    source_record_id VARCHAR(191) NULL COMMENT 'Mã bản ghi gốc từ hệ thống/file nguồn để chống import lặp',
    content_hash CHAR(64) NOT NULL COMMENT 'SHA-256 nội dung chuẩn hóa để gợi ý ứng viên trùng (không chặn phản hồi độc lập)',
    author_hash CHAR(64) NULL COMMENT 'Mã băm ẩn danh nội bộ, không hiển thị ra UI/Export',

    data_origin VARCHAR(32) NOT NULL COMMENT 'TNU_REAL | PUBLIC_BENCHMARK | SYNTHETIC | DEMO',
    origin_note VARCHAR(255) NULL,
    batch_id VARCHAR(64) NULL,

    unit_code VARCHAR(32) NOT NULL DEFAULT 'UNSPECIFIED',
    source_code VARCHAR(64) NOT NULL DEFAULT 'UNSPECIFIED',

    masked_content TEXT NOT NULL COMMENT 'Nội dung đã che PII (<EMAIL>, <PHONE>, <STUDENT_ID>)',
    processed_content TEXT NULL COMMENT 'Văn bản sau tiền xử lý đưa vào mô hình',
    pii_types_detected VARCHAR(128) NULL,
    was_truncated TINYINT(1) NOT NULL DEFAULT 0,
    is_multi_aspect TINYINT(1) NOT NULL DEFAULT 0,
    is_duplicate_candidate TINYINT(1) NOT NULL DEFAULT 0,

    processing_status VARCHAR(24) NOT NULL DEFAULT 'PENDING' COMMENT 'PENDING | RUNNING | DONE | ERROR',
    processing_error VARCHAR(500) NULL,

    predicted_sentiment_label TINYINT NULL COMMENT '0: Negative, 1: Neutral, 2: Positive',
    predicted_sentiment_text VARCHAR(24) NULL,
    confidence_score DECIMAL(6,4) NULL,
    prob_negative DECIMAL(6,4) NULL,
    prob_neutral DECIMAL(6,4) NULL,
    prob_positive DECIMAL(6,4) NULL,

    predicted_topic_code VARCHAR(64) NOT NULL DEFAULT 'UNCLASSIFIED',
    topic_assignment_source VARCHAR(32) NOT NULL DEFAULT 'MANUAL' COMMENT 'MANUAL | MODEL | RULE | IMPORT',

    model_version VARCHAR(128) NULL,
    pipeline_version VARCHAR(64) NULL,
    inference_time_ms DECIMAL(10,2) NULL,

    llm_suggested_sentiment TINYINT NULL,
    llm_suggested_topic_code VARCHAR(64) NULL,
    llm_note VARCHAR(500) NULL,

    review_status VARCHAR(24) NOT NULL DEFAULT 'UNREVIEWED' COMMENT 'UNREVIEWED | NEEDS_REVIEW | VERIFIED',
    review_reason VARCHAR(255) NULL,
    reviewed_sentiment_label TINYINT NULL,
    reviewed_topic_code VARCHAR(64) NULL,
    reviewed_by VARCHAR(128) NULL COMMENT 'Tên người duyệt tự khai báo nếu chưa tích hợp SSO/Auth',
    reviewed_at DATETIME NULL,
    review_note VARCHAR(500) NULL,
    record_version INT NOT NULL DEFAULT 1,

    feedback_date DATETIME NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uq_source_record (source_code, source_record_id),
    INDEX idx_feedbacks_origin_date (data_origin, feedback_date),
    INDEX idx_feedbacks_unit_date (unit_code, feedback_date),
    INDEX idx_feedbacks_status_review (processing_status, review_status),
    INDEX idx_feedbacks_content_hash (content_hash),
    CONSTRAINT fk_feedbacks_unit FOREIGN KEY (unit_code) REFERENCES units (unit_code),
    CONSTRAINT fk_feedbacks_source FOREIGN KEY (source_code) REFERENCES sources (source_code),
    CONSTRAINT fk_feedbacks_pred_topic FOREIGN KEY (predicted_topic_code) REFERENCES topics (topic_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS feedback_review_logs (
    log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    feedback_id BIGINT NOT NULL,
    previous_review_status VARCHAR(24) NULL,
    new_review_status VARCHAR(24) NOT NULL,
    previous_sentiment_label TINYINT NULL,
    new_sentiment_label TINYINT NULL,
    previous_topic_code VARCHAR(64) NULL,
    new_topic_code VARCHAR(64) NULL,
    reviewed_by VARCHAR(128) NOT NULL,
    is_authenticated_user TINYINT(1) NOT NULL DEFAULT 0,
    review_note VARCHAR(500) NULL,
    record_version_before INT NOT NULL,
    record_version_after INT NOT NULL,
    changed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_review_logs_feedback (feedback_id, changed_at),
    CONSTRAINT fk_review_logs_feedback FOREIGN KEY (feedback_id) REFERENCES feedbacks (feedback_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- Danh mục khởi tạo tham khảo trong phạm vi dự án
-- Lưu ý: Các mã đơn vị dưới đây là danh mục tham khảo ban đầu, có thể mở rộng
-- trực tiếp trong bảng units mà không cần sửa mã nguồn.
-- ------------------------------------------------------------------------------
INSERT IGNORE INTO units (unit_code, unit_name, description) VALUES
    ('UNSPECIFIED', 'Chưa xác định', 'Đơn vị chưa được khai báo trong phản hồi'),
    ('TNU_COMMON', 'Khối chung ĐHTN', 'Trung tâm, ban chức năng hoặc phạm vi chung ĐHTN'),
    ('ICTU', 'Trường ĐH Công nghệ Thông tin và Truyền thông', 'Danh mục khởi tạo tham khảo'),
    ('TNUT', 'Trường ĐH Kỹ thuật Công nghiệp', 'Danh mục khởi tạo tham khảo'),
    ('TUEBA', 'Trường ĐH Kinh tế và Quản trị Kinh doanh', 'Danh mục khởi tạo tham khảo'),
    ('TNUE', 'Trường ĐH Sư phạm', 'Danh mục khởi tạo tham khảo'),
    ('TUAF', 'Trường ĐH Nông Lâm', 'Danh mục khởi tạo tham khảo'),
    ('TUMP', 'Trường ĐH Y - Dược', 'Danh mục khởi tạo tham khảo'),
    ('TNUS', 'Trường ĐH Khoa học', 'Danh mục khởi tạo tham khảo'),
    ('SFL', 'Trường Ngoại ngữ', 'Danh mục khởi tạo tham khảo'),
    ('ISTNU', 'Khoa Quốc tế', 'Danh mục khởi tạo tham khảo');

INSERT IGNORE INTO sources (source_code, source_name, description) VALUES
    ('UNSPECIFIED', 'Chưa xác định', 'Nguồn thu thập chưa xác định'),
    ('SURVEY_PORTAL', 'Khảo sát ý kiến người học', 'Phiếu khảo sát định kỳ học kỳ'),
    ('LMS_FEEDBACK', 'Hệ thống học tập trực tuyến (LMS)', 'Phản hồi từ hệ thống quản lý học tập'),
    ('STUDENT_FORUM', 'Kênh phản hồi sinh viên', 'Diễn đàn / cổng thông tin sinh viên'),
    ('QA_BOX', 'Hòm thư góp ý đơn vị', 'Kênh tiếp nhận ý kiến trực tiếp hoặc biểu mẫu góp ý'),
    ('CSV_IMPORT', 'Nhập từ tệp CSV/XLSX', 'Dữ liệu nạp theo lô qua chức năng Import');

INSERT IGNORE INTO topics (topic_code, topic_name, corpus_label_idx) VALUES
    ('UNCLASSIFIED', 'Chưa phân loại', NULL),
    ('LECTURER', 'Giảng viên & Phương pháp giảng dạy', 0),
    ('CURRICULUM', 'Chương trình đào tạo & Học phần', 1),
    ('FACILITY', 'Cơ sở vật chất & Hạ tầng CNTT', 2),
    ('OTHER_SUPPORT', 'Công tác sinh viên & Dịch vụ khác', 3);

