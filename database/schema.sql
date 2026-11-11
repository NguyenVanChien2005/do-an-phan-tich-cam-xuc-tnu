-- ==============================================================================
-- KỊCH BẢN KHỞI TẠO CƠ SỞ DỮ LIỆU MYSQL 8.0 CHUẨN 3NF (utf8mb4_unicode_ci)
-- Đề tài: Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng
--         trong Đại học Thái Nguyên (TNU Sentiment)
-- Sinh viên: Nguyễn Văn Chiến - MSSV: DTC235200079 - Lớp: CN CNTT K22A (ICTU)
-- Giảng viên hướng dẫn: TS. Hà Thị Thanh
-- ==============================================================================

SET NAMES utf8mb4;
SET time_zone = '+07:00';

-- ------------------------------------------------------------------------------
-- 1. Bảng Danh mục 9 Đơn vị Trường / Khoa thành viên thuộc Đại học Thái Nguyên
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tnu_units (
    unit_id INT AUTO_INCREMENT PRIMARY KEY,
    unit_code VARCHAR(20) NOT NULL UNIQUE,
    unit_name VARCHAR(255) NOT NULL,
    unit_type VARCHAR(100) NOT NULL DEFAULT 'Trường/Khoa thành viên',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- 2. Bảng Danh mục Nguồn thu thập ý kiến phản hồi
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS data_sources (
    source_id INT AUTO_INCREMENT PRIMARY KEY,
    source_code VARCHAR(50) NOT NULL UNIQUE,
    source_name VARCHAR(255) NOT NULL,
    description VARCHAR(500) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- 3. Bảng Danh mục 4 Chủ đề phản ánh học thuật
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topics (
    topic_id INT AUTO_INCREMENT PRIMARY KEY,
    topic_code VARCHAR(50) NOT NULL UNIQUE,
    topic_name VARCHAR(255) NOT NULL,
    description VARCHAR(500) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- 4. Bảng Phản hồi Người học (Tách biệt Nhãn Dự đoán AI & Nhãn Kiểm duyệt Active Learning)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS student_feedbacks (
    feedback_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    content_hash CHAR(64) NOT NULL UNIQUE COMMENT 'Băm SHA-256 nội dung chuẩn hóa để chống trùng lặp khi import hoặc bấm lưu nhiều lần',
    author_hash CHAR(64) NULL COMMENT 'Băm ẩn danh SHA-256 kèm SECRET_SALT theo Nghị định 13/2023/NĐ-CP',

    unit_id INT NOT NULL,
    source_id INT NOT NULL,
    topic_id INT NOT NULL COMMENT 'Chủ đề dự đoán gốc của AI (1..4)',

    raw_content TEXT NOT NULL COMMENT 'Nội dung phản hồi đã ẩn danh PII (<EMAIL>, <PHONE>, <STUDENT_ID>)',
    processed_content TEXT NULL COMMENT 'Chuỗi văn bản sau tiền xử lý 6 bước và tách từ PyVi',

    sentiment_label TINYINT NOT NULL COMMENT 'Nhãn cảm xúc dự đoán gốc của AI (0: Negative, 1: Neutral, 2: Positive)',
    sentiment_text VARCHAR(20) NOT NULL COMMENT 'Negative | Neutral | Positive',
    confidence_score DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
    prob_negative DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
    prob_neutral DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
    prob_positive DECIMAL(6,4) NOT NULL DEFAULT 0.0000,
    model_used VARCHAR(100) NOT NULL DEFAULT 'PhoBERT-base-v2-Focal',
    latency_ms DECIMAL(10,2) NULL,

    is_synthetic TINYINT(1) NOT NULL DEFAULT 0 COMMENT '0: Dữ liệu thực tế ĐHTN, 1: Dữ liệu mô phỏng/Seed Demo',
    is_verified TINYINT(1) NOT NULL DEFAULT 0,

    review_status ENUM('AUTO_APPROVED', 'NEEDS_REVIEW', 'VERIFIED') NOT NULL DEFAULT 'AUTO_APPROVED',
    reviewed_sentiment_label TINYINT NULL COMMENT 'Nhãn cảm xúc do cán bộ quản trị hậu kiểm (0, 1, 2)',
    reviewed_topic_id INT NULL COMMENT 'Mã chủ đề do cán bộ quản trị hậu kiểm (1..4)',
    reviewed_by VARCHAR(100) NULL,
    reviewed_at DATETIME NULL,
    review_note VARCHAR(500) NULL,

    feedback_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_dashboard_filter (feedback_date, unit_id, source_id, topic_id, is_synthetic),
    INDEX idx_review_queue (review_status, confidence_score),
    CONSTRAINT fk_sf_unit FOREIGN KEY (unit_id) REFERENCES tnu_units (unit_id),
    CONSTRAINT fk_sf_source FOREIGN KEY (source_id) REFERENCES data_sources (source_id),
    CONSTRAINT fk_sf_topic FOREIGN KEY (topic_id) REFERENCES topics (topic_id),
    CONSTRAINT fk_sf_reviewed_topic FOREIGN KEY (reviewed_topic_id) REFERENCES topics (topic_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ------------------------------------------------------------------------------
-- 5. Bảng Lưu trữ Kết quả Thực nghiệm & Đối sánh Mô hình (Tách biệt Validation và Test)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_evaluations (
    eval_id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(150) NOT NULL,
    evaluation_split VARCHAR(100) NOT NULL COMMENT 'Phân biệt rõ Test (3.166) và Validation (1.709 - Epoch 4)',
    n_samples INT NOT NULL,
    accuracy DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    macro_precision DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    macro_recall DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    macro_f1 DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    weighted_f1 DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    f1_negative DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    f1_neutral DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    f1_positive DECIMAL(6,2) NOT NULL COMMENT 'Đơn vị %',
    notes VARCHAR(500) NULL,
    evaluated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_model_split (model_name, evaluation_split)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ==============================================================================
-- DỮ LIỆU DANH MỤC KHỞI TẠO (9 ĐƠN VỊ THÀNH VIÊN ĐHTN, 4 NGUỒN, 4 CHỦ ĐỀ)
-- ==============================================================================
INSERT IGNORE INTO tnu_units (unit_id, unit_code, unit_name, unit_type) VALUES
    (1, 'ICTU',  'Trường Đại học Công nghệ Thông tin và Truyền thông (ICTU)', 'Trường Đại học thành viên'),
    (2, 'TNUT',  'Trường Đại học Kỹ thuật Công nghiệp (TNUT)', 'Trường Đại học thành viên'),
    (3, 'TUEBA', 'Trường Đại học Kinh tế và Quản trị Kinh doanh (TUEBA)', 'Trường Đại học thành viên'),
    (4, 'TNUE',  'Trường Đại học Sư phạm (TNUE)', 'Trường Đại học thành viên'),
    (5, 'TUAF',  'Trường Đại học Nông Lâm (TUAF)', 'Trường Đại học thành viên'),
    (6, 'TUMP',  'Trường Đại học Y - Dược (TUMP)', 'Trường Đại học thành viên'),
    (7, 'TNUS',  'Trường Đại học Khoa học (TNUS)', 'Trường Đại học thành viên'),
    (8, 'SFL',   'Trường Ngoại ngữ (SFL)', 'Trường thuộc ĐHTN'),
    (9, 'ISTNU', 'Khoa Quốc tế (ISTNU)', 'Khoa trực thuộc ĐHTN');

INSERT IGNORE INTO data_sources (source_id, source_code, source_name, description) VALUES
    (1, 'SURVEY_PORTAL', 'Hệ thống khảo sát người học cuối học kỳ', 'Phiếu khảo sát chính thức của các đơn vị thuộc ĐHTN'),
    (2, 'LMS_ELEARNING', 'Hệ thống quản lý học tập trực tuyến (LMS)', 'Phản hồi từ học phần trên LMS / E-Learning'),
    (3, 'FANPAGE_FORUM', 'Diễn đàn & Fanpage cộng đồng sinh viên ĐHTN', 'Bình luận thu thập qua Graph API / Fanpage đơn vị'),
    (4, 'QA_IMPORT_BOX', 'Hòm thư góp ý trực tuyến & Import CSV/XLSX', 'Kênh tiếp nhận ý kiến trực tiếp hoặc nhập theo lô'),
    (5, 'FACEBOOK',      'Bình luận Fanpage / Hội nhóm Facebook', 'Bình luận thu thập qua Facebook Graph API'),
    (6, 'YOUTUBE',       'Bình luận kênh YouTube', 'Bình luận thu thập từ video/kênh YouTube'),
    (7, 'NEWS_COMMENT',  'Bình luận bài viết / Bản tin điện tử', 'Bình luận thu thập từ chuyên trang tin tức/bài đăng');

INSERT IGNORE INTO topics (topic_id, topic_code, topic_name, description) VALUES
    (1, 'LECTURER',   'Giảng viên & Phương pháp giảng dạy', 'Thái độ, chuyên môn, phương pháp truyền đạt và hỗ trợ của giảng viên'),
    (2, 'CURRICULUM', 'Chương trình đào tạo & Học phần', 'Khối lượng kiến thức, giáo trình, lịch học, lịch thi, thực hành'),
    (3, 'FACILITY',   'Cơ sở vật chất & Hạ tầng CNTT', 'Phòng học, phòng máy, máy chiếu, điều hòa, Wifi, cổng đăng ký tín chỉ'),
    (4, 'OTHER',      'Khác (Hành chính, Học phí, CTSV)', 'Thủ tục hành chính, học phí, học bổng, cố vấn học tập, hoạt động SV');

-- ==============================================================================
-- DỮ LIỆU ĐỐI SÁNH THỰC NGHIỆM (TÁCH BIỆT TẬP TEST 3.166 MẪU & VALIDATION 1.709 MẪU)
-- ==============================================================================
INSERT IGNORE INTO model_evaluations (
    model_name, evaluation_split, n_samples,
    accuracy, macro_precision, macro_recall, macro_f1, weighted_f1,
    f1_negative, f1_neutral, f1_positive, notes
) VALUES
    ('Naive Bayes + TF-IDF', 'Test (3.166)', 3166,
     86.83, 59.01, 60.00, 59.45, 84.51, 89.93, 0.00, 88.41,
     'Baseline thống kê: Recall-Neutral = 0.00%, không học được ranh giới lớp thiểu số Neutral (4.32%)'),
    ('Maximum Entropy + TF-IDF', 'Test (3.166)', 3166,
     89.17, 78.61, 63.34, 64.99, 87.28, 91.54, 12.09, 91.34,
     'Logistic Regression đa lớp bị hai lớp đa số lấn át, F1-Neutral đạt 12.09%'),
    ('Linear SVM (Balanced) + TF-IDF', 'Test (3.166)', 3166,
     89.29, 72.89, 74.01, 73.41, 88.85, 91.89, 36.75, 91.58,
     'Áp dụng class_weight=balanced giúp F1-Neutral tăng gấp hơn 3 lần lên 36.75%'),
    ('PhoBERT-base-v2 + Focal Loss (Đợt 1)', 'Test (3.166)', 3166,
     93.78, 84.12, 81.95, 82.91, 93.46, 95.23, 57.86, 95.63,
     'Đánh giá độc lập trên Tập Kiểm thử chuẩn UIT-VSFC (N = 3.166 mẫu)'),
    ('PhoBERT-base-v2 + Focal Loss (Đợt 2 - Đỉnh Epoch 4)', 'Validation (1.709)', 1709,
     95.61, 91.17, 91.10, 91.13, 95.57, 96.74, 80.00, 96.65,
     'Tập Phát triển mở rộng (1.583 UIT-VSFC Val + 126 mẫu miền ĐHTN); EarlyStopping dừng tại Epoch 10, đỉnh tại Epoch 4');

