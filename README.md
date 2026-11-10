# TNU Sentiment — Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên

## 1. Thông tin đồ án tốt nghiệp

- **Tên đề tài:** Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên
- **Sinh viên thực hiện:** Nguyễn Văn Chiến
- **Mã số sinh viên:** `DTC235200079`
- **Lớp:** CN CNTT K22A — Trường Đại học Công nghệ Thông tin và Truyền thông (ICTU)
- **Giảng viên hướng dẫn:** TS. Hà Thị Thanh
- **Công nghệ triển khai:** Python 3.13, Streamlit, Plotly, SQLAlchemy 2.x, MySQL 8.x, PyTorch, HuggingFace Transformers (PhoBERT-base-v2), `pyvi`.

---

## 2. Cấu trúc kiến trúc phân lớp

Ứng dụng được tổ chức tách biệt rõ ràng giữa giao diện, nghiệp vụ, truy cập dữ liệu, tiền xử lý NLP và adapter mô hình:

```text
DO_AN_PHAN_TICH_DU_LIEU/
├── app.py                                # Điểm khởi chạy Streamlit (định tuyến, cache tài nguyên, trạng thái phiên)
├── .streamlit/config.toml                # Cấu hình giao diện sáng học thuật chính thức của Streamlit
├── .env.example                          # Tệp mẫu biến môi trường (không chứa bí mật)
├── requirements.txt                      # Danh sách thư viện phụ thuộc đã xác nhận trên môi trường
├── tnu_sentiment/
│   ├── config.py                         # Đọc cấu hình môi trường, ngưỡng thử nghiệm, múi giờ Asia/Ho_Chi_Minh
│   ├── constants.py                      # Bảng màu cố định, ánh xạ nhãn cảm xúc (0/1/2), danh mục tham khảo
│   ├── domain/models.py                  # Dataclass định kiểu chặt cho phản hồi, dự đoán, bộ lọc, KPI, benchmark
│   ├── nlp/
│   │   ├── pii_masker.py                 # Che PII thống nhất (<EMAIL>, <PHONE>, <STUDENT_ID>) & băm ẩn danh
│   │   └── preprocessor.py               # Pipeline chuẩn hóa tiếng Việt + tách từ pyvi tương thích checkpoint
│   ├── models/
│   │   ├── sentiment_adapter.py          # Nạp và kiểm tra nghiêm ngặt checkpoint PhoBERT (từ chối base model thô)
│   │   ├── topic_adapter.py              # Quản lý trạng thái gán chủ đề thủ công theo mã ổn định
│   │   └── llm_helper.py                 # Trợ lý LLM tùy chọn (mặc định tắt, che PII trước khi gửi, kiểm tra JSON)
│   ├── repositories/
│   │   ├── base.py                       # Interface FeedbackRepository & EvaluationRepository dùng chung
│   │   ├── mysql_repo.py                 # SQL tham số hóa, lọc & phân trang tại MySQL, giao dịch Transaction
│   │   ├── demo_repo.py                  # Kho dữ liệu mô phỏng cô lập theo phiên từ fixture JSON
│   │   └── evaluation_repo.py            # Đọc kết quả thực nghiệm từ artifacts/evaluation/
│   ├── services/
│   │   ├── analysis_service.py           # Luồng phân tích đơn lẻ, phát hiện sửa văn bản sau phân tích, lưu idempotent
│   │   ├── stats_service.py              # Tính toán N_total, N_valid, tỷ lệ cảm xúc, NSS, xu hướng ngày/tuần
│   │   ├── import_service.py             # Import CSV/XLSX 8 bước (chặn công thức Excel, che PII, tổng kết 6 nhóm)
│   │   ├── review_service.py             # Kiểm duyệt theo feedback_id + record_version, bảo toàn dự đoán gốc
│   │   └── benchmark_service.py          # Kiểm tra điều kiện so sánh trực tiếp cùng evaluation_set_id
│   └── ui/
│       ├── theme.py                      # CSS tối giản (escape HTML đầy đủ) và Plotly template dùng chung
│       ├── components.py                 # Header trạng thái, banner dữ liệu, bộ lọc đồng bộ, footer đồ án
│       └── pages/                        # 4 phân hệ: overview.py, analyze.py, data_review.py, evaluation.py
├── sql/
│   ├── schema_init.sql                   # Schema cài mới an toàn cho MySQL 8.0+ (không chứa lệnh DROP TABLE)
│   └── migrations/001_upgrade_legacy_seed_schema.sql
├── fixtures/                             # Bộ dữ liệu Demo ẩn danh, 6 nhóm câu mẫu và tệp mẫu Import
├── artifacts/evaluation/                 # Artifact JSON kết quả đánh giá xác thực trên Validation & Test
├── scripts/evaluate_checkpoint.py        # Script đánh giá thực nghiệm sinh artifact từ checkpoint thực tế
└── tests/                                # Bộ kiểm thử tự động (pytest + Streamlit AppTest)
```

---

## 3. Tương thích Checkpoint & Quy tắc Trung thực Dữ liệu

### 3.1. Checkpoint và Pipeline tiền xử lý
- **Checkpoint cảm xúc:** Sử dụng trực tiếp `phobert_tnu_best_model` (`RobertaForSequenceClassification`), giữ nguyên ánh xạ nhãn `0: Negative (Tiêu cực)`, `1: Neutral (Trung tính)`, `2: Positive (Tích cực)` và giới hạn an toàn `max_length = 256` token.
- **Tách từ & chuẩn hóa:** Khớp với tập dữ liệu `TNU_Sentiment_Dataset_Final` (`pyvi.ViTokenizer`), giữ dấu câu cơ bản, mở rộng từ viết tắt học đường ĐHTN, bảo vệ tuyệt đối các token `<EMAIL>`, `<PHONE>`, `<STUDENT_ID>` qua mọi bước làm sạch.
- **Điều chỉnh tương thích có ghi nhận:**
  1. Khớp biểu tượng cảm xúc/emoticon theo thứ tự **chuỗi dài trước chuỗi ngắn** (`:))))` → `:)))` → `:))` → `:)`), và **không mặc định ép mọi `:)))` thành tiêu cực** để tránh làm sai lệch các câu phản hồi tích cực có biểu cảm cười.
  2. Chỉ rút gọn ký tự lặp kéo dài từ **3 ký tự liên tiếp trở lên** (`siêuuuu` → `siêu`), tránh làm biến dạng các từ có 2 chữ cái liền nhau hợp lệ (`oop`, `moodle`, `google`, `cntt`).
- **Phân loại chủ đề:** Repository hiện chưa có checkpoint fine-tune riêng cho chủ đề; hệ thống công bố rõ trạng thái **“Chủ đề được gán thủ công”** trên Header và sử dụng mã chủ đề ổn định (`LECTURER`, `CURRICULUM`, `FACILITY`, `OTHER_SUPPORT`, `UNCLASSIFIED`), tuyệt đối không dùng `argmax + 1` hoặc sinh chủ đề ngẫu nhiên.

### 3.2. Phân tách nguồn gốc dữ liệu (`data_origin`)
Hệ thống phân biệt tường minh 4 nhóm nguồn gốc:
- `TNU_REAL`: Phản hồi khảo sát thực tế trong phạm vi ĐHTN đã xác minh nguồn.
- `PUBLIC_BENCHMARK`: Dữ liệu học thuật công khai (như UIT-VSFC).
- `SYNTHETIC`: Dữ liệu sinh bổ sung phục vụ huấn luyện.
- `DEMO`: Dữ liệu minh họa giao diện hoặc câu mẫu kiểm thử.

> **Lưu ý về tệp `TNU_Sentiment_Dataset_Final/tnu_mysql_seed_data.csv`:** Qua rà soát kỹ thuật, tệp 3.500 dòng này được tổng hợp từ tập benchmark công khai và các câu sinh bổ sung, kèm phân phối xác suất hai lớp phụ bằng `(1 - confidence_score) / 2`. Do đó, không được nạp tệp này dưới nhãn `TNU_REAL` để trình bày như kết quả khảo sát thực tế của ĐHTN.

---

## 4. Quy tắc Chống ghi lặp & Loại trùng dữ liệu

Hệ thống tách bạch 3 trường hợp khác nhau thay vì chỉ dựa vào khóa nút giao diện:

1. **Bấm nút “Lưu phản hồi” lặp lại trên cùng một lượt phân tích:** Mỗi lần bấm *Phân tích* tạo một khóa `idempotency_key` duy nhất kèm ràng buộc `UNIQUE` tại tầng Repository/MySQL, đảm bảo chỉ ghi tối đa 1 bản ghi. Nếu người dùng sửa văn bản sau khi phân tích, chữ ký văn bản thay đổi sẽ vô hiệu hóa nút *Lưu* cho tới khi *Phân tích* lại.
2. **Import lại cùng một bản ghi nguồn từ tệp CSV/XLSX:** Sử dụng khóa nghiệp vụ duy nhất `(source_code, source_record_id)` (`UNIQUE KEY uq_source_record`). Các dòng trùng mã nguồn sẽ được đưa vào nhóm **Bỏ qua do trùng** trong báo cáo tổng kết.
3. **Hai phản hồi độc lập có nội dung giống nhau:** Mã băm `content_hash` (SHA-256 trên văn bản chuẩn hóa đã che PII) **không** đặt ràng buộc xóa/chặn cứng để tránh làm mất ý kiến trùng lặp tự nhiên của nhiều người học (ví dụ hai sinh viên cùng viết *"Phòng học thoáng mát"*). Thay vào đó, bản ghi đến sau vẫn được lưu và gắn cờ `is_duplicate_candidate = 1` để quản trị viên xem xét khi cần.

---

## 5. Hướng dẫn Cài đặt & Khởi chạy trên Windows / VS Code

### Bước 1: Mở thư mục dự án và cài đặt thư viện
Mở **Terminal (PowerShell)** tại thư mục gốc dự án trong VS Code:

```powershell
# Kiểm tra phiên bản Python (khuyến nghị Python 3.11 - 3.13)
python --version

# Cài đặt đầy đủ các gói phụ thuộc
python -m pip install -r requirements.txt
```

### Bước 2: Khởi chạy ngay ở Chế độ Demo giao diện (không cần MySQL)
```powershell
python -m streamlit run app.py
```
Mở trình duyệt tại địa chỉ hiển thị (mặc định `http://localhost:8501`). Bạn có thể chuyển đổi giữa **Chế độ demo giao diện** và **Chế độ kết nối dữ liệu thật (MySQL)** ngay trên thanh điều hướng bên trái.

### Bước 3: Cấu hình Chế độ Kết nối Dữ liệu Thật (MySQL)
1. Sao chép tệp cấu hình môi trường:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Tạo cơ sở dữ liệu trong MySQL 8.0+ và khởi tạo bảng an toàn bằng `sql/schema_init.sql`:
   ```sql
   CREATE DATABASE IF NOT EXISTS tnu_sentiment_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   USE tnu_sentiment_db;
   SOURCE sql/schema_init.sql;
   ```
3. Cập nhật thông tin `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD` trong tệp `.env` (có thể đặt `APP_MODE=real`).
4. Khởi động lại ứng dụng bằng `python -m streamlit run app.py`.

### Bước 4: Chạy đánh giá thực nghiệm & Kiểm thử tự động
```powershell
# Tái tạo các artifact đánh giá xác thực từ checkpoint trên tập Validation (1.709 mẫu) & Test (3.166 mẫu)
python scripts/evaluate_checkpoint.py

# Chạy toàn bộ bộ kiểm thử đơn vị và tích hợp giao diện (pytest + Streamlit AppTest)
python -m pytest -v
```

---

## 6. Giới hạn hiện tại của hệ thống

1. **Chưa có mô hình phân loại chủ đề riêng:** Hiện mới có checkpoint fine-tune cho bài toán cảm xúc 3 lớp; chủ đề được người nhập/người duyệt gán thủ công hoặc nạp theo cột `topic_code` khi import.
2. **Phân tích cảm xúc ở mức toàn câu (Sentence-level):** Với các câu ghép nhiều ý vừa khen vừa chê (ví dụ khen giảng viên nhưng chê máy chiếu), mô hình trả về một nhãn tổng thể và gắn cờ gợi ý rà soát thay vì tách cảm xúc theo từng khía cạnh (ABSA).
3. **Chưa tích hợp xác thực người dùng (SSO/Auth):** Phiên bản đồ án hiện vận hành cho một người quản trị cục bộ; trường `reviewed_by` do người duyệt tự nhập chỉ phục vụ ghi vết minh họa, chưa phải danh tính xác thực.
4. **Các ngưỡng vận hành mang tính thử nghiệm:** Ngưỡng gắn cờ cần kiểm duyệt (`confidence < 0.65`) và quy tắc cảnh báo tỷ lệ tiêu cực (`>= 35%` khi `N_valid >= 30`) là tham số cấu hình phục vụ theo dõi quản trị minh họa, cần hiệu chỉnh thêm khi có dữ liệu vận hành thực tế quy mô lớn.

