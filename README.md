# Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên (TNU Sentiment)

## 1. Thông tin Đồ án Tốt nghiệp

- **Tên đề tài:** Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên
- **Sinh viên thực hiện:** Nguyễn Văn Chiến
- **Mã số sinh viên:** `DTC235200079`
- **Lớp:** CN CNTT K22A — Trường Đại học Công nghệ Thông tin và Truyền thông (ICTU), Đại học Thái Nguyên
- **Giảng viên hướng dẫn:** TS. Hà Thị Thanh
- **Công nghệ cốt lõi:** Python 3.13, PyTorch, HuggingFace Transformers (`vinai/phobert-base-v2`), Multi-Class Focal Loss ($\gamma = 2,0$, label smoothing $\epsilon = 0,05$), PyVi (`ViTokenizer`), Gemini API (`gemini-2.5-flash` Hybrid Router), Streamlit, Plotly, SQLAlchemy 2.x, MySQL 8.0 (`utf8mb4_unicode_ci`).

---

## 2. Cấu trúc Thư mục Chuẩn của Dự án

```text
do-an-phan-tich-cam-xuc-tnu/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py                # Class TNUStudentFeedbackPreprocessor (Pipeline NLP 6 bước + SHA-256 + Ẩn danh PII)
│   ├── llm_fallback.py                # Định tuyến lai Gemini API (gemini-2.5-flash) khi confidence < 0.65 + Keyword Fallback
│   └── trainer_focal_loss.py          # MultiClassFocalLoss (gamma=2.0, epsilon=0.05) & FocalLossPhoBERTTrainer
├── database/
│   ├── schema.sql                     # Kịch bản MySQL 8.0 chuẩn 3NF gồm 5 bảng (tnu_units, data_sources, topics, student_feedbacks, model_evaluations)
│   └── seed_data.py                   # Script nạp 3.500 bản ghi mô phỏng từ tnu_mysql_seed_data.csv và bảng điểm thực nghiệm
├── dataset_final/
│   └── final_model_comparison_chapter3.csv  # Bảng đối sánh kết quả thực nghiệm Chương 3 (tách rõ Test và Validation)
├── TNU_Sentiment_Dataset_Final/
│   ├── tnu_train_final.csv            # Tập huấn luyện mở rộng: 12.140 mẫu (11.426 UIT-VSFC Train + 714 mẫu miền ĐHTN)
│   ├── tnu_val_final.csv              # Tập phát triển mở rộng: 1.709 mẫu (1.583 UIT-VSFC Val + 126 mẫu miền ĐHTN)
│   ├── tnu_test_final.csv             # Tập kiểm thử độc lập chuẩn UIT-VSFC: 3.166 mẫu
│   └── tnu_mysql_seed_data.csv        # Dữ liệu Seed Demo phục vụ khởi tạo giao diện (3.500 bản ghi, is_synthetic = 1)
├── app.py                             # Ứng dụng Web Streamlit + Plotly + SQLAlchemy hoàn chỉnh 4 phân hệ (Tab 1 -> Tab 4)
├── final_model_comparison_chapter3.csv
├── confusion_matrix_phobert_tnu.png   # Ma trận nhầm lẫn của PhoBERT-base-v2 + Focal Loss trên Test (3.166) và Validation (1.709)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 3. Phương pháp luận Học thuật, Thống kê Dữ liệu & Kết quả Thực nghiệm (Chương 3)

### 3.1. Mô hình PhoBERT-base-v2 và Quy mô Tập dữ liệu
1. **Dữ liệu tiền huấn luyện của PhoBERT-base-v2:** Mô hình `vinai/phobert-base-v2` được tiền huấn luyện trên **140GB văn bản tiếng Việt (gồm 20GB Wikipedia & tin tức kết hợp 120GB từ kho ngữ liệu OSCAR-2301)**, sử dụng thuật toán tách từ con BPE trên văn bản đã tách từ tiếng Việt bằng RDRSegmenter/PyVi.
2. **Quy mô tập UIT-VSFC gốc (16.175 câu):** Gồm **`11.426` mẫu Train + `1.583` mẫu Validation + `3.166` mẫu Test = `16.175` câu** phản hồi sinh viên (trong đó lớp Trung tính — Neutral chỉ chiếm **4,32%** tổng số mẫu, gây mất cân bằng lớp nghiêm trọng).
3. **Quy mô tập dữ liệu mở rộng (17.015 câu):** Đồ án bổ sung **840 mẫu câu mang từ vựng miền ĐHTN** (chia 85% tương ứng **`714` mẫu** vào tập Train $\rightarrow$ **`12.140` mẫu**, và 15% tương ứng **`126` mẫu** vào tập Validation $\rightarrow$ **`1.709` mẫu**; giữ nguyên **`3.166` mẫu** Test gốc độc lập).
   - *Lưu ý minh bạch nguồn dữ liệu:* 840 câu bổ sung mang từ vựng miền ĐHTN và 3.500 bản ghi trong `tnu_mysql_seed_data.csv` là dữ liệu mở rộng/mô phỏng (`is_synthetic = 1`), được tách biệt rõ với dữ liệu khảo sát thực tế (`is_synthetic = 0`) trên CSDL và Dashboard.
4. **Danh mục 9 đơn vị thành viên thuộc ĐHTN:** Hệ thống đồng bộ nhất quán **9 đơn vị trường/khoa thành viên** giữa báo cáo và `database/schema.sql`: `ICTU`, `TNUT`, `TUEBA`, `TNUE`, `TUAF`, `TUMP`, `TNUS`, `SFL`, `ISTNU`.

### 3.2. Ghi chú học thuật về Bộ chỉ số Đánh giá so với Đề cương H02 (PI 3.1)
> *Do bài toán phân tích cảm xúc và phân loại chủ đề là bài toán phân loại rời rạc đa lớp (Multi-class Classification) trên dữ liệu mất cân bằng, đồ án sử dụng bộ chỉ số Accuracy, Precision, Recall, Macro-F1, Weighted-F1 và Confusion Matrix; không sử dụng độ đo sai số tuyệt đối trung bình (MAE) vốn chỉ áp dụng cho bài toán hồi quy liên tục.*

### 3.3. Bảng 3.1 — Kết quả đối sánh chính thức trên Tập Kiểm thử độc lập UIT-VSFC (Test Split: $N = 3.166$ mẫu)

| Mô hình thực nghiệm | Tập đánh giá | Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro-F1 (%) | Weighted-F1 (%) | F1 - Negative (%) | F1 - Neutral (%) | F1 - Positive (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes + TF-IDF** | Test (3.166) | 86,83 | 59,01 | 60,00 | 59,45 | 84,51 | 89,93 | 0,00 | 88,41 |
| **Maximum Entropy + TF-IDF** | Test (3.166) | 89,17 | 78,61 | 63,34 | 64,99 | 87,28 | 91,54 | 12,09 | 91,34 |
| **Linear SVM (Balanced) + TF-IDF** | Test (3.166) | 89,29 | 72,89 | 74,01 | 73,41 | 88,85 | 91,89 | 36,75 | 91,58 |
| **PhoBERT-base-v2 + Focal Loss (Đợt 1)** | Test (3.166) | **93,78** | **84,12** | **81,95** | **82,91** | **93,46** | **95,23** | **57,86** | **95,63** |

#### Giải thích toán học giữa $\text{F1-score}$ và $\text{Recall}$ (Khắc phục ngụy biện $1 - \text{F1}$)
- Về mặt toán học, $\text{F1-score}$ là **trung bình điều hòa** của $\text{Precision}$ và $\text{Recall}$:
  $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
  Trong khi đó, tỷ lệ bỏ sót mẫu thực tế (False Negative Rate) của một lớp $c$ phải được tính theo độ đo $\text{Recall}_c$:
  $$\text{FNR}_c = 1 - \text{Recall}_c = \frac{\text{FN}_c}{\text{TP}_c + \text{FN}_c}$$
- **Nhận xét cho nhóm Baseline:** *"Do lớp Trung tính (Neutral) chỉ chiếm 4,32% tập dữ liệu, mô hình Naive Bayes hoàn toàn không học được ranh giới quyết định của lớp này (Recall-Neutral = 0,00%, F1-Neutral = 0,00%), trong khi Maximum Entropy cũng bị hai lớp đa số lấn át khiến F1-Neutral chỉ đạt 12,09% trên tập Test. Khi áp dụng cơ chế cân bằng trọng số lớp ở Linear SVM (class_weight='balanced'), F1-Neutral tăng gấp hơn 3 lần lên 36,75% (Macro-F1 đạt 73,41%)."*

### 3.4. Bảng 3.2 — Nhật ký tinh chỉnh sâu PhoBERT-base-v2 + Focal Loss trên Tập Phát triển mở rộng (Validation Split: $N = 1.709$ mẫu)
- **Cấu hình huấn luyện:** Đặt tối đa `num_train_epochs = 20` (hoặc `50` trên bản Colab cập nhật) kết hợp `EarlyStoppingCallback(early_stopping_patience = 6)`, tối ưu theo tiêu chí `metric_for_best_model = "macro_f1"`.
- **Quá trình hội tụ:** Dừng sớm tại **Epoch 10** và trích xuất checkpoint tốt nhất tại **Epoch 4**:
  - **Accuracy (Val):** `95,61%`
  - **Macro Precision (Val):** `91,17%`
  - **Macro Recall (Val):** `91,10%`
  - **Macro-F1 (Val):** `91,13%`
  - **Weighted-F1 (Val):** `95,57%`
  - **F1 từng lớp (Val):** Negative `96,74%` | Neutral `80,00%` | Positive `96,65%`.
- *(Hàm `evaluate_checkpoint_on_test_split()` trong `src/trainer_focal_loss.py` hỗ trợ nạp trực tiếp checkpoint Epoch 4 và chạy `model(**enc)` trên 3.166 câu Test độc lập khi cần kiểm định lại).*

---

## 4. Hướng dẫn Cài đặt & Vận hành

### 4.1. Một lệnh duy nhất để máy mới tải trọng số (`model.safetensors`) và chạy Inference thật
Do tệp trọng số `model.safetensors` (~515MB) không lưu trực tiếp trên Git thông thường, thông tin kiểm định toàn vẹn (`SHA-256`, phiên bản `transformers`, `pyvi`, `underthesea`, `preprocessor_version`) được khai báo trong [`models/checkpoint_manifest.json`](models/checkpoint_manifest.json). Trên máy mới, chỉ cần chạy **1 lệnh duy nhất** dưới đây để tự động tải trọng số, đối khớp mã băm `SHA-256` và khởi chạy ứng dụng với mô hình thật:

```powershell
# Tải trọng số từ HuggingFace Hub / Google Drive URL, kiểm tra SHA-256 và khởi chạy ứng dụng:
python scripts/download_checkpoint.py --url $env:TNU_CHECKPOINT_URL && python -m streamlit run app.py
```

Hoặc kiểm tra nhanh tính toàn vẹn `SHA-256` của trọng số đã có trên đĩa:
```powershell
python scripts/download_checkpoint.py --verify-only
```

### 4.2. Cài đặt môi trường, Khởi tạo CSDL & Kiểm thử tự động (`pytest`)

```powershell
# 1. Cài đặt các thư viện phụ thuộc
python -m pip install -r requirements.txt

# 2. (Tùy chọn) Khởi tạo cấu trúc MySQL 8.0 và nạp 3.500 bản ghi Seed Demo
python database/seed_data.py

# 3. Chạy bộ kiểm thử tự động (tự động skip nhóm @pytest.mark.requires_checkpoint nếu chưa tải model.safetensors)
pytest

# 4. Khởi chạy ứng dụng Web Streamlit (tự động chuyển sang CSV Fallback nếu chưa bật MySQL)
python -m streamlit run app.py
```
