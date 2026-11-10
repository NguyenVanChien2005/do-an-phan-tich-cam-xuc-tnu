import pandas as pd
import re

# 1. Đọc dataset
df = pd.read_csv('synthetic_train.csv', encoding='utf-8')

# Hàm chuẩn hóa và làm sạch văn bản tiếng Việt cơ bản
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Chuyển về chữ thường
    text = text.lower()
    
    # Loại bỏ URL/đường dẫn web
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Loại bỏ các ký tự đặc biệt, số và dấu câu thừa (chỉ giữ lại chữ cái tiếng Việt và khoảng trắng)
    # Lưu ý: Biểu thức chính quy này giữ lại các dấu tiếng Việt
    text = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', ' ', text)
    
    # Xử lý khoảng trắng thừa (nhiều khoảng trắng liên tiếp thành 1)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 2. Áp dụng hàm làm sạch cho toàn bộ cột 'sentence'
df['cleaned_sentence'] = df['sentence'].apply(clean_text)

# 3. Hiển thị kết quả so sánh trước và sau khi làm sạch
print("=== SO SÁNH TRƯỚC VÀ SAU KHI LÀM SẠCH ===")
for i in range(5):
    print(f"Gốc: {df['sentence'].iloc[i]}")
    print(f"Sạch: {df['cleaned_sentence'].iloc[i]}")
    print("-" * 40)

# 4. Lưu lại thành file mới để chuẩn bị cho bước huấn luyện mô hình (Tuần 4)
df.to_csv('synthetic_train_cleaned.csv', index=False, encoding='utf-8')
print("\n Đã lưu file dữ liệu sạch vào 'synthetic_train_cleaned.csv' thành công!")