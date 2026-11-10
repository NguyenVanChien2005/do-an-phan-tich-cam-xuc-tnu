import pandas as pd

# Đường dẫn đến file dataset trong thư mục dự án của bạn
file_path = 'synthetic_train.csv'

try:
    # Đọc file CSV bằng pandas với encoding='utf-8' để hiển thị chuẩn tiếng Việt
    df = pd.read_csv(file_path, encoding='utf-8')
    
    print("=== 5 DÒNG DỮ LIỆU ĐẦU TIÊN ===")
    print(df.head())
    
    print("\n=== THÔNG TIN TỔNG QUAN VỀ DATASET ===")
    print(df.info())
    
    print("\n=== THỐNG KÊ SỐ LƯỢNG CÁC NHÃN CẢM XÚC (SENTIMENT) ===")
    if 'sentiment' in df.columns:
        print(df['sentiment'].value_counts())

except Exception as e:
    print(f"Đã xảy ra lỗi khi đọc file: {e}")