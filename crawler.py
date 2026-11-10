import requests
from bs4 import BeautifulSoup
import csv
import datetime

def crawl_news_articles(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }
    
    print(f"Đang kết nối tới trang: {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Lỗi: Không thể truy cập trang web (Mã lỗi: {response.status_code})")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ví dụ này cấu hình để cào các tiêu đề bài báo (thường nằm trong thẻ h3 hoặc h2 có chứa thẻ a)
        # Tùy trang web bạn có thể đổi thẻ và class cho phù hợp
        articles = soup.find_all('h3', class_='title-news') # Hoặc thẻ tương ứng của trang bạn chọn
        
        # Nếu trang không dùng class title-news, ta có thể thử tìm tất cả các thẻ h3
        if not articles:
            articles = soup.find_all('h3')
            
        data_list = []
        for item in articles:
            link_tag = item.find('a')
            if link_tag:
                title = link_tag.get_text(strip=True)
                link = link_tag.get('href', '')
                if title:
                    data_list.append({
                        'content': title, # Lấy tiêu đề bài viết làm nội dung ý kiến mẫu
                        'source': link,
                        'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
        print(f"Thu thập thành công {len(data_list)} bài viết từ trang.")
        return data_list

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình crawl: {e}")
        return []

def save_to_csv(data, filename='dulieu_bao_chi.csv'):
    if not data:
        print("Không có dữ liệu để lưu file.")
        return
        
    keys = data[0].keys()
    try:
        with open(filename, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"Đã lưu thành công dữ liệu vào file: {filename}")
    except Exception as e:
        print(f"Lỗi khi ghi file CSV: {e}")

if __name__ == "__main__":
    # Thay thế bằng một trang báo điện tử mở thực tế (ví dụ mục Giáo dục của VNExpress hoặc Dân Trí)
    TARGET_URL = "https://dantri.com.vn/giao-duc.htm"
    
    raw_data = crawl_news_articles(TARGET_URL)
    
    if raw_data:
        save_to_csv(raw_data)