##📈 StockAnalysis - Vietnam Stock Market

Trạng thái dự án: 🚧 Work In Progress (Đang thực hiện)

Dự án nghiên cứu và phân tích dữ liệu thị trường chứng khoán Việt Nam. Mục tiêu là xây dựng các công cụ trực quan hóa và áp dụng các mô hình phân tích để hỗ trợ việc đưa ra quyết định đầu tư.

##🚀 Tính năng hiện tại

[x] Kết nối và lấy dữ liệu giao dịch lịch sử từ API của vnstock.

[x] Xử lý dữ liệu cơ bản (làm sạch, định dạng thời gian).

[x] Trực quan hóa biểu đồ nến (Candlestick chart) cơ bản.

##🛠 Công nghệ sử dụng

Ngôn ngữ: Python

Thư viện chính:

vnstock: Nguồn cung cấp dữ liệu chứng khoán Việt Nam.

pandas: Xử lý và phân tích cấu trúc dữ liệu.

matplotlib/plotly: Trực quan hóa dữ liệu.

Môi trường: WSL (Windows Subsystem for Linux)

##📋 Hướng dẫn cài đặt & Chạy dự án
Để mở và chạy dự án này trên môi trường của bạn (đặc biệt là WSL), hãy làm theo các bước sau:

1. Clone dự án
   
```git clone https://github.com/username/StockAnalysis.git```

```cd StockAnalysis```

2. Cài đặt môi trường ảo
   
```python3 -m venv venv```

```source venv/bin/activate```

3. Cài đặt các thư viện cần thiết
   
```pip install vnstock pandas matplotlib```

4. Chạy phân tích
   
(Thay đổi mã chứng khoán trong file chính trước khi chạy)

```python main.py```

   📅 Roadmap (Kế hoạch phát triển)
   
[ ] Tích hợp thêm các chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands).

[ ] Xây dựng dashboard tương tác bằng Streamlit.

[ ] Thử nghiệm áp dụng mô hình Machine Learning để dự báo xu hướng ngắn hạn.

[ ] Phân tích báo cáo tài chính của doanh nghiệp từ nguồn dữ liệu vnstock.
