"""
===============================================================================
TỆP: app.py
MỤC ĐÍCH: 
    - File chạy chính của ứng dụng web Flask (Entry point).
    - Khởi tạo web server và định tuyến (routing) các đường dẫn URL đến các 
      hàm xử lý dữ liệu và kết xuất giao diện (HTML templates).
    - Kết nối các module trong thư mục src/ (data_loader, analysis, model)
      với các template tương ứng trong thư mục templates/.
===============================================================================
"""

from flask import Flask, render_template

# Nhập các hàm xử lý phân tích và mô hình từ thư mục mã nguồn src/
from src.analysis import churn_rate_by_category
from src.data_loader import load_raw, summarize
from src.model import run as run_model

# Khởi tạo đối tượng ứng dụng Flask
app = Flask(__name__)


@app.route("/")
def index():
    """
    Trang 1: Tổng quan dữ liệu (Bước 1)
    - Đường dẫn: http://localhost:5000/
    - Chức năng: 
        + Nạp tập dữ liệu thô từ CSV thông qua `load_raw()`.
        + Tính toán các thống kê tổng quan (số dòng, số cột, giá trị thiếu, 
          ô trống ẩn, tỷ lệ churn, mẫu 10 dòng đầu) bằng hàm `summarize()`.
        + Truyền các tham số đã chuẩn bị sang template 'index.html' để hiển thị.
    """
    df = load_raw()
    info = summarize(df)
    
    return render_template(
        "index.html",
        n_rows=info["n_rows"],
        n_cols=info["n_cols"],
        duplicated_ids=info["duplicated_ids"],
        # Chuyển DataFrame thông tin cột thành danh sách dictionary để Jinja2 lặp qua
        columns=info["columns"].to_dict("records"),
        # Chuyển bảng tần số Churn thành danh sách dictionary
        churn=info["churn"].reset_index().to_dict("records"),
        # Danh sách tên các cột để làm tiêu đề bảng
        head_cols=list(info["head"].columns),
        # 10 dòng dữ liệu mẫu đầu tiên dạng mảng lồng nhau
        head_rows=info["head"].values.tolist(),
    )


@app.route("/factors")
def factors():
    """
    Trang 2: Phân tích tương quan theo nhóm (Bước 2)
    - Đường dẫn: http://localhost:5000/factors
    - Chức năng:
        + Nạp tập dữ liệu thô qua `load_raw()`.
        + Tính toán bảng chéo (cross-tabulation) tỷ lệ Rời bỏ (Churn=Yes) và Ở lại (Churn=No)
          cho từng nhóm của 16 biến phân loại thông qua hàm `churn_rate_by_category(df)`.
        + Truyền dictionary dữ liệu sang 'factors.html' để Chart.js hiển thị biểu đồ tương tác.
    """
    df = load_raw()
    data = churn_rate_by_category(df)
    return render_template("factors.html", data=data)


@app.route("/model")
def model_page():
    """
    Trang 3: Huấn luyện và Đánh giá mô hình máy học (Bước 3)
    - Đường dẫn: http://localhost:5000/model
    - Chức năng:
        + Gọi hàm `run_model()` để thực thi quy trình: làm sạch dữ liệu, mã hóa biến, 
          chia tập Train/Test, huấn luyện 3 mô hình (Logistic Regression, Random Forest, 
          HistGradientBoosting), đánh giá chỉ số, trích xuất hệ số hồi quy và độ quan trọng đặc trưng.
        + Kết quả được nạp nhanh từ file đệm `src/model_cache.json` nếu đã có sẵn.
        + Truyền kết quả `r` vào template 'model.html'.
    """
    results = run_model()
    return render_template("model.html", r=results)


if __name__ == "__main__":
    # Chạy ứng dụng web ở chế độ Debug (tự động tải lại mã khi có thay đổi và hiển thị chi tiết lỗi)
    app.run(debug=True)
