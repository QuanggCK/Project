"""
===============================================================================
BƯỚC TIỀN XỬ LÝ: LÀM SẠCH VÀ CHUẨN BỊ DỮ LIỆU (DATA CLEANING)
===============================================================================
Mục đích module:
    - Xử lý các lỗi dữ liệu đã được phát hiện ở Bước 1 (Data Exploration).
    - Cụ thể trong bộ dữ liệu Telco Customer Churn:
        + Cột `TotalCharges` (Tổng cước phí tích lũy) ban đầu bị đọc là kiểu chuỗi (object)
          do có 11 dòng chứa ký tự khoảng trắng " " thay vì giá trị số.
        + Những dòng này thuộc về các khách hàng mới có `tenure = 0` (chưa dùng đủ 1 tháng),
          do đó họ chưa tích lũy tổng cước phí nào.
        + Hàm `clean()` sẽ chuyển đổi cột `TotalCharges` thành kiểu số thực (float) 
          và gán giá trị hợp lý là 0.0 cho các trường hợp này.
===============================================================================
"""

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiến hành làm sạch bảng dữ liệu Telco Churn:
    
    1. Tạo một bản sao độc lập (copy) của DataFrame để tránh sửa đổi trực tiếp lên dữ liệu gốc.
    2. Cắt bỏ khoảng trắng thừa ở cột `TotalCharges`.
    3. Dùng `pd.to_numeric` với tham số `errors="coerce"` để tự động chuyển các ô trống hoặc 
       ký tự không hợp lệ thành giá trị `NaN`.
    4. Dùng `.fillna(0.0)` để điền 0.0 vào các vị trí `NaN` đó, vì đây là những khách hàng mới
       gia nhập dịch vụ (thời gian sử dụng tenure = 0 tháng).
       
    Tham số:
        df (pd.DataFrame): DataFrame thô ban đầu.
        
    Trả về:
        pd.DataFrame: DataFrame sau khi đã chuẩn hóa cột TotalCharges về kiểu số thực.
    """
    df = df.copy()
    
    # Ép kiểu TotalCharges từ chuỗi sang số thực, điền 0.0 cho các ô trống
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)
    
    return df
