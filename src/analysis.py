"""
===============================================================================
BƯỚC 2: PHÂN TÍCH TƯƠNG QUAN ĐƠN BIẾN (CATEGORICAL FEATURE ANALYSIS)
===============================================================================
Mục đích module:
    - Khảo sát mối liên hệ giữa từng biến phân loại (nhân khẩu học, dịch vụ viễn thông,
      gói hợp đồng, phương thức thanh toán) với tỷ lệ rời bỏ (Churn) của khách hàng.
    - Tính toán bảng tần số chéo (crosstab) và chuyển đổi thành tỷ lệ phần trăm (%).
    - Sắp xếp các nhóm theo thứ tự tỷ lệ rời bỏ từ cao nhất xuống thấp nhất để người dùng
      dễ dàng nhận diện nhóm đối tượng có nguy cơ rời bỏ cao nhất.
===============================================================================
"""

import pandas as pd

# Danh sách 16 biến phân loại (categorical features) trong bộ dữ liệu cần phân tích
CATEGORICAL = [
    # Nhóm nhân khẩu học:
    "gender",           # Giới tính (Female / Male)
    "SeniorCitizen",    # Người cao tuổi (0: Không, 1: Có)
    "Partner",          # Có bạn đời/vợ chồng (Yes / No)
    "Dependents",       # Có người phụ thuộc/con cái (Yes / No)
    
    # Nhóm dịch vụ viễn thông cơ bản:
    "PhoneService",     # Dịch vụ điện thoại (Yes / No)
    "MultipleLines",    # Sử dụng nhiều đường dây điện thoại (No, Yes, No phone service)
    "InternetService",  # Loại đường truyền Internet (DSL, Fiber optic, No)
    
    # Nhóm dịch vụ giá trị gia tăng (VAS):
    "OnlineSecurity",   # Dịch vụ bảo mật mạng trực tuyến (Yes, No, No internet service)
    "OnlineBackup",     # Dịch vụ sao lưu dữ liệu đám mây (Yes, No, No internet service)
    "DeviceProtection", # Dịch vụ bảo hiểm thiết bị (Yes, No, No internet service)
    "TechSupport",      # Dịch vụ hỗ trợ kỹ thuật chuyên dụng (Yes, No, No internet service)
    "StreamingTV",      # Dịch vụ truyền hình trực tuyến (Yes, No, No internet service)
    "StreamingMovies",  # Dịch vụ xem phim trực tuyến (Yes, No, No internet service)
    
    # Nhóm điều khoản tài chính & thanh toán:
    "Contract",         # Loại hợp đồng (Month-to-month, One year, Two year)
    "PaperlessBilling", # Nhận hóa đơn điện tử không giấy (Yes / No)
    "PaymentMethod",    # Phương thức thanh toán (Electronic check, Mailed check, Bank transfer, Credit card)
]


def churn_rate_by_category(df: pd.DataFrame) -> dict:
    """
    Tính toán tỷ lệ phần trăm khách hàng rời bỏ (Yes) và ở lại (No) trong từng nhóm danh mục.
    
    Quy trình xử lý:
        1. Tạo bản sao dữ liệu.
        2. Chuyển đổi giá trị cột `SeniorCitizen` từ nhị phân (0, 1) thành chuỗi nhãn ('No', 'Yes')
           để thống nhất và trực quan khi vẽ biểu đồ.
        3. Với mỗi thuộc tính trong danh sách CATEGORICAL:
           - Lập bảng chéo (crosstab) đếm số lượng khách hàng theo từng danh mục và nhãn Churn.
           - Chuẩn hóa thành tỷ lệ phần trăm (%) theo từng hàng (tổng tỷ lệ của mỗi hàng = 100%).
           - Sắp xếp giảm dần theo cột 'Yes' (nhóm có tỷ lệ rời bỏ cao nhất sẽ được đưa lên đầu danh sách).
           - Trích xuất danh sách nhãn (categories), tỷ lệ churn (churn_yes), tỷ lệ ở lại (churn_no),
             và tổng số khách hàng thực tế trong từng nhóm (customers).
             
    Tham số:
        df (pd.DataFrame): Bảng dữ liệu gốc.
        
    Trả về:
        dict: Cấu trúc từ điển dạng JSON, thuận tiện để truyền trực tiếp vào Chart.js:
            {
                "Contract": {
                    "categories": ["Month-to-month", "One year", "Two year"],
                    "churn_yes": [42.71, 11.27, 2.83],
                    "churn_no": [57.29, 88.73, 97.17],
                    "customers": [3875, 1473, 1695]
                },
                ...
            }
    """
    d = df.copy()
    
    # Chuẩn hóa biến nhị phân số sang nhãn chữ thân thiện
    d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes"})

    result = {}
    for col in CATEGORICAL:
        # Bảng đếm tần số khách hàng giữa nhóm thuộc tính và trạng thái Churn
        counts = pd.crosstab(d[col], d["Churn"])
        
        # Tính tỷ lệ phần trăm trên tổng số khách của mỗi nhóm (chia theo trục hàng axis=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Sắp xếp nhóm có tỷ lệ Churn = Yes cao nhất lên trên đầu để tạo điểm nhấn
        pct = pct.sort_values("Yes", ascending=False)
        
        # Đóng gói dữ liệu biểu đồ cho thuộc tính col
        result[col] = {
            "categories": [str(c) for c in pct.index],
            "churn_yes": pct["Yes"].round(2).tolist(),
            "churn_no": pct["No"].round(2).tolist(),
            # Tổng số khách trong nhóm tương ứng với thứ tự sắp xếp của index
            "customers": [int(counts.loc[c].sum()) for c in pct.index],
        }
        
    return result
