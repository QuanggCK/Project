"""
===============================================================================
BƯỚC 1: ĐỌC VÀ KHÁM PHÁ CẤU TRÚC DỮ LIỆU BAN ĐẦU (DATA LOADING & EXPLORATION)
===============================================================================
Mục đích module:
    - Định vị và tải tập dữ liệu gốc Telco Customer Churn từ file CSV.
    - Cung cấp hàm kiểm tra nhanh tình trạng chất lượng dữ liệu:
        + Kích thước bảng (số dòng, số cột).
        + Kiểm tra khóa chính `customerID` có bị trùng lặp không.
        + Phát hiện dữ liệu thiếu thông thường (NaN/null).
        + Phát hiện dữ liệu thiếu "ẩn" (các ô chuỗi chỉ chứa ký tự khoảng trắng ' ').
        + Thống kê tỷ lệ phân bố của biến mục tiêu `Churn`.
        + Trích xuất dữ liệu mẫu đầu tiên.
===============================================================================
"""

from pathlib import Path
import pandas as pd

# Xác định đường dẫn gốc của dự án (thư mục cha của src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn tuyệt đối dẫn tới file dữ liệu Telco Customer Churn
DATA_PATH = BASE_DIR / "dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Đọc dữ liệu thô từ file CSV vào DataFrame của Pandas.
    
    Tham số:
        path (Path): Đường dẫn tới file CSV dữ liệu. Mặc định dùng DATA_PATH.
        
    Trả về:
        pd.DataFrame: Toàn bộ bảng dữ liệu gốc chưa qua xử lý hay làm sạch.
    """
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> dict:
    """
    Tổng hợp các thông số kỹ thuật và chất lượng của bảng dữ liệu.
    
    Các bước phân tích:
        1. Đếm ô trống ẩn: Một số cột (như TotalCharges) chứa chuỗi khoảng trắng ' ' 
           khiến hàm isna() mặc định không nhận diện được. Ta cần strip() chuỗi 
           để đếm chính xác số ô bị bỏ trống.
        2. Thống kê cột: Tên cột, kiểu dữ liệu (dtype), số giá trị thiếu (NaN), 
           số ô trống (blank), và số giá trị phân biệt (unique values).
        3. Phân bố biến nhãn Churn: Số lượng và tỷ lệ phần trăm khách hàng rời bỏ (Yes/No).
        4. Trích 10 dòng đầu tiên làm mẫu trực quan.
        
    Tham số:
        df (pd.DataFrame): DataFrame cần kiểm tra.
        
    Trả về:
        dict: Chứa các thông tin thống kê:
            - "n_rows": Tổng số bản ghi (dòng).
            - "n_cols": Tổng số đặc trưng (cột).
            - "duplicated_ids": Số lượng mã khách hàng (customerID) bị trùng.
            - "columns": DataFrame chi tiết về từng cột.
            - "churn": DataFrame số lượng và phần trăm của từng lớp Churn.
            - "head": DataFrame 10 dòng đầu.
    """
    # Bước 1: Quét tất cả các cột dạng chuỗi (object/string) để đếm các ô chỉ có khoảng trắng
    blank_counts = {
        col: int((df[col].astype(str).str.strip() == "").sum())
        for col in df.select_dtypes(include=["object", "string"]).columns
    }
    # Chỉ giữ lại những cột thực sự có xuất hiện ô trống
    blank_counts = {col: n for col, n in blank_counts.items() if n > 0}

    # Bước 2: Tạo bảng tổng hợp thông số cấu trúc cho toàn bộ các cột
    columns = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing": df.isna().sum().values,
            # Lấy số lượng ô trống ẩn đã đếm ở trên (nếu không có thì trả về 0)
            "blank": [blank_counts.get(c, 0) for c in df.columns],
            "unique": df.nunique().values,
        }
    )

    # Bước 3: Thống kê số lượng và tỷ lệ % của biến mục tiêu Churn
    churn_counts = df["Churn"].value_counts()
    churn_pct = (df["Churn"].value_counts(normalize=True) * 100).round(2)

    # Bước 4: Đóng gói kết quả vào dictionary trả về
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "duplicated_ids": int(df["customerID"].duplicated().sum()),
        "columns": columns,
        "churn": pd.DataFrame({"count": churn_counts, "percent": churn_pct}),
        "head": df.head(10),
    }


if __name__ == "__main__":
    # Đảm bảo console Windows in đúng các ký tự tiếng Việt có dấu
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        
    # Chạy thử trực tiếp module này từ dòng lệnh để kiểm tra dữ liệu
    print("--- KIỂM TRA DỮ LIỆU TELCO CHURN ---")
    data = load_raw()
    info = summarize(data)
    
    print(f"Kích thước tập dữ liệu: {info['n_rows']} dòng x {info['n_cols']} cột")
    print(f"Số lượng customerID bị trùng lặp: {info['duplicated_ids']}\n")
    print("Chi tiết các cột dữ liệu:")
    print(info["columns"].to_string(index=False))
    print("\nPhân bố biến mục tiêu (Churn):")
    print(info["churn"])
