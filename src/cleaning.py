"""Làm sạch và tiền xử lý dữ liệu Telco Customer Churn."""
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch dữ liệu:
    - Ép kiểu TotalCharges từ chuỗi sang số thực.
    - Xử lý các ô trống ' ' thành 0.0 (khách hàng mới có tenure = 0 chưa phát sinh phí).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)
    return df
