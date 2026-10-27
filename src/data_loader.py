"""Bước 1: Đọc dữ liệu Telco Customer Churn và tóm tắt cấu trúc."""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR /"dataset"/"WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Đọc file CSV gốc, giữ nguyên dữ liệu (chưa làm sạch)."""
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> dict:
    """Tóm tắt nhanh dữ liệu: kích thước, kiểu dữ liệu, giá trị thiếu, phân bố """
    # TotalCharges đang là chuỗi, có ô trống " " mà isna() không bắt được.
    blank_counts = {
        col: int((df[col].astype(str).str.strip() == "").sum())
        for col in df.select_dtypes(include=["object", "string"]).columns
    }
    blank_counts = {col: n for col, n in blank_counts.items() if n > 0}

    columns = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing": df.isna().sum().values,
            "blank": [blank_counts.get(c, 0) for c in df.columns],
            "unique": df.nunique().values,
        }
    )

    churn_counts = df["Churn"].value_counts()
    churn_pct = (df["Churn"].value_counts(normalize=True) * 100).round(2)

    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "duplicated_ids": int(df["customerID"].duplicated().sum()),
        "columns": columns,
        "churn": pd.DataFrame({"count": churn_counts, "percent": churn_pct}),
        "head": df.head(10),
    }

if __name__ == "__main__":
    data = load_raw()
    info = summarize(data)
    print(f"Kích thước: {info['n_rows']} dòng x {info['n_cols']} cột")
    print(f"customerID trùng: {info['duplicated_ids']}\n")
    print(info["columns"].to_string(index=False))
    print("\nPhân bố Churn:")
    print(info["churn"])
