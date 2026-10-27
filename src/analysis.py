"""Bước 2: Tính tỷ lệ churn theo từng nhóm của các biến phân loại."""
import pandas as pd

CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def churn_rate_by_category(df: pd.DataFrame) -> dict:
    """Với mỗi biến, trả về % khách rời bỏ (Yes) và ở lại (No) trong từng nhóm."""
    d = df.copy()
    d["SeniorCitizen"] = d["SeniorCitizen"].map({0: "No", 1: "Yes"})

    result = {}
    for col in CATEGORICAL:
        counts = pd.crosstab(d[col], d["Churn"])
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        pct = pct.sort_values("Yes", ascending=False)  # nhóm churn cao nhất lên đầu
        result[col] = {
            "categories": [str(c) for c in pct.index],
            "churn_yes": pct["Yes"].round(2).tolist(),
            "churn_no": pct["No"].round(2).tolist(),
            "customers": [int(counts.loc[c].sum()) for c in pct.index],
        }
    return result
