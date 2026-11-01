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

"""Phân tích các yếu tố liên quan đến Customer Churn cho Telco dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

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
NUMERICAL = ["tenure", "MonthlyCharges", "TotalCharges"]
TENURE_BINS = [-1, 6, 12, 24, 48, 60, 72]
TENURE_LABELS = ["0–6 tháng", "7–12 tháng", "13–24 tháng",
                 "25–48 tháng", "49–60 tháng", "61–72 tháng"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa bản sao dữ liệu, không làm thay đổi DataFrame đầu vào."""
    d = df.copy()
    if "TotalCharges" in d:
        d["TotalCharges"] = pd.to_numeric(
            d["TotalCharges"].replace(r"^\s*$", np.nan, regex=True),
            errors="coerce"
        )
    if "SeniorCitizen" in d:
        d["SeniorCitizen"] = d["SeniorCitizen"].map(
            {0: "No", 1: "Yes", "0": "No", "1": "Yes"}
        ).fillna(d["SeniorCitizen"].astype(str))
    if "Churn" in d:
        d["Churn"] = d["Churn"].astype(str).str.strip()
    return d


def _safe_rate(series: pd.Series) -> float:
    return round(float(series.mean() * 100), 2) if len(series) else 0.0


def churn_rate_by_category(df: pd.DataFrame) -> dict:
    """Tỷ lệ Churn Yes/No, số lượng và chênh lệch so với toàn bộ dataset."""
    d = _prepare(df)
    overall = _safe_rate(d["Churn"].eq("Yes"))
    result = {}
    for col in CATEGORICAL:
        if col not in d.columns:
            continue
        valid = d[[col, "Churn"]].dropna()
        counts = pd.crosstab(valid[col].astype(str), valid["Churn"])
        for label in ("Yes", "No"):
            if label not in counts:
                counts[label] = 0
        pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        pct = pct.sort_values("Yes", ascending=False)
        result[col] = {
            "categories": [str(x) for x in pct.index],
            "churn_yes": pct["Yes"].round(2).tolist(),
            "churn_no": pct["No"].round(2).tolist(),
            "customers": [int(counts.loc[x].sum()) for x in pct.index],
            "difference": (pct["Yes"] - overall).round(2).tolist(),
            "overall_rate": overall,
        }
        
    return result


def churn_rate_by_tenure(df: pd.DataFrame) -> dict:
    d = _prepare(df)
    d["tenure_group"] = pd.cut(
        pd.to_numeric(d["tenure"], errors="coerce"),
        bins=TENURE_BINS, labels=TENURE_LABELS
    )
    grouped = d.dropna(subset=["tenure_group"]).groupby(
        "tenure_group", observed=False
    )["Churn"].agg(
        customers="size",
        churned=lambda x: int(x.eq("Yes").sum()),
        churn_rate=lambda x: _safe_rate(x.eq("Yes"))
    ).reset_index()
    return {
        "labels": grouped["tenure_group"].astype(str).tolist(),
        "rates": grouped["churn_rate"].astype(float).tolist(),
        "customers": grouped["customers"].astype(int).tolist(),
        "churned": grouped["churned"].astype(int).tolist(),
    }


def numerical_analysis(df: pd.DataFrame) -> dict:
    """Thống kê mô tả từng biến số theo hai nhóm Churn."""
    d = _prepare(df)
    result = {}
    for col in NUMERICAL:
        if col not in d:
            continue
        d[col] = pd.to_numeric(d[col], errors="coerce")
        groups = {}
        for churn_value in ("Yes", "No"):
            s = d.loc[d["Churn"].eq(churn_value), col].dropna()
            groups[churn_value] = {
                "count": int(s.count()),
                "mean": round(float(s.mean()), 2) if len(s) else None,
                "median": round(float(s.median()), 2) if len(s) else None,
                "std": round(float(s.std()), 2) if len(s) > 1 else None,
                "q1": round(float(s.quantile(.25)), 2) if len(s) else None,
                "q3": round(float(s.quantile(.75)), 2) if len(s) else None,
            }
        result[col] = groups
    return result


def correlation_analysis(df: pd.DataFrame) -> dict:
    """Pearson correlation giữa các biến số và Churn được mã hóa 0/1."""
    d = _prepare(df)
    cols = [c for c in NUMERICAL if c in d.columns]
    work = d[cols].apply(pd.to_numeric, errors="coerce")
    work["Churn"] = d["Churn"].map({"No": 0, "Yes": 1})
    corr = work.corr(method="pearson").replace([np.inf, -np.inf], np.nan).fillna(0)
    return {
        "labels": corr.columns.tolist(),
        "matrix": [[round(float(v), 3) for v in row] for row in corr.to_numpy()],
        "churn_correlations": [
            {"feature": c, "correlation": round(float(corr.loc[c, "Churn"]), 3)}
            for c in cols
        ],
    }


def _cramers_v(table: pd.DataFrame) -> float:
    """Cramér's V có hiệu chỉnh sai lệch mẫu nhỏ."""
    if table.empty or min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n <= 1:
        return 0.0
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, giữ nguyên thứ tự ban đầu."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(n)
    result[order] = adjusted
    return result.tolist()


def categorical_association(df: pd.DataFrame) -> dict:
    """Chi-square và Cramér's V cho từng biến phân loại với Churn."""
    d = _prepare(df)
    rows, p_values = [], []
    for col in CATEGORICAL:
        if col not in d:
            continue
        table = pd.crosstab(d[col].fillna("Missing").astype(str), d["Churn"])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(table)
        v = _cramers_v(table)
        rows.append({
            "feature": col,
            "chi2": round(float(chi2), 3),
            "p_value": float(p),
            "dof": int(dof),
            "cramers_v": round(v, 3),
        })
        p_values.append(float(p))
    adjusted = _bh_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = round(float(p_adj), 6)
        row["significant"] = bool(p_adj < 0.05)
    rows.sort(key=lambda x: x["cramers_v"], reverse=True)
    return {"results": rows, "alpha": 0.05}


def numerical_hypothesis_test(df: pd.DataFrame) -> dict:
    """Mann-Whitney U test so sánh phân phối biến số giữa Churn Yes/No."""
    d = _prepare(df)
    rows, p_values = [], []
    for col in NUMERICAL:
        if col not in d:
            continue
        yes = pd.to_numeric(d.loc[d["Churn"].eq("Yes"), col], errors="coerce").dropna()
        no = pd.to_numeric(d.loc[d["Churn"].eq("No"), col], errors="coerce").dropna()
        if len(yes) == 0 or len(no) == 0:
            continue
        stat, p = mannwhitneyu(yes, no, alternative="two-sided")
        rows.append({
            "feature": col,
            "yes_median": round(float(yes.median()), 2),
            "no_median": round(float(no.median()), 2),
            "u_statistic": round(float(stat), 2),
            "p_value": float(p),
        })
        p_values.append(float(p))
    adjusted = _bh_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["p_adjusted"] = round(float(p_adj), 6)
        row["significant"] = bool(p_adj < 0.05)
    return {"results": rows, "alpha": 0.05}


def churn_profile(df: pd.DataFrame) -> dict:
    """So sánh số lượng, tỷ lệ và trung bình biến số giữa Yes/No."""
    d = _prepare(df)
    result = {}
    for value in ("Yes", "No"):
        subset = d[d["Churn"].eq(value)]
        numeric = {}
        for col in NUMERICAL:
            if col in subset:
                s = pd.to_numeric(subset[col], errors="coerce").dropna()
                numeric[col] = round(float(s.mean()), 2) if len(s) else None
        result[value] = {
            "customers": int(len(subset)),
            "share": round(len(subset) / len(d) * 100, 2) if len(d) else 0,
            "numeric_means": numeric,
        }
    return result


def segment_analysis(df: pd.DataFrame) -> dict:
    """Phân tích các phân khúc kết hợp Contract và InternetService."""
    d = _prepare(df)
    required = ["Contract", "InternetService", "Churn"]
    if not all(c in d.columns for c in required):
        return {"labels": [], "rates": [], "customers": [], "rows": []}
    grouped = d.groupby(
        ["Contract", "InternetService"], dropna=False
    )["Churn"].agg(
        customers="size",
        churned=lambda x: int(x.eq("Yes").sum()),
        churn_rate=lambda x: _safe_rate(x.eq("Yes"))
    ).reset_index()
    grouped = grouped.sort_values("churn_rate", ascending=False)
    rows = [{
        "contract": str(r.Contract),
        "internet": str(r.InternetService),
        "customers": int(r.customers),
        "churned": int(r.churned),
        "churn_rate": float(r.churn_rate),
    } for r in grouped.itertuples(index=False)]
    return {
        "labels": [f"{r['contract']} / {r['internet']}" for r in rows],
        "rates": [r["churn_rate"] for r in rows],
        "customers": [r["customers"] for r in rows],
        "rows": rows,
    }
