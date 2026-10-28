"""Huấn luyện, so sánh mô hình dự đoán churn và xác định các yếu tố liên quan."""
import sys
from functools import lru_cache
from pathlib import Path

# Đảm bảo import được khi chạy trực tiếp file từ IDE (VS Code) hoặc dòng lệnh
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import clean
from src.data_loader import load_raw

RANDOM_STATE = 42
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(clf, cat_cols: list) -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def _pretty(name: str) -> str:
    """'cat__Contract_Two year' -> 'Contract = Two year'."""
    name = name.split("__", 1)[1]
    for col in ["Contract", "InternetService", "PaymentMethod", "gender"]:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    col, _, level = name.rpartition("_")
    return f"{col} = {level}" if col else name


CACHE_FILE = ROOT_DIR / "src" / "model_cache.json"


@lru_cache(maxsize=1)
def run(use_cache: bool = True) -> dict:
    """Chạy toàn bộ quy trình (hoặc nạp tức thì từ cache nếu đã tính sẵn)."""
    if use_cache and CACHE_FILE.exists():
        try:
            import json
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    df = clean(load_raw())
    X = df.drop(columns=["customerID", "Churn"])
    y = (df["Churn"] == "Yes").astype(int)
    cat_cols = [c for c in X.columns if c not in NUM_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models, results = {}, []
    for name, clf in classifiers.items():
        pipe = build_pipeline(clf, cat_cols)
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        models[name] = pipe
        results.append({
            "model": name,
            "cv_auc": round(cv_auc.mean(), 4),
            "cv_auc_std": round(cv_auc.std(), 4),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    # Baseline: luôn đoán "không rời bỏ"
    baseline_acc = round(1 - y_test.mean(), 4)

    # Logistic Regression: hệ số -> odds ratio
    lr = models["Logistic Regression"]
    names = [_pretty(n) for n in lr.named_steps["prep"].get_feature_names_out()]
    coefs = lr.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": names, "coef": coefs})
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
    top_coef = coef_df.head(15).round(4).to_dict("records")

    # Permutation importance (theo từng biến gốc) của mô hình có ROC-AUC cross-validation cao nhất
    # (chọn theo CV, không chọn theo tập test để tránh thiên lệch)
    best_name = max(results, key=lambda r: r["cv_auc"])["model"]
    perm = permutation_importance(
        models[best_name], X_test, y_test, scoring="roc_auc",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False)
    top_perm = perm_df.head(10).round(4).to_dict("records")

    out = {
        "n_train": len(X_train), "n_test": len(X_test),
        "baseline_acc": baseline_acc,
        "results": results,
        "best_name": best_name,
        "top_coef": top_coef,
        "top_perm": top_perm,
    }

    try:
        import json
        CACHE_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return out


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    out = run()
    print(pd.DataFrame(out["results"]).to_string(index=False))
    print("\nBaseline accuracy:", out["baseline_acc"])
    print("\nTop hệ số Logistic Regression:")
    print(pd.DataFrame(out["top_coef"]).to_string(index=False))
    print(f"\nPermutation importance ({out['best_name']}):")
    print(pd.DataFrame(out["top_perm"]).to_string(index=False))