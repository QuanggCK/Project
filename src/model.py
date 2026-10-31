"""
===============================================================================
BƯỚC 3: XÂY DỰNG, HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH DỰ ĐOÁN (MACHINE LEARNING)
===============================================================================
Mục đích module:
    - Xây dựng quy trình học máy hoàn chỉnh (Machine Learning Pipeline) bao gồm:
        + Tiền xử lý dữ liệu: Chuẩn hóa biến số (StandardScaler) và mã hóa biến phân loại (OneHotEncoder).
        + Chia tập dữ liệu phân tầng (Stratified Train/Test Split) theo tỷ lệ 80/20.
        + Huấn luyện và so sánh 3 mô hình phân loại phổ biến:
            1. Logistic Regression (Mô hình tuyến tính cổ điển, diễn giải tốt qua hệ số).
            2. Random Forest (Mô hình ensemble dạng bagging, chống overfitting).
            3. HistGradientBoosting (Mô hình boosting hiện đại, hiệu quả và tốc độ cao).
        + Kiểm định chéo (Stratified 5-Fold Cross Validation) trên tập huấn luyện để đo ROC-AUC.
        + Đánh giá toàn diện trên tập kiểm tra: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.
        + Phân tích hệ số hồi quy (Coefficients) và tỉ số Odds Ratio ($e^{\\beta}$) của Logistic Regression.
        + Tính toán độ quan trọng của biến bằng phương pháp Permutation Importance (xáo trộn đặc trưng).
        + Lưu kết quả vào bộ nhớ đệm `model_cache.json` để giao diện web phản hồi tức thì.
===============================================================================
"""

import sys
from functools import lru_cache
from pathlib import Path

# Đảm bảo đường dẫn gốc của dự án luôn có trong sys.path khi chạy từ bất kỳ đâu (VS Code, terminal, v.v.)
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

# Nhập hàm làm sạch và nạp dữ liệu từ các module nội bộ
from src.cleaning import clean
from src.data_loader import load_raw

# Khởi tạo hằng số để đảm bảo tính tái lập (reproducibility) của các thí nghiệm
RANDOM_STATE = 42

# Danh sách 3 biến định lượng liên tục cần chuẩn hóa tỷ lệ
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_pipeline(clf, cat_cols: list) -> Pipeline:
    """
    Xây dựng một Pipeline tích hợp tiền xử lý dữ liệu và thuật toán phân loại.
    
    Quy trình tiền xử lý (ColumnTransformer):
        - Biến số (NUM_COLS): Sử dụng StandardScaler để biến đổi về phân phối có trung bình = 0,
          độ lệch chuẩn = 1. Giúp các mô hình như Logistic Regression hội tụ nhanh và hệ số có thể so sánh được.
        - Biến phân loại (cat_cols): Sử dụng OneHotEncoder với `drop="first"` (bỏ bớt 1 cột danh mục cơ sở)
          để tránh hiện tượng đa cộng tuyến hoàn hảo (Dummy Variable Trap).
          
    Tham số:
        clf: Đối tượng mô hình phân loại (classifier của scikit-learn).
        cat_cols (list): Danh sách tên các cột phân loại cần mã hóa one-hot.
        
    Trả về:
        Pipeline: Đường ống học máy sẵn sàng cho việc fit và predict.
    """
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def _pretty(name: str) -> str:
    """
    Định dạng lại tên đặc trưng sau khi mã hóa One-Hot để hiển thị thân thiện trên giao diện web.
    Ví dụ: 'cat__Contract_Two year' -> 'Contract = Two year'
           'cat__PaymentMethod_Electronic check' -> 'PaymentMethod = Electronic check'
    """
    # Cắt bỏ tiền tố transformer 'cat__' hoặc 'num__'
    name = name.split("__", 1)[1]
    
    # Xử lý các cột phân loại có tiền tố cụ thể
    for col in ["Contract", "InternetService", "PaymentMethod", "gender"]:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
            
    col, _, level = name.rpartition("_")
    return f"{col} = {level}" if col else name


# Đường dẫn file JSON lưu bộ nhớ đệm kết quả mô hình đã huấn luyện
CACHE_FILE = ROOT_DIR / "src" / "model_cache.json"


@lru_cache(maxsize=1)
def run(use_cache: bool = True) -> dict:
    """
    Chạy toàn bộ chu trình huấn luyện, đánh giá và giải thích mô hình (Model Pipeline).
    
    Tối ưu hóa:
        - Sử dụng cơ chế cache file: Nếu file `model_cache.json` đã tồn tại và tham số `use_cache=True`,
          hàm sẽ đọc và trả về kết quả ngay lập tức (dưới 1ms), giúp giao diện web phản hồi cực nhanh.
        - Nếu chạy lần đầu hoặc không dùng cache, hàm sẽ thực hiện toàn bộ việc tính toán và ghi lại vào cache.
        
    Trả về:
        dict: Chứa các thông tin đầy đủ về:
            - Số lượng mẫu tập huấn luyện (n_train), kiểm tra (n_test).
            - Độ chính xác cơ sở (baseline_acc).
            - Bảng kết quả so sánh các chỉ số của 3 mô hình.
            - Tên mô hình tốt nhất theo điểm kiểm định chéo (best_name).
            - Top 15 hệ số Logistic Regression quan trọng nhất và Odds Ratio.
            - Top 10 đặc trưng có Permutation Importance cao nhất.
    """
    # 1. Đọc kết quả từ file cache nếu có
    if use_cache and CACHE_FILE.exists():
        try:
            import json
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 2. Tải dữ liệu và làm sạch
    df = clean(load_raw())
    
    # Tách biến đặc trưng (X) và biến mục tiêu (y)
    # Loại bỏ customerID vì là mã định danh cá nhân không mang giá trị quy luật dự đoán
    X = df.drop(columns=["customerID", "Churn"])
    # Chuyển đổi nhãn Churn ("Yes"/"No") thành số nhị phân (1/0)
    y = (df["Churn"] == "Yes").astype(int)
    
    # Xác định danh sách các cột phân loại (tất cả các cột trừ 3 cột số liên tục)
    cat_cols = [c for c in X.columns if c not in NUM_COLS]

    # 3. Phân chia tập dữ liệu huấn luyện (80%) và kiểm tra (20%)
    # `stratify=y` đảm bảo tỷ lệ nhãn Churn ở cả 2 tập tương đồng với tỷ lệ gốc trong toàn bộ dữ liệu (~26.5% Yes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 4. Khởi tạo danh sách 3 mô hình học máy cần so sánh
    # Đều thiết lập class_weight="balanced" để tự động điều chỉnh trọng số phạt đối với lớp thiểu số (Churn = 1)
    classifiers = {
        # Mô hình 1: Logistic Regression - Cổ điển, giải thích rõ phương hướng ảnh hưởng qua hệ số beta
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        # Mô hình 2: Random Forest - Tập hợp cây quyết định phân tán, tránh overfitting
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        # Mô hình 3: Gradient Boosting - Tăng cường độ chính xác liên tiếp bằng cây quyết định dựa trên histogram
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    # Thiết lập kiểm định chéo phân tầng 5 phần (5-fold Stratified Cross Validation)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models, results = {}, []

    # 5. Huấn luyện và đánh giá từng mô hình
    for name, clf in classifiers.items():
        pipe = build_pipeline(clf, cat_cols)
        
        # Đánh giá độ ổn định và khái quát hóa trên tập huấn luyện thông qua 5-fold CV ROC-AUC
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        
        # Huấn luyện mô hình hoàn chỉnh trên toàn bộ tập Train
        pipe.fit(X_train, y_train)
        
        # Dự đoán nhãn lớp (0 hoặc 1) và xác suất dự đoán (P(Churn=1)) trên tập Test
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        
        # Tính toán các giá trị của Ma trận nhầm lẫn (Confusion Matrix):
        # TN: Thực tế ở lại, Đoán ở lại
        # FP: Thực tế ở lại, Báo động nhầm là rời bỏ
        # FN: Thực tế rời bỏ, Bỏ sót là ở lại
        # TP: Thực tế rời bỏ, Dự đoán chính xác rời bỏ
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        
        models[name] = pipe
        
        # Tổng hợp các chỉ số đo lường hiệu năng
        results.append({
            "model": name,
            "cv_auc": round(cv_auc.mean(), 4),          # Điểm ROC-AUC trung bình qua 5 folds
            "cv_auc_std": round(cv_auc.std(), 4),        # Độ lệch chuẩn của ROC-AUC giữa các folds
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    # 6. Tính Baseline Accuracy: Độ chính xác nếu một mô hình ngây thơ luôn đoán khách hàng "Ở lại" (lớp đa số)
    baseline_acc = round(1 - y_test.mean(), 4)

    # 7. Trích xuất và giải thích Hệ số hồi quy (Coefficients) của mô hình Logistic Regression
    lr = models["Logistic Regression"]
    # Lấy danh sách tên đặc trưng sau khi qua One-Hot Encoder và chuẩn hóa
    names = [_pretty(n) for n in lr.named_steps["prep"].get_feature_names_out()]
    coefs = lr.named_steps["clf"].coef_[0]
    
    coef_df = pd.DataFrame({"feature": names, "coef": coefs})
    # Tính tỉ số khả dĩ Odds Ratio: OR = exp(coef)
    # OR > 1: Làm tăng khả năng rời bỏ; OR < 1: Làm giảm khả năng rời bỏ
    coef_df["odds_ratio"] = np.exp(coef_df["coef"])
    # Sắp xếp danh sách theo giá trị tuyệt đối của hệ số giảm dần (ảnh hưởng mạnh nhất lên đầu)
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
    top_coef = coef_df.head(15).round(4).to_dict("records")

    # 8. Tính toán Độ quan trọng của biến bằng Permutation Importance trên tập Test
    # Chọn mô hình có điểm CV ROC-AUC cao nhất trên tập huấn luyện để tránh thiên lệch chọn lựa
    best_name = max(results, key=lambda r: r["cv_auc"])["model"]
    perm = permutation_importance(
        models[best_name], X_test, y_test, scoring="roc_auc",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False)
    top_perm = perm_df.head(10).round(4).to_dict("records")

    # 9. Đóng gói toàn bộ kết quả phân tích
    out = {
        "n_train": len(X_train), 
        "n_test": len(X_test),
        "baseline_acc": baseline_acc,
        "results": results,
        "best_name": best_name,
        "top_coef": top_coef,
        "top_perm": top_perm,
    }

    # Ghi đè vào bộ nhớ đệm JSON để phục vụ cho các lần tải trang kế tiếp
    try:
        import json
        CACHE_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return out


if __name__ == "__main__":
    # Đảm bảo console in đúng định dạng tiếng Việt UTF-8
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("--- HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH DỰ ĐOÁN CHURN ---")
    out = run(use_cache=False)  # Buộc chạy tính toán lại để kiểm tra
    print("\nKết quả so sánh các mô hình:")
    print(pd.DataFrame(out["results"]).to_string(index=False))
    print(f"\nĐộ chính xác mô hình cơ sở (Baseline Accuracy): {out['baseline_acc']}")
    print("\nTop 15 đặc trưng có ảnh hưởng mạnh nhất (Hệ số Logistic Regression):")
    print(pd.DataFrame(out["top_coef"]).to_string(index=False))
    print(f"\nTop 10 đặc trưng quan trọng nhất theo Permutation Importance ({out['best_name']}):")
    print(pd.DataFrame(out["top_perm"]).to_string(index=False))
