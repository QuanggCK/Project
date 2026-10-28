# Phân Tích & Dự Đoán Khách Hàng Rời Bỏ Dịch Vụ Viễn Thông (Telco Customer Churn)

Dự án môn học **Khoa học Dữ liệu (Data Science)** nhằm phân tích hành vi khách hàng, xác định các yếu tố rủi ro dẫn đến việc hủy dịch vụ (Churn), và xây dựng các mô hình học máy (Machine Learning) để dự đoán nguy cơ rời bỏ của khách hàng dịch vụ viễn thông.

---

## 📌 Tổng Quan & Câu Hỏi Nghiệp Vụ
- **Bài toán:** Trong ngành viễn thông, chi phí thu hút khách hàng mới cao hơn nhiều so với chi phí giữ chân khách hàng hiện tại. Việc chủ động nhận diện sớm các khách hàng có xu hướng rời mạng giúp doanh nghiệp triển khai các chính sách chăm sóc kịp thời.
- **Câu hỏi cốt lõi:**
  1. *Những yếu tố nào liên quan chặt chẽ nhất đến việc khách hàng rời bỏ dịch vụ?*
  2. *Làm thế nào để xây dựng mô hình dự đoán chính xác khách hàng có nguy cơ rời bỏ mà không bị đánh lừa bởi sự mất cân bằng dữ liệu?*
- **Bộ dữ liệu:** [Telco Customer Churn](dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv) gồm **7.043 dòng** và **21 cột** ghi nhận thông tin nhân khẩu học, các gói dịch vụ đăng ký, chi tiết hợp đồng và cước phí.

---

## 🔬 Quy Trình Áp Dụng Từng Bước Khoa Học Dữ Liệu (Data Science Lifecycle)

Dự án tuân thủ nghiêm ngặt quy trình chuẩn Khoa học Dữ liệu, được tổ chức thành các mô-đun rõ ràng:

```
[1. Data Understanding] ──► [2. Data Cleaning] ──► [3. Exploratory Analysis]
           │                         │                           │
           └─────────────────────────┼───────────────────────────┘
                                     ▼
                          [4. Machine Learning]
                       (Pipeline + CV + Importance)
                                     ▼
                          [5. Web Deployment]
                       (Interactive Flask App)
```

---

### Bước 1: Khám Phá & Đánh Giá Chất Lượng Dữ Liệu Ban Đầu (Data Understanding)
- **Mã nguồn:** [`src/data_loader.py`](src/data_loader.py) | **Giao diện:** Route `/`
- **Công việc thực hiện:**
  - Đọc dữ liệu gốc bằng pandas qua hàm `load_raw()`.
  - Tóm tắt kích thước dữ liệu (7.043 khách hàng $\times$ 21 thuộc tính).
  - Kiểm tra tính duy nhất của khóa chính: cột `customerID` có 0 giá trị trùng lặp.
  - Phân tích kiểu dữ liệu (`dtypes`) và phát hiện dữ liệu thiếu tiềm ẩn:
    - Phát hiện cột `TotalCharges` có kiểu dữ liệu là chuỗi (`object`) thay vì số thực, chứa **11 ô chỉ có khoảng trắng (`" "`)** mà hàm `isna()` mặc định không bắt được.
  - Phân tích phân bố biến mục tiêu `Churn`:
    - Nhóm ở lại (`No`): **5.174 khách (73,46%)**
    - Nhóm rời bỏ (`Yes`): **1.869 khách (26,54%)**
- **Mục đích:**
  - Nhận diện các vấn đề về chất lượng dữ liệu để định hướng bước làm sạch.
  - Xác định đây là bài toán **dữ liệu mất cân bằng (Imbalanced Classification)**, từ đó thiết lập chiến lược đánh giá mô hình phù hợp (không chỉ dùng `Accuracy`).

---

### Bước 2: Tiền Xử Lý & Làm Sạch Dữ Liệu (Data Cleaning & Preprocessing)
- **Mã nguồn:** [`src/cleaning.py`](src/cleaning.py)
- **Công việc thực hiện:**
  - Ép kiểu cột `TotalCharges` từ kiểu chuỗi (`object`) sang số thực (`float64`) thông qua `pd.to_numeric(..., errors='coerce')`.
  - Phân tích ngữ cảnh nghiệp vụ: Tất cả 11 khách hàng có `TotalCharges` bị trống đều có thời gian gắn bó `tenure = 0` (khách hàng mới ký hợp đồng trong tháng, chưa kết thúc chu kỳ tính cước đầu tiên).
  - Điền giá trị `0.0` thay vì xóa bỏ dòng dữ liệu, nhằm bảo toàn cỡ mẫu và phản ánh đúng thực tế thanh toán.
- **Mục đích:**
  - Đảm bảo dữ liệu chuẩn hóa về mặt toán học, sẵn sàng cho các phép biến đổi đặc trưng và thuật toán học máy mà không gây lỗi runtime (`NaN` hoặc sai kiểu dữ liệu).

---

### Bước 3: Phân Tích Đơn Biến & Hai Biến (Exploratory Data Analysis - EDA)
- **Mã nguồn:** [`src/analysis.py`](src/analysis.py) | **Giao diện:** Route `/factors`
- **Công việc thực hiện:**
  - Phân tích mối quan hệ giữa 16 biến phân loại (nhân khẩu học, dịch vụ viễn thông, hình thức thanh toán, hợp đồng) với tỷ lệ rời bỏ (`Churn`).
  - Lập bảng chéo tỷ lệ (`crosstab`) tính toán tỷ lệ % rời bỏ và quy mô mẫu của từng phân khúc khách hàng.
  - Sắp xếp và trực quan hóa tương tác trên biểu đồ thanh (stacked bar chart với Chart.js).
- **Mục đích:**
  - Tìm ra các giả thuyết kinh doanh ban đầu và các phân khúc rủi ro cao:
    - **Loại hợp đồng (`Contract`):** Khách hàng hợp đồng theo tháng (*Month-to-month*) có tỷ lệ rời mạng lên tới **42,71%**, trong khi khách hợp đồng 2 năm (*Two year*) chỉ có **2,83%**.
    - **Dịch vụ Internet (`InternetService`):** Khách hàng dùng Cáp quang (*Fiber optic*) rời bỏ cao bất thường (**41,89%** so với DSL chỉ **18,96%**).
    - **Phương thức thanh toán (`PaymentMethod`):** Khách hàng thanh toán qua séc điện tử (*Electronic check*) rời mạng cao nhất (**45,29%**).

---

### Bước 4: Xây Dựng Mô Hình, Đánh Giá & Giải Thích (Modeling & Explainability)
- **Mã nguồn:** [`src/model.py`](src/model.py) | **Giao diện:** Route `/model`
- **Công việc thực hiện:**
  1. **Thiết kế Pipeline chống rò rỉ dữ liệu (Preventing Data Leakage):**
     - Dùng `ColumnTransformer` tích hợp vào `Pipeline` của scikit-learn:
       - Chuẩn hóa biến số (`StandardScaler`): `tenure`, `MonthlyCharges`, `TotalCharges`.
       - Mã hóa biến phân loại (`OneHotEncoder(drop='first')`): các cột còn lại.
     - Các phép tính thống kê (mean, std, dummy categories) chỉ học trên tập huấn luyện của từng fold, không rò rỉ sang tập kiểm tra.
  2. **Chiến lược Thẩm định (Validation Strategy):**
     - Chia tập dữ liệu Stratified Train/Test (80% huấn luyện, 20% kiểm tra).
     - Áp dụng **Stratified 5-Fold Cross-Validation** trên tập huấn luyện để đánh giá độ ổn định của các mô hình.
  3. **Huấn luyện & So sánh Đa Mô hình:**
     - Huấn luyện 3 lớp thuật toán đại diện:
       - **Logistic Regression:** Mô hình tuyến tính, khả năng giải thích hệ số cao.
       - **Random Forest:** Mô hình phi tuyến tính dạng quần thể (Bagging ensemble) với 300 cây quyết định.
       - **Histogram Gradient Boosting:** Thuật toán Boosting hiện đại tối ưu cho dữ liệu bảng.
     - Kích hoạt `class_weight='balanced'` trên tất cả các mô hình để bù trừ cho tỷ lệ mất cân bằng dữ liệu.
  4. **Thước đo Đánh giá Đa chiều:**
     - Thiết lập Dummy Baseline (luôn dự đoán không rời bỏ) đạt Accuracy $73,46\%$, chứng minh Accuracy không phải thước đo đáng tin cậy.
     - Sử dụng **ROC-AUC** làm thước đo xếp hạng chính, kết hợp **Precision, Recall, F1-Score** và **Ma trận nhầm lẫn (Confusion Matrix)**.
  5. **Giải Thích Mô Hình & Đo Lường Tầm Quan Trọng (Explainability):**
     - **Odds Ratio từ Logistic Regression:** Tính $e^\beta$ để xác định đặc trưng nào làm tăng/giảm tỷ số số chênh rời bỏ khi các yếu tố khác giữ nguyên.
     - **Permutation Importance:** Đo lường mức độ suy giảm ROC-AUC trên tập kiểm tra khi xáo trộn ngẫu nhiên từng biến gốc của mô hình tốt nhất (tránh thiên lệch của impurity-based feature importance).
- **Mục đích:**
  - Chọn ra mô hình có năng lực tổng quát hóa cao nhất.
  - Cung cấp hiểu biết có thể hành động được (Actionable Insights) cho ban quản trị doanh nghiệp thay vì một mô hình "hộp đen" (black box).

---

### Bước 5: Triển Khai Ứng Dụng Web Báo Cáo Tương Tác (Deployment)
- **Mã nguồn:** [`app.py`](app.py), [`templates/`](templates/), [`static/`](static/)
- **Công việc thực hiện:**
  - Xây dựng web app bằng Flask với 3 trang tương tác:
    - `/`: Dashboard tổng quan dữ liệu và kiểm tra cấu trúc cột.
    - `/factors`: Khám phá động tỷ lệ Churn theo từng biến với Chart.js và diễn giải tự động.
    - `/model`: Bảng xếp hạng mô hình, ma trận nhầm lẫn, biểu đồ hệ số Odds Ratio và biểu đồ Permutation Importance.
  - Sử dụng `@lru_cache` để lưu kết quả huấn luyện mô hình, giúp web phản hồi tức thì cho các lượt truy cập sau.
- **Mục đích:**
  - Trực quan hóa kết quả nghiên cứu khoa học dữ liệu thành công cụ trực quan, giúp các bên liên quan (Stakeholders) dễ dàng theo dõi và khai thác.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dữ liệu gốc Telco Customer Churn
├── src/
│   ├── __init__.py
│   ├── data_loader.py                         # Bước 1: Đọc & tóm tắt cấu trúc dữ liệu
│   ├── cleaning.py                            # Bước 2: Tiền xử lý & làm sạch TotalCharges
│   ├── analysis.py                            # Bước 3: Tính bảng chéo tỷ lệ churn theo nhóm
│   └── model.py                               # Bước 4: Pipeline ML, CV 5-fold, Permutation Importance
├── static/
│   └── style.css                              # Giao diện CSS tối giản, hỗ trợ Dark/Light mode
├── templates/
│   ├── nav.html                               # Component thanh điều hướng chung
│   ├── index.html                             # Trang 1: Tổng quan dữ liệu & EDA
│   ├── factors.html                           # Trang 2: Biểu đồ tỷ lệ rời bỏ theo nhóm
│   └── model.html                             # Trang 3: Bảng so sánh mô hình & tầm quan trọng yếu tố
├── .gitignore                                 # Bỏ qua venv, __pycache__, file rác
├── requirements.txt                           # Danh sách thư viện phụ thuộc
├── app.py                                     # Ứng dụng Flask chính điều phối các trang
└── README.md                                  # Tài liệu thuyết minh dự án
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Yêu cầu môi trường
- Python 3.10 trở lên.
- Trình duyệt web hiện đại (hỗ trợ JavaScript ES6).

### 2. Cài đặt các thư viện phụ thuộc
Mở terminal tại thư mục gốc của dự án và chạy lệnh:

```bash
pip install -r requirements.txt
```

### 3. Hướng dẫn chạy mô hình Machine Learning độc lập (Terminal / CLI)
Nếu muốn kiểm tra nhanh kết quả huấn luyện mô hình, đánh giá các chỉ số và xem tầm quan trọng của các đặc trưng trực tiếp từ dòng lệnh mà không cần mở trình duyệt:

```bash
python -m src.model
```

**Quá trình thực hiện khi chạy lệnh:**
1. Tải và làm sạch dữ liệu thông qua `clean(load_raw())`.
2. Phân chia tập dữ liệu `Stratified Train/Test (80/20)`.
3. Huấn luyện và đánh giá 3 mô hình (`Logistic Regression`, `Random Forest`, `Histogram Gradient Boosting`) qua `Stratified 5-Fold Cross-Validation`.
4. Dự đoán trên tập kiểm tra và tính toán các chỉ số: CV ROC-AUC, Test ROC-AUC, Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
5. Tính toán Tỷ số số chênh (*Odds Ratio*) từ các hệ số của Logistic Regression.
6. Tính toán mức độ quan trọng của đặc trưng (*Permutation Importance*) đối với mô hình có CV ROC-AUC cao nhất.
7. Xuất toàn bộ bảng so sánh mô hình và bảng xếp hạng đặc trưng ra màn hình terminal.

### 4. Khởi chạy giao diện Web Dashboard (Flask App)
Để xem toàn bộ báo cáo phân tích, biểu đồ tương tác và bảng điều khiển mô hình trên trình duyệt web:

```bash
python app.py
```

Sau khi khởi chạy thành công, mở trình duyệt và truy cập các trang:
- 👉 **Tổng quan dữ liệu:** `http://127.0.0.1:5000/` (kiểm tra phân bố Churn, cấu trúc cột và dữ liệu thiếu).
- 👉 **Tỷ lệ theo nhóm:** `http://127.0.0.1:5000/factors` (khám phá tỷ lệ rời bỏ theo từng đặc trưng với biểu đồ tương tác).
- 👉 **Mô hình dự đoán:** `http://127.0.0.1:5000/model` (bảng so sánh mô hình, ma trận nhầm lẫn, biểu đồ hệ số Odds Ratio và biểu đồ Permutation Importance).

> **Lưu ý:** Lần đầu tiên truy cập trang `/model`, hệ thống sẽ mất khoảng 10–15 giây để huấn luyện và tính toán toàn diện các mô hình. Kết quả sẽ được lưu vào bộ nhớ cache (`@lru_cache`) nên các lần xem hoặc tải lại trang tiếp theo sẽ hiển thị ngay lập tức.

---

## 📊 Những Phát Hiện Nghiệp Vụ Chính (Key Insights)

1. **Ràng buộc hợp đồng là yếu tố giữ chân mạnh mẽ nhất:**
   Khách hàng có hợp đồng 2 năm có tỷ lệ rời bỏ cực thấp ($2,83\%$), trong khi khách hàng trả theo từng tháng rời bỏ tới $42,71\%$. Tỷ số số chênh (Odds Ratio) trong Logistic Regression xác nhận hợp đồng dài hạn giảm mạnh nhất nguy cơ rời bỏ.
2. **Khách hàng mới là đối tượng rủi ro cao nhất:**
   Thời gian sử dụng (`tenure`) là biến có Permutation Importance cao nhất. Phần lớn khách hàng rời bỏ rơi vào những tháng đầu tiên sử dụng dịch vụ.
3. **Cảnh báo chất lượng gói Cáp quang (Fiber optic):**
   Mặc dù là dịch vụ cao cấp mang lại doanh thu cao (`MonthlyCharges` cao), nhóm sử dụng Fiber optic lại rời bỏ tới $41,89\%$. Doanh nghiệp cần rà soát lại chất lượng đường truyền hoặc giá cước dịch vụ này.
4. **Phương thức thanh toán tự động giúp giảm churn:**
   Khách hàng sử dụng séc điện tử (*Electronic check*) rời mạng cao gấp đôi so với khách hàng thanh toán tự động qua thẻ tín dụng (*Credit card automatic*) hoặc chuyển khoản ngân hàng (*Bank transfer automatic*).
