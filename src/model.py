from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = df.drop(columns=["customerID", "Churn"])
y = (df["Churn"] == "Yes").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

prep = ColumnTransformer([
    ("num", StandardScaler(), ["tenure", "MonthlyCharges", "TotalCharges"]),
    ("cat", OneHotEncoder(drop="first"), cat_cols),  # cat_cols: các cột phân loại còn lại
])
model = Pipeline([
    ("prep", prep),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)