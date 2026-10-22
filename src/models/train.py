import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from src.data.pipeline import run_data_pipeline


def train_churn_model():
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    df = run_data_pipeline(data_path)
    X = df.drop(columns=["customerID", "Churn"]).copy()
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    categorical_features = [
        col for col in X_train.columns
        if col not in numeric_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)

        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            )),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel Evaluation")
    print("-" * 30)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "churn_logistic_regression.joblib"

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
    return model

if __name__ == "__main__":
    train_churn_model()