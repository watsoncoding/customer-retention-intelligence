REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

import pandas as pd

def validate_required_columns(df):
    required_columns_set = set(REQUIRED_COLUMNS)
    df_columns_set = set(df.columns)
    missing_columns = required_columns_set - df_columns_set

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def validate_unique_customer_ids(df):
    duplicated_ids = df["customerID"].duplicated().sum()
    if duplicated_ids > 0:
        raise ValueError(f"Found {duplicated_ids} duplicate customer IDs.")

    return True

def validate_churn_values(df):
    allowed_values = {"Yes", "No"}
    actual_values = set(df["Churn"].unique())
    invalid_target_values = actual_values - allowed_values

    if invalid_target_values:
        raise ValueError(f"Invalid target values: {invalid_target_values}")

    return True

def validate_tenure(df):
    numeric_tenure = pd.to_numeric(df["tenure"], errors="coerce")
    missing_tenure_mask = numeric_tenure.isna()
    negative_tenure_mask = numeric_tenure < 0
    whole_num_mask = numeric_tenure % 1 != 0

    invalid_tenure = missing_tenure_mask | negative_tenure_mask | whole_num_mask

    if invalid_tenure.any():
        raise ValueError("Invalid tenure values detected.")

    return True