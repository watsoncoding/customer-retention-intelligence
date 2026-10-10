import pandas as pd

def clean_total_charges(df):
    cleaned_df = df.copy()
    cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")

    invalid_missing_mask = cleaned_df["TotalCharges"].isna() & (cleaned_df["tenure"] > 0)

    if invalid_missing_mask.any():
        raise ValueError(f"Found {(invalid_missing_mask).sum()} customers with missing TotalCharges and positive tenure.")

    valid_missing_mask = cleaned_df["TotalCharges"].isna() & (cleaned_df["tenure"] == 0)
    cleaned_df.loc[valid_missing_mask, "TotalCharges"] = 0
          
    return cleaned_df