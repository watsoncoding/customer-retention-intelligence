from src.data.load_data import load_customer_data

from src.data.validate_data import (
    validate_required_columns,
    validate_unique_customer_ids,
    validate_churn_values,
    validate_tenure,

)

from src.data.clean_data import clean_total_charges

def run_data_pipeline(file_path):
    # Load data
    df = load_customer_data(file_path)

    # Validate data
    validate_required_columns(df)
    validate_tenure(df)
    validate_unique_customer_ids(df)
    validate_churn_values(df)

    # Clean data
    cleaned_df = clean_total_charges(df)
    return cleaned_df