import pandas as pd
import pytest

from src.data.clean_data import clean_total_charges

def test_clean_total_charges_valid():
    df = pd.DataFrame(
        {
            "tenure": [0, 12, 24],
            "TotalCharges": [" ", "850.50", "1750.25"]
        }
    )

    cleaned_df = clean_total_charges(df)
    
    assert cleaned_df["TotalCharges"].isna().sum() == 0
    assert cleaned_df.loc[0, "TotalCharges"] == 0
    assert cleaned_df.loc[1, "TotalCharges"] == 850.50
    assert cleaned_df.loc[2, "TotalCharges"] == 1750.25
    assert df.loc[0, "TotalCharges"] == " "

def test_clean_total_charges_invalid():
    df = pd.DataFrame(
        {
            "tenure": [0, 24],
            "TotalCharges": [" ", "INVALID"]
        }
    )

    with pytest.raises(ValueError):
        clean_total_charges(df)