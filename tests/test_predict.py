import pytest
import numpy as np
import pandas as pd

from unittest.mock import Mock
from src.models.predict import load_churn_model
from src.models.predict import predict_churn

def test_missing_model_file():
    path = "models/nonexistent_model.joblib"

    with pytest.raises(FileNotFoundError):
        load_churn_model(path)


def test_predict_churn_probabilities():
    # create a mock model
    mock_model = Mock()
    mock_model.feature_names_in_ = np.array(["customers"])

    mock_model.predict_proba.return_value = np.array([
        [0.90, 0.10],
        [0.25, 0.75],
        [0.60, 0.40]
    ])

    df = pd.DataFrame(
        {
            "customers": [1, 2, 3]
        }
    )

    test_prediction = predict_churn(mock_model, df)

    pd.testing.assert_frame_equal(
        mock_model.predict_proba.call_args.args[0],
        df
    )
    assert np.allclose(test_prediction, np.array([0.10, 0.75, 0.40]))


def test_predict_churn_invalid_input():
    mock_model = Mock()
    customers_list = [
        1,
        2,
        3,
    ]

    with pytest.raises(TypeError):
        predict_churn(mock_model, customers_list)

    mock_model.predict_proba.assert_not_called()

def test_predict_churn_missing_features():
    mock_model = Mock()
    mock_model.feature_names_in_ = np.array([
        "tenure",
        "MonthlyCharges",
        "Contract"
    ])
    df = pd.DataFrame(
        {
            "tenure": [0, 1, 2],
            "MonthlyCharges": [100, 50, 20]
        }
    )

    with pytest.raises(ValueError, match="Contract"):
        predict_churn(mock_model, df)

    mock_model.predict_proba.assert_not_called()

def test_predict_churn_column_order_and_extra_features():
    mock_model = Mock()
    mock_model.feature_names_in_ = np.array([
        "tenure",
        "MonthlyCharges",
        "Contract"
    ])

    mock_model.predict_proba.return_value = np.array([
        [0.20, 0.80],
        [0.70, 0.30],
        [0.40, 0.60]
    ])

    df = pd.DataFrame(
        {
            "MonthlyCharges": [100, 50, 20],
            "Contract": [True, False, True],
            "tenure": [0, 1, 2],
            "customerID": [1, 2, 3]
        }
    )

    predictions = predict_churn(mock_model, df)

    mock_model.predict_proba.assert_called_once()

    passed_df = mock_model.predict_proba.call_args.args[0]

    assert list(passed_df.columns) == [
        "tenure",
        "MonthlyCharges",
        "Contract"
    ]
