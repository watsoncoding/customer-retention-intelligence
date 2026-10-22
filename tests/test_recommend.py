import pytest
import pandas as pd
import numpy as np

from src.retention.recommend import select_retention_customer, score_retention_customers, generate_retention_campaign
from unittest.mock import Mock

df = pd.DataFrame(
    {
        "expected_net_value": [50, 20, -10, 80, 0]
    }
)

mock_model = Mock()
mock_model.predict_proba.return_value = np.array([
    [0.20, 0.80],
    [0.70, 0.30]
])
mock_model.feature_names_in_ = np.array(["MonthlyCharges"])

customers = pd.DataFrame(
    {
        "MonthlyCharges": [100.00, 50.00]
    }
)


#Test functions
def test_negative_budget():
    with pytest.raises(ValueError):
        select_retention_customer(
            df,
            budget=-100,
            intervention_cost=20
            )

def test_zero_intervention_cost():
    with pytest.raises(ValueError):
        select_retention_customer(
            df,
            budget=2000,
            intervention_cost=0
        )

def test_selects_highest_value_customers():
    selected = select_retention_customer(
        df,
        budget=40,
        intervention_cost=20
    )

    assert len(selected) == 2
    assert selected["expected_net_value"].tolist() == [80, 50]

def test_insufficient_budget():
    selected = select_retention_customer(
        df,
        budget=10,
        intervention_cost=20
    )

    assert selected.empty

def test_excludes_nonpositive_value():
    selected = select_retention_customer(
        df,
        budget=200,
        intervention_cost=20
    )

    assert len(selected) == 3
    assert (selected["expected_net_value"] > 0).all()

def test_score_retention_customers():
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([
        [0.20, 0.80],
        [0.70, 0.30]
    ])
    mock_model.feature_names_in_ = np.array(["MonthlyCharges"])

    customers = pd.DataFrame(
        {
            "MonthlyCharges": [100.00, 50.00]
        }
    )

    scored_customers = score_retention_customers(model=mock_model, customers=customers, contribution_margin=0.70, intervention_effectiveness=0.20, intervention_cost=20)
    assert np.allclose(
        scored_customers["churn_probability"],
        [0.80, 0.30]
    )
    assert np.allclose(
        scored_customers["estimated_customer_value"],
        [840, 420]
    )
    assert np.allclose(
        scored_customers["expected_net_value"],
        [114.40, 5.20]
    )
    assert list(customers.columns) == ["MonthlyCharges"]

def test_generate_retention_campaign():
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([
        [0.20, 0.80],
        [0.70, 0.30]
    ])
    mock_model.feature_names_in_ = np.array(["MonthlyCharges"])

    customers = pd.DataFrame(
        {
            "MonthlyCharges": [100.00, 50.00]
        }
    )

    campaign_customers = generate_retention_campaign(model=mock_model, customers=customers, budget=20, contribution_margin=0.70, intervention_effectiveness=0.20, intervention_cost=20)

    assert len(campaign_customers) == 1
    assert np.allclose(campaign_customers["MonthlyCharges"], 100)
    assert np.allclose(campaign_customers["expected_net_value"], 114.40)
    assert len(campaign_customers) * 20 <= 20

@pytest.mark.parametrize("invalid_margin", [-0.1, 1.5, float("nan"), float("inf"), -float("inf"), "0.70", True])
def test_invalid_contribution_margin(invalid_margin):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([
        [0.20, 0.80],
        [0.70, 0.30]
    ])
    mock_model.feature_names_in_ = np.array(["MonthlyCharges"])
    df = pd.DataFrame(
        {
            "MonthlyCharges": [100.00, 50.00]
        }
    )

    with pytest.raises(ValueError, match="contribution_margin"):
        score_retention_customers(model=mock_model, customers=df, contribution_margin=invalid_margin, intervention_effectiveness=0.2, intervention_cost=20)

    mock_model.predict_proba.assert_not_called()

@pytest.mark.parametrize("invalid_effectiveness", [-0.1, 1.5, float("nan"), float("inf"), -float("inf"), "0.70", False])
def test_invalid_intervention_effectiveness(invalid_effectiveness):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([
        [0.20, 0.80],
        [0.70, 0.30]
    ])
    mock_model.feature_names_in_ = np.array(["MonthlyCharges"])
    df = pd.DataFrame(
        {
            "MonthlyCharges": [100.00, 50.00]
        }
    )

    with pytest.raises(ValueError, match="intervention_effectiveness"):
        score_retention_customers(model=mock_model, customers=df, contribution_margin=0.7, intervention_effectiveness=invalid_effectiveness, intervention_cost=20)

    mock_model.predict_proba.assert_not_called()

@pytest.mark.parametrize("invalid_cost", [-20, 0, float("nan"), float("inf"), -float("inf"), "70", True])
def test_invalid_intervention_cost(invalid_cost):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([
        [0.20, 0.80],
        [0.70, 0.30]
    ])
    mock_model.feature_names_in_ = np.array(["MonthlyCharges"])
    df = pd.DataFrame(
        {
            "MonthlyCharges": [100.00, 50.00]
        }
    )

    with pytest.raises(ValueError, match="intervention_cost"):
        score_retention_customers(model=mock_model, customers=df, contribution_margin=0.7, intervention_effectiveness=0.2, intervention_cost=invalid_cost)

    mock_model.predict_proba.assert_not_called()

@pytest.mark.parametrize(
    "contribution_margin, intervention_effectiveness, intervention_cost",
    [
        (0, 0.20, 20),
        (1, 0.20, 20),
        (0.70, 0, 20),
        (0.70, 1, 20),
        (0.70, 0.20, 0.01),
    ]
)
def test_valid_economic_boundaries(
    contribution_margin,
    intervention_effectiveness,
    intervention_cost
):
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array(
        [
            [0.20, 0.80],
            [0.70, 0.30]
        ]
    )

    mock_model.feature_names_in_ = np.array(["MonthlyCharges"])

    df = pd.DataFrame(
        {
            "MonthlyCharges": [100.00, 50.00]
        }
    )

    scored_df = score_retention_customers(model=mock_model, customers=df, contribution_margin=contribution_margin, intervention_effectiveness=intervention_effectiveness, intervention_cost=intervention_cost)

    assert len(scored_df) == 2
    mock_model.predict_proba.assert_called_once()