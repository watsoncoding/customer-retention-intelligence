import pytest
from src.retention.recommend import select_retention_customer

import pandas as pd

df = pd.DataFrame(
    {
        "expected_net_value": [50, 20, -10, 80, 0]
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