import math
import numbers
from src.models.predict import predict_churn

def select_retention_customer(df, budget, intervention_cost):
    # Validate the budget and intervention cost
    if budget < 0:
        raise ValueError("Invalid budget.")
    if intervention_cost <= 0:
        raise ValueError("Invalid intervention cost.")
    # Calculate num of customers budget can cover
    num_customers = budget // intervention_cost
    # Exclude customers whose expected net value is zero or negative
    filtered_df = df[df["expected_net_value"] > 0]
    # Select the highest-value customers up to the budget limit
    selected_customers = filtered_df.nlargest(int(num_customers), "expected_net_value")
    # Return a DataFrame containing those customers
    return selected_customers

def _validate_economic_parameter(value, name, min_value, max_value=None, inclusive_min=True):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")
    if inclusive_min:
        if value < min_value:
            raise ValueError(f"{name} must be greater than {min_value}")
    else:
        if value <= min_value:
            raise ValueError(f"{name} must be greater than {min_value}")
    if max_value is not None:
        if value > max_value:
            raise ValueError(f"{name} must be less than {max_value}")

def score_retention_customers(
        model,
        customers,
        contribution_margin=0.70,
        intervention_effectiveness=0.20,
        intervention_cost=20
):
    _validate_economic_parameter(
        contribution_margin, "contribution_margin", 0, 1
    )

    _validate_economic_parameter(
        intervention_effectiveness, "intervention_effectiveness", 0, 1
    )

    _validate_economic_parameter(
        intervention_cost, "intervention_cost", 0, inclusive_min=False
    )

    scored_df = customers.copy()
    scored_df["churn_probability"] = predict_churn(model, scored_df)
    scored_df["estimated_customer_value"] = scored_df["MonthlyCharges"] * 12 * contribution_margin
    scored_df["expected_net_value"] = scored_df["churn_probability"] * scored_df["estimated_customer_value"] * intervention_effectiveness - intervention_cost

    return scored_df

def generate_retention_campaign(
        model,
        customers,
        budget,
        contribution_margin=0.70,
        intervention_effectiveness=0.20,
        intervention_cost=20
):
    scored_df = score_retention_customers(model, customers, contribution_margin, intervention_effectiveness, intervention_cost)
    campaign_customers = select_retention_customer(scored_df, budget, intervention_cost)

    return campaign_customers