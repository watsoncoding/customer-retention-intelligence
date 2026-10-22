import argparse
import math
from pathlib import Path

from src.data.pipeline import run_data_pipeline
from src.models.predict import load_churn_model
from src.retention.recommend import generate_retention_campaign

def main():
    parser = argparse.ArgumentParser(
        description="Generate a budget-constrained customer retention campaign."
    )

    parser.add_argument(
        "--budget",
        type=float,
        required=True,
        help="Total retention campaign budget in dollars."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/retention_campaign.csv",
        help="Path where the campaign CSV will be saved."
    )

    args = parser.parse_args()

    if not math.isfinite(args.budget) or args.budget < 0:
        parser.error("--budget must be a finite, non-negative number.")

    project_root = Path(__file__).resolve().parents[2]

    model_path = project_root / "models" / "churn_logistic_regression.joblib"

    data_path = (
        project_root
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    model = load_churn_model(model_path)

    df = run_data_pipeline(data_path)
    customers = df.drop(columns=["customerID", "Churn"])

    campaign_customers = generate_retention_campaign(
        model=model,
        customers=customers,
        budget=args.budget,
        contribution_margin=0.70,
        intervention_effectiveness=0.20,
        intervention_cost=20
    )

    campaign_customers["customerID"] = df.loc[campaign_customers.index, "customerID"]

    assert campaign_customers["customerID"].notna().all()
    assert campaign_customers["customerID"].is_unique

    campaign_export = campaign_customers[
        [
            "customerID",
            "MonthlyCharges",
            "churn_probability",
            "estimated_customer_value",
            "expected_net_value"
        ]
    ].copy()

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = project_root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_export.to_csv(output_path, index=False)

    print(f"Campaign exported to: {output_path}")


    # print(f"Campaign budget: ${args.budget:,.2f}")
    # print(f"Loaded customers: {len(df):,}")
    # print(f"Model loaded: {type(model).__name__}")

    print(f"Selected customers: {len(campaign_customers):,}")
    print(f"Campaign spend: ${len(campaign_customers) * 20:,.2f}")
    print(
        f"Total expected net value: "
        f"${campaign_customers["expected_net_value"].sum():,.2f}"
    )

if __name__ == "__main__":
    main()