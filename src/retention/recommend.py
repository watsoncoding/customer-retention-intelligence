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