# imports
import joblib
import pandas as pd
from pathlib import Path

# create a function to load model
def load_churn_model(model_path):
    churn_model_path = Path(model_path)

    if not churn_model_path.is_file():
        raise FileNotFoundError(f"Model file not found:", {churn_model_path})

    return joblib.load(churn_model_path)


# create the prediction function
def predict_churn(model, customers):
   # verify that customers is a pandas DataFrame
   if not isinstance(customers, pd.DataFrame):
      raise TypeError("customers must be a pandas DataFrame.")

   # retrieve the model's expected feature names
   missing_features = [
      feature for feature in model.feature_names_in_
      if feature not in customers.columns
   ]

   if missing_features:
      raise ValueError(f"Missing required features:, {missing_features}")

   # check for extra columns
   expected_features = list(model.feature_names_in_)
   validated_customers = customers[expected_features]

   churn_proba = model.predict_proba(validated_customers)[:, 1]

   return churn_proba