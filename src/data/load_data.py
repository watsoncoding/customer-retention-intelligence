import pandas as pd
from pathlib import Path

# Create a function to load data
def load_customer_data(file_path):
    data = pd.read_csv(file_path)
    return data