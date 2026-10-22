# Customer Retention Intelligence Platform

## Project Overview

The Customer Retention Intelligence Platform is a Python-based machine learning project designed to identify customers at risk of churning and prioritize retention interventions based on their estimated financial value.

Using the Telco Customer Churn dataset, the platform trains a logistic regression model to predict customer churn probabilities. These predictions are combined with estimated customer value, intervention costs, and assumed retention effectiveness to calculate the expected net value of contacting each customer.

The platform then generates a prioritized retention campaign that identifies which customers to contact within a specified budget, helping businesses allocate limited retention resources more effectively.

## Business Objective

Traditional churn prediction strategies often prioritize customers with the highest probability of leaving. However, targeting the highest-risk customers does not necessarily maximize financial returns, since customers differ in their estimated value and the potential benefit of retaining them.

The objective of this project is to maximize modeled expected net value under a fixed retention budget rather than simply identifying customers with the highest churn probability.

By incorporating churn risk, estimated customer value, intervention effectiveness, and intervention costs, the platform prioritizes customers who offer the greatest expected financial benefit from a retention intervention.

## Dataset and Preprocessing

This project uses the Telco Customer Churn dataset, containing 7,043 customer records and 21 columns. Of these columns, 19 were used as predictive features: 3 numerical and 16 categorical. The remaining columns, customerID and Churn, served as the customer identifier and target variable, respectively.

During data cleaning, 11 missing values were identified in the TotalCharges column. Further investigation revealed that all 11 customers had a tenure of zero months. These missing values were replaced with zero, reflecting the absence of recorded customer tenure.

The dataset had an overall churn rate of 26.54%.

Before model training, numerical features were standardized using StandardScaler, while categorical features were encoded using OneHotEncoder. Data validation checks were also implemented to verify required columns, unique customer identifiers, valid churn labels, and tenure values.

The dataset was divided into training (80%) and testing (20%) subsets using a stratified split to preserve the original churn distribution.

## Model Performance

The logistic regression model achieved an accuracy of 80.55%, a precision of 65.72%, and a recall of 55.88% on the test dataset. It also achieved an ROC-AUC score of 0.8420, demonstrating its ability to distinguish between customers who churned and those who remained.

While overall accuracy was relatively high, recall revealed an important limitation. At the default classification threshold of 0.50, the model failed to identify approximately 44% of customers who actually churned. These false negatives represent potentially missed retention opportunities.

False positives also present a business concern, as customers incorrectly predicted to churn could receive unnecessary retention interventions, consuming resources that might otherwise be allocated to more valuable customers.

Rather than relying exclusively on binary churn predictions, the platform uses predicted churn probabilities alongside estimated customer value and intervention costs to prioritize customers by modeled expected net value. This approach seeks to allocate retention resources more efficiently, although its effectiveness depends on the accuracy of the predictions and underlying financial assumptions.

## Retention Optimization

The platform prioritizes customers for retention interventions using a modeled expected net value calculation rather than churn probability alone.

Each customer's estimated value is calculated by multiplying their monthly charges by 12 months and an assumed 70% contribution margin. This represents an estimate of the customer's annual financial contribution.

Expected net value is then calculated as:

**Expected Net Value = (Churn Probability × Estimated Customer Value × Intervention Effectiveness) − Intervention Cost**

The calculation incorporates the predicted probability of churn, the estimated financial value of retaining the customer, the assumed probability that an intervention prevents churn, and the cost of contacting or incentivizing that customer.

The platform only selects customers with a positive expected net value, avoiding interventions whose modeled costs exceed their expected financial benefits.

A fixed campaign budget determines the maximum number of interventions. Customers are ranked by expected net value, and the highest-value eligible customers are selected until the budget limit or the pool of positive-value candidates is exhausted.

Using a $2,000 budget, a $20 intervention cost, and an assumed 20% intervention effectiveness rate, the platform selected 100 customers with a combined modeled expected net value of $11,020.90.

These financial results are scenario-based estimates rather than observed retention outcomes.

## Installation and Usage

### 1. Clone the Repository

Clone the repository from GitHub and navigate to the project directory.

### 2. Set Up the Python Environment

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the Telco Customer Churn dataset and place the CSV file in the following directory:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is not included in the GitHub repository and must be downloaded separately.

### 4. Train the Model

Run the model training script:

```powershell
python -m src.models.train
```

This command validates and cleans the dataset, trains the logistic regression pipeline, and saves the trained model to:

```text
models/churn_logistic_regression.joblib
```

### 5. Generate a Retention Campaign

Generate a retention campaign using a $2,000 budget:

```powershell
python -m src.retention.run_campaign --budget 2000
```

The platform ranks customers by modeled expected net value, selects eligible customers within the budget, and exports the results to:

```text
data/processed/retention_campaign.csv
```

To use a different budget or output filename:

```powershell
python -m src.retention.run_campaign --budget 500 --output data/processed/campaign_500.csv
```

### 6. Run Automated Tests

Verify the project's functionality using pytest:

```powershell
python -m pytest -q
```

At the time of development, the project contains 52 passing automated tests covering data validation, preprocessing, prediction, retention recommendations, model training, and campaign generation.

## Project Structure

```text
customer-retention-intelligence/
├── data/
│   ├── raw/             # Original Telco Customer Churn dataset
│   └── processed/       # Generated retention campaign CSV files
├── models/              # Saved trained machine learning models
├── notebooks/           # Exploratory data analysis and baseline modeling
├── reports/             # Model explainability results and figures
├── src/
│   ├── data/            # Data loading, validation, cleaning, and pipeline
│   ├── models/          # Model training and churn prediction
│   └── retention/       # Retention scoring, optimization, and campaign CLI
├── tests/               # Automated tests for project functionality
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

The project follows a modular structure that separates data processing, machine learning, and retention optimization. Exploratory analysis is maintained in Jupyter notebooks, while reusable functionality is implemented in Python modules under `src/`.

Automated tests verify the reliability of the core functionality, and the `reports/` directory contains model explainability outputs, including feature importance and logistic regression coefficient analyses.

Raw datasets, generated campaign files, and serialized model artifacts are excluded from version control and can be recreated using the documented workflow.

## Limitations and Future Improvements

While the Customer Retention Intelligence Platform demonstrates how machine learning can support retention decisions, several limitations should be considered.

The platform assumes an intervention effectiveness rate of 20%. This value is hypothetical and was not established through a controlled retention experiment. Consequently, the modeled financial benefits should not be interpreted as realized savings or guaranteed returns.

Estimated customer value is calculated using an assumed 70% contribution margin and a 12-month value horizon. Actual customer profitability was not observed, meaning these estimates may differ from real financial outcomes.

Additionally, the model was developed using historical customer data. Its predictive performance may differ when applied to new customers or changing market conditions.

Another important limitation is that predicting customer churn does not establish whether an intervention would successfully prevent that churn. Future improvements could incorporate experimental campaign data to estimate customer-specific intervention effectiveness.

Planned enhancements include:

- **SQL integration:** Store, query, and analyze customer data using a relational database.
- **Interactive dashboard:** Visualize churn risk, campaign recommendations, estimated financial value, and model performance.
- **Intervention effectiveness analysis:** Use actual campaign outcomes to evaluate retention effectiveness and improve financial estimates.