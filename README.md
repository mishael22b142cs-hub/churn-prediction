# Churn Prediction

## Problem Statement

Customer churn — the loss of subscribers or customers — is one of the most
expensive problems a subscription-based business faces, since acquiring a
new customer typically costs far more than retaining an existing one. This
project builds an end-to-end machine learning pipeline that predicts whether
a telecom customer is likely to churn, so that retention teams can
proactively target at-risk customers with offers or interventions before
they leave.

## Dataset

This project uses the **Telco Customer Churn** dataset (commonly distributed
by IBM / available on Kaggle). Each row represents a customer of a telecom
company, with columns describing:

- **Demographics**: gender, SeniorCitizen, Partner, Dependents
- **Account information**: tenure, Contract, PaperlessBilling, PaymentMethod,
  MonthlyCharges, TotalCharges
- **Services signed up for**: PhoneService, MultipleLines, InternetService,
  OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
  StreamingMovies
- **Target**: `Churn` (Yes/No) — whether the customer left within the last
  month

Place the CSV file (e.g. `WA_Fn-UseC_-Telco-Customer-Churn.csv`) in the
`data/` directory before running the notebooks. The loading notebook expects
a single CSV file in `data/` and will pick it up automatically.

## Approach

The project follows a linear notebook pipeline, with each notebook persisting
its output as a pickle/artifact for the next stage to consume:

1. **`01_data_loading.ipynb`** — Load the raw CSV, inspect shape, dtypes,
   missing values, and summary statistics. Save the raw DataFrame to
   `data/raw_df.pkl`.
2. **`02_eda.ipynb`** — Exploratory data analysis: churn distribution, churn
   by contract type, tenure vs. churn, and a correlation heatmap of numeric
   features, with commentary on each insight.
3. **`03_preprocessing.ipynb`** — Clean the data (fix `TotalCharges`, drop
   identifiers, handle missing values), encode categorical features and the
   target, split into train/test (stratified 80/20), and apply **SMOTE** to
   the training set only to correct class imbalance. Saves
   `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl` to `data/`.
4. **`04_train_model.ipynb`** — Train a **Logistic Regression** baseline and
   an **XGBoost** classifier on the resampled training data. Both models are
   saved to `models/` with `joblib`.
5. **`05_evaluate.ipynb`** — Evaluate both models on the held-out test set
   (accuracy, precision, recall, F1, classification report, confusion
   matrices saved as PNGs to `results/`), compare them, and persist the
   better-performing model as `models/best_model.pkl`.
6. **`app/streamlit_app.py`** — A Streamlit app that loads
   `models/best_model.pkl` and lets a user enter a customer's attributes in
   the sidebar to get a live churn prediction with probability.

## How to Run

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Place the Telco Customer Churn CSV in `data/`.

3. Run the notebooks **in order** (each one depends on artifacts saved by
   the previous one):

   ```
   notebooks/01_data_loading.ipynb
   notebooks/02_eda.ipynb
   notebooks/03_preprocessing.ipynb
   notebooks/04_train_model.ipynb
   notebooks/05_evaluate.ipynb
   ```

   You can run them with Jupyter (`jupyter notebook`) or
   `jupyter nbconvert --to notebook --execute <notebook>`.

4. Launch the Streamlit app:

   ```bash
   streamlit run app/streamlit_app.py
   ```

## Results

_To be filled in after running `05_evaluate.ipynb` on your data._

| Model               | Accuracy | Precision | Recall | F1 |
|---------------------|----------|-----------|--------|----|
| Logistic Regression |          |           |        |    |
| XGBoost             |          |           |        |    |

Confusion matrices for both models are saved to `results/` after running the
evaluation notebook, and the better-performing model (by F1 score) is saved
as `models/best_model.pkl` for use by the Streamlit app.
