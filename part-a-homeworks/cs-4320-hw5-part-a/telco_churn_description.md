## Dataset: Customer Churn (Telecommunications)

In this assignment, you will work with a **real-world customer churn dataset** derived from a telecommunications provider.

Each row in the dataset represents a **single customer account**. The goal is to predict whether a customer **churned** (left the service) based on demographic information, service characteristics, and billing details.

This dataset is intentionally realistic rather than perfectly clean. Your focus should be on **building a correct end-to-end classification workflow** and on **interpreting evaluation metrics**, not on achieving maximum predictive performance.

---

## Target Variable

### `Churn`

- Binary classification target
- Encoded as:
  - `1` → customer churned
  - `0` → customer did not churn

---

## Features

The dataset includes a mix of numeric and categorical features:

### Customer & Demographics
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

### Account & Contract Information
- `tenure`  
  Number of months the customer has been with the company
- `Contract`  
  Type of contract (e.g., month-to-month, one year, two year)
- `PaperlessBilling`
- `PaymentMethod`

### Service Information
- `InternetService`

### Billing Information
- `MonthlyCharges`
- `TotalCharges`

---

## Important Notes

- The dataset contains **both numeric and categorical features**, requiring appropriate preprocessing.
- Some features may require **type conversion or missing value handling** as part of a safe machine learning pipeline.
- The dataset exhibits **class imbalance**, which is intentional and should be treated as part of the problem.
- You should not assume that accuracy alone is sufficient to evaluate model performance.

---

## Intended Use in This Assignment

This dataset is designed to support:

- Train / validation / test splitting
- Pipeline-based preprocessing
- Baseline classification modeling
- Confusion matrix interpretation
- Comparison of classification metrics
- ROC and Precision–Recall analysis
- Reasoned decision-threshold selection

Your grade is based on **correct workflow and clear reasoning**, not on maximizing performance metrics.

