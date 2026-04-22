# SaaS Customer Churn Dataset for Assignment 11 – Part A

This package provides a synthetic but interpretable **tabular binary classification** dataset designed for the neural network assignment on MLPs, baseline comparison, and training-dynamics interpretation.

## Narrative

Each row represents a software-as-a-service customer account approaching renewal. The target `churn_risk` indicates whether that account is at elevated risk of non-renewal.

The dataset is intentionally constructed so that:

- a **linear baseline** is reasonable but limited
- a **small MLP** can capture nonlinear feature interactions
- a **larger MLP** can overfit if regularization is removed or training is pushed too far
- preprocessing matters because the data contains both numeric and categorical features plus missing values

## Files

- `saas_customer_churn_mlp.csv` — full dataset
- `feature_dictionary.md` — feature descriptions

## Notes

You should:

- treat this as a standard supervised tabular classification problem
- use train / validation / test isolation
- preprocess numeric and categorical columns safely
- compare a simple baseline against a small MLP
- interpret training curves instead of only chasing accuracy

You should not assume that a bigger network is automatically better.
