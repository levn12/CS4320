"""
CS 4320 - Assignment 6 (Part A)


Workflow:
1) Baseline train/validation result
2) One explicit regularization control (C)
3) Validation curve (train + validation scores)
4) Small hyperparameter search
5) Final one-time test comparison
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
# Use a previous-homework dataset (HW5 Part A) for HW6 Part A.
DATA_PATH = BASE_DIR.parent / "cs-4320-hw5-part-a" / "telco_churn.csv"
TARGET_COL = "Churn"
PLOT_PATH = BASE_DIR / "hw6_part_a_validation_curve.png"

# Set a fixed random state for reproducibility across all steps.
RANDOM_STATE = 4320

# Use F1 as the primary metric for model selection and evaluation.
PRIMARY_METRIC = "f1"

# Use a wide range of C values for regularization control and validation curve.
C_VALUES = np.logspace(-4, 3, 50)

# Helper function to create a pipeline with the given preprocessor and logistic regression parameters.
def make_pipeline(preprocessor, c_value=1.0, l1_ratio=0.0):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    C=c_value,
                    l1_ratio=l1_ratio,  # 0.0 ~= L2, 1.0 ~= L1 for this sklearn version
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

# Helper function to evaluate a model and return accuracy and F1 scores.
def evaluate(model, X, y):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

def main():
    # Load data and prepare train/validation/test splits.
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target from dataset.
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Keep test isolated until final step.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    # Further split train_val into train and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE
    )

    # Identify numeric and categorical columns for preprocessing.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Define a preprocessor that median imputes and standard scales numeric features, 
    # and mode imputes and one-hot encodes categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    # Print dataset and split information.
    print("Assignment 6 Part A")
    print("===================")
    print(
        f"Split sizes: train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )

    # 1) Baseline: simple logistic regression with default regularization (L2, C=1.0).
    baseline = make_pipeline(preprocessor, c_value=1.0, l1_ratio=0.0)
    baseline.fit(X_train, y_train)
    baseline_train = evaluate(baseline, X_train, y_train)
    baseline_val = evaluate(baseline, X_val, y_val)

    print("\nBaseline (L2, C=1.0)")
    print(f"Train {PRIMARY_METRIC}: {baseline_train['f1']:.4f}")
    print(f"Val   {PRIMARY_METRIC}: {baseline_val['f1']:.4f}")
    print(f"Train accuracy: {baseline_train['accuracy']:.4f}")
    print(f"Val   accuracy: {baseline_val['accuracy']:.4f}")

    # Simple fit statement based on train vs. validation F1 gap and absolute levels.
    gap = baseline_train["f1"] - baseline_val["f1"]
    # Note: these thresholds are somewhat arbitrary and just for illustrative purposes.
    if gap > 0.1:
        fit_note = "Overfitting signal: train is much higher than validation."
    elif baseline_train["f1"] < 0.70 and baseline_val["f1"] < 0.70:
        fit_note = "Underfitting signal: both train and validation are low."
    else:
        fit_note = "No strong underfitting/overfitting signal."
    print(f"Fit statement: {fit_note}")

    # 2) Regularization control: vary C only (all else fixed).
    sweep_rows = []
    # This loop iteratively trains models with different C values and evaluates them on train and validation sets.
    for c_value in C_VALUES:
        model = make_pipeline(preprocessor, c_value=float(c_value), l1_ratio=0.0)
        model.fit(X_train, y_train)
        train_scores = evaluate(model, X_train, y_train)
        val_scores = evaluate(model, X_val, y_val)
        sweep_rows.append(
            {
                "C": float(c_value),
                "train_f1": train_scores["f1"],
                "val_f1": val_scores["f1"],
            }
        )

    # Create a DataFrame from the sweep results, sort by C, and identify the best C based on validation F1.
    sweep_df = pd.DataFrame(sweep_rows).sort_values("C")
    best_sweep = sweep_df.loc[sweep_df["val_f1"].idxmax()]
    print("\nRegularization sweep (varying C only)")
    print(sweep_df.to_string(index=False))
    print(f"Best holdout-val C by f1: {best_sweep['C']:.6g}")

    # 3) Validation curve on train_val only (still no test leakage).
    """ StratifiedKFold defines the cross-validation splitting strategy for the validation curve.
    It will split the data into n folds, ensuring that each fold has a similar distribution of the
    target classes (stratified). The data will be shuffled before splitting, and a fixed random state 
    is set for reproducibility. This allows us to evaluate the model's performance across different 
    subsets of the training data while maintaining class balance, which is important for reliable 
    validation curve results. """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    curve_model = make_pipeline(preprocessor, c_value=1.0, l1_ratio=0.0)
    train_scores, val_scores = validation_curve(
        estimator=curve_model,
        X=X_train_val,
        y=y_train_val,
        param_name="model__C",
        param_range=C_VALUES,
        cv=cv,
        scoring=PRIMARY_METRIC,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    chosen_c = float(C_VALUES[np.argmax(val_mean)])

    plt.figure(figsize=(8, 5))
    plt.semilogx(C_VALUES, train_mean, marker="o", label="Train F1")
    plt.semilogx(C_VALUES, val_mean, marker="o", label="Validation F1")
    plt.xlabel("C (larger C = weaker regularization)")
    plt.ylabel("F1")
    plt.title("HW6 Part A - Validation Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"\nValidation curve saved: {PLOT_PATH}")
    print(f"Chosen C from validation curve: {chosen_c:.6g}")

    # 4) Small hyperparameter search.
    grid = GridSearchCV(
        estimator=make_pipeline(preprocessor),
        param_grid={
            "model__l1_ratio": [0.0, 1.0],
            "model__C": np.logspace(-3, 2, 10),
        },
        scoring=PRIMARY_METRIC,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train_val, y_train_val)
    best_params = dict(grid.best_params_)
    best_params["model__C"] = float(best_params["model__C"])

    print("\nSmall grid search")
    print(f"Selection metric: {PRIMARY_METRIC}")
    print(f"Best params: {best_params}")
    print(f"Best CV f1: {grid.best_score_:.4f}")

    # 5) Final test evaluation once.
    baseline_final = make_pipeline(preprocessor, c_value=1.0, l1_ratio=0.0)
    baseline_final.fit(X_train_val, y_train_val)
    tuned_final = grid.best_estimator_

    baseline_test = evaluate(baseline_final, X_test, y_test)
    tuned_test = evaluate(tuned_final, X_test, y_test)

    print("\nFinal TEST comparison (one-time)")
    print(f"Baseline test f1: {baseline_test['f1']:.4f}")
    print(f"Tuned    test f1: {tuned_test['f1']:.4f}")
    print(f"Baseline test accuracy: {baseline_test['accuracy']:.4f}")
    print(f"Tuned    test accuracy: {tuned_test['accuracy']:.4f}")

    # Interpretation of test results based on F1 difference, with a simple threshold for "helped" vs. "hurt" vs. "mostly did not matter".
    delta = tuned_test["f1"] - baseline_test["f1"]
    if delta > 0.01:
        print("Interpretation: Regularization/tuning helped.")
    elif delta < -0.01:
        print("Interpretation: Regularization/tuning hurt.")
    else:
        print("Interpretation: Regularization/tuning mostly did not matter.")
    print("Likely explanation: limited data, noise, or already-low model complexity.")


if __name__ == "__main__":
    main()
