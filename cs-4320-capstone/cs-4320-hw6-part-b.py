"""
CS 4320 - Assignment 6 (Part B)

Capstone-focused regularization and hyperparameter control.
Goal: assess whether complexity control meaningfully changes generalization.
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


FAULT_COLS = ["G", "C", "B", "A"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electrical_fault_data.csv"
CURVE_PATH = BASE_DIR / "hw6_part_b_validation_curve.png"

RANDOM_STATE = 4320
PRIMARY_METRIC = "f1"
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
                    l1_ratio=l1_ratio,  # 0.0 ~= L2, 1.0 ~= L1 in this environment
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

# Helper function to evaluate a model and return accuracy and F1 scores.
def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def main() -> None:
    # Load data and prepare train/validation/test splits.
    df = pd.read_csv(DATA_PATH)

    # Binary target for this scoped study: any fault vs no fault.
    X = df.drop(columns=FAULT_COLS)
    y = (df[FAULT_COLS].sum(axis=1) > 0).astype(int)

    # Leakage-safe split: keep test untouched until final step.
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

    # Define preprocessing: numeric columns get median imputation + scaling,
    # categorical get most frequent imputation + one-hot encoding.
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
    print("Assignment 6 Part B - Capstone Regularization Check")
    print("===================================================")
    print(f"Rows: {len(df)}")
    print(f"Split sizes: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Baseline (fixed regularization).
    baseline = make_pipeline(preprocessor, c_value=1.0, l1_ratio=0.0)
    baseline.fit(X_train, y_train)
    baseline_train = evaluate(baseline, X_train, y_train)
    baseline_val = evaluate(baseline, X_val, y_val)

    # Print baseline results and a simple fit statement based on train vs. validation F1 gap and absolute levels.
    print("\nBaseline (L2-like, C=1.0)")
    print(f"Train f1={baseline_train['f1']:.4f}, accuracy={baseline_train['accuracy']:.4f}")
    print(f"Val   f1={baseline_val['f1']:.4f}, accuracy={baseline_val['accuracy']:.4f}")

    # Scoped one-parameter sweep: vary C only.
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
    best_holdout_c = float(sweep_df.loc[sweep_df["val_f1"].idxmax(), "C"])
    print("\nScoped C sweep (L2-like)")
    print(sweep_df.to_string(index=False))
    print(f"Best C on validation holdout by f1: {best_holdout_c:.6g}")

    # Validation curve (CV on train_val only).
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

    # Compute mean scores across folds for plotting and identify the best C based on validation mean F1.
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    chosen_curve_c = float(C_VALUES[np.argmax(val_mean)])

    # Plot the validation curve and save it to a file.
    plt.figure(figsize=(8, 5))
    plt.semilogx(C_VALUES, train_mean, marker="o", label="Train F1")
    plt.semilogx(C_VALUES, val_mean, marker="o", label="Validation F1")
    plt.xlabel("C (larger C = weaker regularization)")
    plt.ylabel("F1")
    plt.title("HW6 Part B - Validation Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=200)
    plt.close()

    # Print validation curve results and the chosen C from the curve.
    print(f"\nSaved validation curve: {CURVE_PATH}")
    print(f"C selected from validation curve: {chosen_curve_c:.6g}")

    # Small hyperparameter search.
    grid = GridSearchCV(
        estimator=make_pipeline(preprocessor),
        param_grid={
            "model__l1_ratio": [0.0, 1.0],
            "model__C": np.logspace(-3, 2, 6),
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
    print(f"Best params: {best_params}")
    print(f"Best CV f1: {grid.best_score_:.4f}")

    # Final one-time test evaluation.
    baseline_final = make_pipeline(preprocessor, c_value=1.0, l1_ratio=0.0)
    baseline_final.fit(X_train_val, y_train_val)
    tuned_final = grid.best_estimator_

    # Evaluate both the baseline and tuned models on the test set and print results.
    baseline_test = evaluate(baseline_final, X_test, y_test)
    tuned_test = evaluate(tuned_final, X_test, y_test)

    print("\nFinal TEST comparison")
    print(f"Baseline test: f1={baseline_test['f1']:.4f}, accuracy={baseline_test['accuracy']:.4f}")
    print(f"Tuned test   : f1={tuned_test['f1']:.4f}, accuracy={tuned_test['accuracy']:.4f}")

    # Interpretation of test results based on F1 difference, with a simple threshold for "helped" vs. "hurt" vs. "mostly did not matter".
    delta_f1 = tuned_test["f1"] - baseline_test["f1"]
    if delta_f1 > 0.01:
        print("Interpretation: regularization/tuning helped.")
    elif delta_f1 < -0.01:
        print("Interpretation: regularization/tuning hurt.")
    else:
        print("Interpretation: regularization/tuning mostly did not matter.")


if __name__ == "__main__":
    main()
