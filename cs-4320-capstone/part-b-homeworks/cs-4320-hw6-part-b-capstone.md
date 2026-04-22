## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection
* **Data Modality:** Tabular
* **Task Type:** Classification
* **One-Sentence Goal:** Use current/voltage measurements to detect whether any grid fault occurred, then use that as a base for more detailed fault-type modeling later.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Overfitting detection and complexity control using regularization and scoped hyperparameter tuning (logistic regression with regularization strength control).
* **Key Assumptions of This Technique:**
  * Train/validation behavior is informative about generalization (not just noise).
  * A controlled change in complexity (for example `C`) can produce measurable tradeoffs in bias/variance.

**Fit Assessment (required):**

> I expect this technique to be a **partial** fit for my project because:

My capstone data is tabular and works well with regularized linear models, so the methods of this week do apply. However, this dataset appears to have limited separability for the current binary target, becuase the labels don't create clean boundaries when checked for any fault, so large improvements from tuning are unlikely. The technique is still useful because it gives evidence about whether model complexity is actually the bottleneck. It can also be adapted easily later if the target is modified to be more separable when I want to search for a more specific fault.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** Raw numeric measurements as features, with a binary proxy target: `1` if any line indicates a fault and a  `0` otherwise.
* **Why this representation was reasonable for this week:**
The assignment focus is regularization and generalization behavior, so a binary target is a clean way to inspect train-vs-validation patterns without mixing in multiclass label complexity.

---

## 4. What Was Attempted

I implemented a scoped regularization of a logistic regression model with my electrical fault data. Here were the steps I followed:
1. Load the dat into a pandas data frame.
2. Build the binary target and separate the features from the target.
3. Split the data into train, val, and test to prevent data leakage.
4. Identify numerical and categorical features for preprocessing.
5. Build preprocessing pipeline with meadian imputation and standard scaling for numeric features and mode/one-hot for categorical features.
6. Combine the preprocesser and logistic regression model in a pipeline.
7. Train and evaluate a baseline model for initial stats to compare against.
8. Do a regularization sweep to find a better C.
9. Build a validation curve using cross validation on train/val data.
10. Run a small grid search to also test L1_ratio in combo with C.
11. Find the best hyperparameters from the grid search and train a tuned model.
12. Compare the performance of the tuned and untuned model with the test data.

What I intentionally did not attempt:

* No large hyperparameter search; I intentionally kept search scope small.
* No model-family changes, because this week was about disciplined complexity control, not absolute best optimization.
* No multilabel fault-type model in this step.

Constraints encountered:

* This method appears to have a performance ceiling. The C values did not have a huge range that I could pick from to tune my model and make it much better.
* Limited feature engineering this week by design.

---

## 5. Results or Observations

Key outputs from the run:

* Baseline (train): `f1=0.8229`, `accuracy=0.6991`
* Baseline (validation): `f1=0.8229`, `accuracy=0.6991`
* Best holdout validation `C` from one-parameter sweep: `0.000599484`
* Best settings from small grid search: `{'model__C': 0.001, 'model__l1_ratio': 0.0}`
* Best CV F1: `0.8229`
* Final test baseline: `f1=0.8230`, `accuracy=0.6993`
* Final test tuned: `f1=0.8230`, `accuracy=0.6993`

Validation-curve observation:

* Train and validation F1 are almost flat across most `C` values, suggesting that increasing/decreasing regularization in this range does not significantly change generalization behavior for this target representation.

---

## 6. Interpretation and Judgment

This week’s process was useful because it clearly showed the bias/variance issue instead of assuming it. In this run, there was little to no difference between baseline and tuned performance on validation or test, and the validation curve was mostly flat. This shows that controlling model complexity is not a huge problem for my data set. Tuning hyperparameters for a logistic regression won't super help my project's performance.

The likely implication is that representational limits, such as feature signal for this target, class structure, or simplifying labels into “any fault,” are affecting performance more than overfitting. The idea that tuning regularization alone would lead to significant improvements did not prove true in this case. Still, this result is valuable because it focuses future efforts: attention should shift to feature engineering, task reframing, or different model families.

---

## 7. Forward-Looking Adjustment

Next iteration, I will keep the same leakage-safe evaluation protocol but change one of these:

1. Add richer engineered features (maybe considering interactions between different features) before trying larger model tuning.
2. Reframe from binary “any fault” to per-fault-label modeling.
3. After representation updates, re-run the same regularization curve and small search to re-check whether complexity control becomes meaningful.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

Not applicable for this project at this stage.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code file included below: `cs-4320-hw6-part-b.py`


```python
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
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"
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

```
