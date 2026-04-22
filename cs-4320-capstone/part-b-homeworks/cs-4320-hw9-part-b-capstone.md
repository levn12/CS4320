## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection
* **Data Modality:** Tabular
* **Task Type:** Classification
* **One-Sentence Goal:** Use current and voltage measurements from a simulated three-phase electrical system to classify the observed fault pattern.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Ensemble learning with a single decision tree baseline, Random Forest bagging, and Gradient Boosting.
* **Key Assumptions of This Technique:**
  * A single tree can capture useful non-linear structure, but it may have high variance and overfit when it grows too flexible.
  * Bagging helps when unstable trees can be improved by averaging many decorrelated versions.
  * Boosting helps when the remaining mistakes contain structure that later trees can correct sequentially.

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for my project because:

Tree-based models are a natural match for my electrical dataset because the current and voltage features are numeric, low-dimensional, and likely to interact in non-linear ways. This week made more sense once I moved away from the easier binary `any fault` target and instead predicted the actual fault pattern. That harder multiclass version gives a better test of whether bagging or boosting really helps compared with a single tree.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** Raw numeric sensor values `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc`, with a multiclass target built from the four fault-indicator columns as pattern labels such as `0000`, `0110`, `0111`, `1001`, `1011`, and `1111`.

* **Why this representation was reasonable for this week:**

This representation keeps the project grounded in the original data rather than inventing a separate proxy task. It is also a better fit for this week's ensemble comparison than the binary `any fault` target because the multiclass fault-pattern version is harder and reveals more about model behavior. Using the actual fault-bit combinations also keeps the target interpretable, since each class directly reflects a real observed indicator pattern.

---

## 4. What Was Attempted

I implemented a Part B version of the Week 9 workflow using my electrical fault dataset, but I changed the target from binary `any fault` to multiclass fault pattern. I originally tried the usual target, but all three models were a perfect predictor, so I decided to make the problem a little bit harder so I can see different performance between the models. For the multiclass target I ended up using, the main steps were:

1. Load the CSV and build the multiclass label by concatenating the four indicator columns into fault-pattern strings.
2. Separate the six measurement features from the four fault-indicator columns.
3. Build a reproducible `60/20/20` stratified train/validation/test split with seed `4320`.
4. Use a leakage-safe preprocessing pipeline with median imputation for the numeric features.
5. Train several single decision trees across multiple `max_depth` values to show how tree complexity changes performance.
6. Train a default Random Forest and report both validation metrics and out-of-bag accuracy.
7. Tune a small Random Forest grid over `n_estimators`, `max_depth`, and `max_features`.
8. Tune a small Gradient Boosting grid over `n_estimators`, `learning_rate`, and `max_depth`.
9. Compare the best single-tree, bagging, and boosting models with one concise validation table.
10. Refit the validation winner on train/validation and evaluate it once on the held-out test split.

What I intentionally did not attempt:

* I did not run a large hyperparameter search, because the assignment emphasizes small and interpretable tuning.
* I did not move to a multilabel formulation where each indicator is predicted separately.
* I did not add more advanced boosted-tree libraries, because the assignment only required standard bagging and boosting comparisons.

Constraints encountered:

* Moving to multiclass fault-type prediction made the task more informative, but it also made the model comparison less obviously favorable to ensembles than I expected.
* The dataset appears structured enough that a single unrestricted tree is already very strong, which reduced the room for Random Forest or Gradient Boosting to improve.

---

## 5. Results or Observations

The main result this week is that the multiclass fault-pattern target produced a much more informative comparison than the binary target, but the single decision tree still ended up slightly ahead of the ensemble models on validation. That was not the pattern I was expecting, since I thought the ensembles would have a clearer advantage once the task became harder. However, it goes to to show that a more complex model is not always better. Depending on the task, simpler models may even perform better.

Results snapshot from the run:

* Dataset rows: `7,861`
* Split sizes: `train=4716`, `val=1572`, `test=1573`
* Fault types observed: `0000`, `0110`, `0111`, `1001`, `1011`, `1111`
* Best single-tree validation setting: `max_depth=None`
* Best single-tree validation metrics: `accuracy=0.8664`, `balanced_accuracy=0.8426`, `precision=0.8430`, `recall=0.8426`, `f1=0.8428`
* Default Random Forest validation metrics: `accuracy=0.8658`, `balanced_accuracy=0.8419`, `precision=0.8423`, `recall=0.8419`, `f1=0.8421`
* Default Random Forest OOB accuracy: `0.8584`
* Best tuned Random Forest setting: `n_estimators=100`, `max_depth=None`, `max_features="sqrt"`
* Best tuned Random Forest validation metrics: `accuracy=0.8658`, `balanced_accuracy=0.8419`, `precision=0.8423`, `recall=0.8419`, `f1=0.8421`
* Best Gradient Boosting setting: `n_estimators=300`, `learning_rate=0.1`, `max_depth=3`
* Best Gradient Boosting validation metrics: `accuracy=0.8562`, `balanced_accuracy=0.8305`, `precision=0.8302`, `recall=0.8305`, `f1=0.8302`
* Final selected winner: `Decision Tree`
* Final test metrics for the winner: `accuracy=0.8798`, `balanced_accuracy=0.8590`, `precision=0.8587`, `recall=0.8590`, `f1=0.8585`

Additional observations:

* The tree-depth sweep clearly showed underfitting at shallow depths and much stronger performance as the tree became more flexible.
* Random Forest stayed extremely close to the best single tree, which suggests bagging helped stability but did not unlock a clearly better boundary.
* Gradient Boosting was still strong, but it trailed the best tree and forest on this dataset rather than passing them.

---

## 6. Interpretation and Judgment

This week was useful because it showed that changing the target really did make the assignment more informative. The shallow trees underfit badly, so the multiclass fault-pattern task is clearly harder than the old binary `any fault` version. But once the tree was allowed to grow freely, it slightly outperformed both Random Forest and Gradient Boosting on validation balanced accuracy. That tells me the data has strong non-linear structure, but that structure may already line up well with one flexible tree instead of needing an ensemble to smooth everything out.

The ensemble results were still informative, just not in the way I expected. Random Forest stayed very close to the single tree, which fits the idea that averaging can stabilize predictions without always giving a better final boundary. Gradient Boosting also did well, but it did not beat the tree or the forest, which suggests the stage-by-stage error correction was not especially helpful for this target. So the method transfer worked, but the dataset favored a strong single-tree solution more than a classic ensemble win.

It also makes me a little bit suspicious that I made a mistake somewhere. Even though the simpler single tree may have been able to do well with this data, it still doesn't make complete sense to me that a random forest would perform worse than the single tree.

---

## 7. Forward-Looking Adjustment

The next change I would make is to keep the multiclass fault-type target and push the analysis in one of these directions:

1. Compare class-specific confusion patterns to see which fault types are most often mixed up.
2. Try more targeted tree and forest settings rather than a broad but shallow tuning grid.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

There was a mild mismatch this week. Ensemble learning still fit the project, but the results did not support the simple expectation that Random Forest or boosting would clearly outperform a single decision tree. Instead, the multiclass electrical task still favored one very strong tree. Even so, that mismatch was useful because it revealed something real about the data: the fault-type boundary structure may already align unusually well with one expressive tree.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code file included below: `cs-4320-hw9-part-b.py`


```python
# CS 4320 - Assignment 9 Part B
# Comparing a single decision tree, Random Forest, and Gradient Boosting on electrical fault-type classification.

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


FAULT_COLS = ["G", "C", "B", "A"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"

# Keep the split and model comparisons reproducible across runs.
RANDOM_STATE = 4320
# Use balanced accuracy as the main model-selection metric because the fault types are not perfectly balanced.
PRIMARY_METRIC = "balanced_accuracy"
TREE_DEPTH_VALUES = [3, 5, 8, None]
RF_TUNE_GRID = [
    {"n_estimators": 100, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 8, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 8, "max_features": 0.5},
]
GB_TUNE_GRID = [
    {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 200, "learning_rate": 0.10, "max_depth": 2},
    {"n_estimators": 300, "learning_rate": 0.10, "max_depth": 3},
]


def load_data(data_path: Path = DATA_PATH):
    # Read the electrical fault dataset from disk and build a multiclass fault-pattern target.
    df = pd.read_csv(data_path)
    y = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1).to_numpy()
    X = df.drop(columns=FAULT_COLS)
    return df, X, y


def build_preprocessor(X: pd.DataFrame):
    # The electrical measurements are numeric, so median imputation is enough for this week.
    numeric_cols = X.columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(X: pd.DataFrame, model):
    # Keep preprocessing and model fitting together so every model uses the same leakage-safe workflow.
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


def evaluate(model, X, y):
    # Compute multiclass metrics using macro averages so each fault type matters equally.
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def format_metrics(metrics):
    # Format a metric dictionary into one short printable line.
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )


def print_split_info(y_train, y_val, y_test):
    # Print split sizes and the fault-type distribution for each partition.
    print("Split sizes:")
    print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print("Fault-type counts:")
    for split_name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = pd.Series(split_y).value_counts().sort_index()
        formatted = ", ".join([f"{label}={count}" for label, count in counts.items()])
        print(f"  {split_name}: {formatted}")


def main():
    # Load the electrical fault dataset and create a reproducible 60/20/20 stratified split.
    df, X, y = load_data(DATA_PATH)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print("Assignment 9 Part B - Ensemble Learning on Electrical Fault Types")
    print("=================================================================")
    print(f"Rows: {len(df)}")
    print_split_info(y_train, y_val, y_test)
    print(f"Primary validation metric: {PRIMARY_METRIC}")

    tree_results = []
    best_tree_metrics = None
    best_tree_params = None

    # Try a few tree depths so we can show how a single tree can overfit.
    for max_depth in TREE_DEPTH_VALUES:
        model = make_pipeline(
            X_train,
            DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=RANDOM_STATE,
            ),
        )
        model.fit(X_train, y_train)
        train_metrics = evaluate(model, X_train, y_train)
        val_metrics = evaluate(model, X_val, y_val)
        tree_results.append(
            {
                "max_depth": str(max_depth),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

        if best_tree_metrics is None or val_metrics[PRIMARY_METRIC] > best_tree_metrics[PRIMARY_METRIC]:
            best_tree_metrics = val_metrics
            best_tree_params = {"max_depth": max_depth}

    tree_df = pd.DataFrame(tree_results)
    print("\nDecision Tree validation results:")
    print(tree_df.to_string(index=False))
    print(
        f"Best tree by validation {PRIMARY_METRIC}: "
        f"max_depth={best_tree_params['max_depth']}, {format_metrics(best_tree_metrics)}"
    )

    # Train one default-style Random Forest to establish the bagging baseline.
    default_rf = make_pipeline(
        X_train,
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
            oob_score=True,
        ),
    )
    default_rf.fit(X_train, y_train)
    default_rf_metrics = evaluate(default_rf, X_val, y_val)
    default_rf_oob = float(default_rf.named_steps["model"].oob_score_)
    print("\nRandom Forest default validation result:")
    print(f"  {format_metrics(default_rf_metrics)}")
    print(f"  OOB accuracy={default_rf_oob:.4f}")

    rf_results = []
    best_rf_metrics = None
    best_rf_params = None
    best_rf_oob = None

    # Tune a small Random Forest grid to compare bias/variance settings.
    for params in RF_TUNE_GRID:
        model = make_pipeline(
            X_train,
            RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                random_state=RANDOM_STATE,
                n_jobs=1,
                oob_score=True,
            ),
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        oob_score = float(model.named_steps["model"].oob_score_)
        rf_results.append(
            {
                "n_estimators": params["n_estimators"],
                "max_depth": str(params["max_depth"]),
                "max_features": params["max_features"],
                "oob_accuracy": oob_score,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_rf_metrics is None or metrics[PRIMARY_METRIC] > best_rf_metrics[PRIMARY_METRIC]:
            best_rf_metrics = metrics
            best_rf_params = params.copy()
            best_rf_oob = oob_score

    rf_df = pd.DataFrame(rf_results)
    print("\nRandom Forest tuned validation results:")
    print(rf_df.to_string(index=False))
    print(
        f"Best Random Forest by validation {PRIMARY_METRIC}: "
        f"n_estimators={best_rf_params['n_estimators']}, "
        f"max_depth={best_rf_params['max_depth']}, "
        f"max_features={best_rf_params['max_features']}, "
        f"OOB accuracy={best_rf_oob:.4f}, {format_metrics(best_rf_metrics)}"
    )

    gb_results = []
    best_gb_metrics = None
    best_gb_params = None

    # Tune a small Gradient Boosting grid to compare sequential error-correction settings.
    for params in GB_TUNE_GRID:
        model = make_pipeline(
            X_train,
            GradientBoostingClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                random_state=RANDOM_STATE,
            ),
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        gb_results.append(
            {
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_gb_metrics is None or metrics[PRIMARY_METRIC] > best_gb_metrics[PRIMARY_METRIC]:
            best_gb_metrics = metrics
            best_gb_params = params.copy()

    gb_df = pd.DataFrame(gb_results)
    print("\nGradient Boosting validation results:")
    print(gb_df.to_string(index=False))
    print(
        f"Best Gradient Boosting by validation {PRIMARY_METRIC}: "
        f"n_estimators={best_gb_params['n_estimators']}, "
        f"learning_rate={best_gb_params['learning_rate']}, "
        f"max_depth={best_gb_params['max_depth']}, "
        f"{format_metrics(best_gb_metrics)}"
    )

    comparison_df = pd.DataFrame(
        [
            {
                "model": "Decision Tree",
                "key_hyperparameters": f"max_depth={best_tree_params['max_depth']}",
                "val_accuracy": best_tree_metrics["accuracy"],
                "val_balanced_accuracy": best_tree_metrics["balanced_accuracy"],
                "val_f1": best_tree_metrics["f1"],
            },
            {
                "model": "Random Forest",
                "key_hyperparameters": (
                    f"n_estimators={best_rf_params['n_estimators']}, "
                    f"max_depth={best_rf_params['max_depth']}, "
                    f"max_features={best_rf_params['max_features']}"
                ),
                "val_accuracy": best_rf_metrics["accuracy"],
                "val_balanced_accuracy": best_rf_metrics["balanced_accuracy"],
                "val_f1": best_rf_metrics["f1"],
            },
            {
                "model": "Gradient Boosting",
                "key_hyperparameters": (
                    f"n_estimators={best_gb_params['n_estimators']}, "
                    f"learning_rate={best_gb_params['learning_rate']}, "
                    f"max_depth={best_gb_params['max_depth']}"
                ),
                "val_accuracy": best_gb_metrics["accuracy"],
                "val_balanced_accuracy": best_gb_metrics["balanced_accuracy"],
                "val_f1": best_gb_metrics["f1"],
            },
        ]
    )
    print("\nValidation comparison table:")
    print(comparison_df.to_string(index=False))

    winner_name = "Decision Tree"
    winner_params = best_tree_params
    winner_metrics = best_tree_metrics

    if best_rf_metrics[PRIMARY_METRIC] > winner_metrics[PRIMARY_METRIC]:
        winner_name = "Random Forest"
        winner_params = best_rf_params
        winner_metrics = best_rf_metrics

    if best_gb_metrics[PRIMARY_METRIC] > winner_metrics[PRIMARY_METRIC]:
        winner_name = "Gradient Boosting"
        winner_params = best_gb_params
        winner_metrics = best_gb_metrics

    print("\nValidation winner:")
    print(f"  model={winner_name}, {format_metrics(winner_metrics)}")

    # Refit the single-tree baseline on train+validation for the final test comparison.
    final_tree = make_pipeline(
        X_train_val,
        DecisionTreeClassifier(
            max_depth=best_tree_params["max_depth"],
            random_state=RANDOM_STATE,
        ),
    )
    final_tree.fit(X_train_val, y_train_val)
    final_tree_test_metrics = evaluate(final_tree, X_test, y_test)

    # Refit the chosen winner on all non-test data before evaluating once on test.
    if winner_name == "Decision Tree":
        final_model = make_pipeline(
            X_train_val,
            DecisionTreeClassifier(
                max_depth=winner_params["max_depth"],
                random_state=RANDOM_STATE,
            ),
        )
    elif winner_name == "Random Forest":
        final_model = make_pipeline(
            X_train_val,
            RandomForestClassifier(
                n_estimators=winner_params["n_estimators"],
                max_depth=winner_params["max_depth"],
                max_features=winner_params["max_features"],
                random_state=RANDOM_STATE,
                n_jobs=1,
                oob_score=True,
            ),
        )
    else:
        final_model = make_pipeline(
            X_train_val,
            GradientBoostingClassifier(
                n_estimators=winner_params["n_estimators"],
                learning_rate=winner_params["learning_rate"],
                max_depth=winner_params["max_depth"],
                random_state=RANDOM_STATE,
            ),
        )

    final_model.fit(X_train_val, y_train_val)
    final_test_metrics = evaluate(final_model, X_test, y_test)

    print("\nFinal test evaluation:")
    print(f"  selected model={winner_name}")
    print(f"  selected model test: {format_metrics(final_test_metrics)}")
    print(f"  single-tree baseline test: {format_metrics(final_tree_test_metrics)}")
    print(
        f"  test {PRIMARY_METRIC} difference vs. single tree="
        f"{final_test_metrics[PRIMARY_METRIC] - final_tree_test_metrics[PRIMARY_METRIC]:.4f}"
    )


if __name__ == "__main__":
    main()
```
