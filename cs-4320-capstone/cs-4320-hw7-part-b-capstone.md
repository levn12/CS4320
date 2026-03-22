## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection
* **Data Modality:** Tabular
* **Task Type:** Classification
* **One-Sentence Goal:** Use current/voltage measurements from a simulated three-phase electrical system to detect whether any fault occurred.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Comparing model families for classification, specifically Naive Bayes and k-Nearest Neighbors.
* **Key Assumptions of This Technique:**
  * Naive Bayes assumes the features are conditionally independent within each class, and the chosen Naive Bayes variant should match the data type.
  * kNN assumes that examples from the same class tend to be close together in feature space under the chosen distance metric.

**Fit Assessment (required):**

> I expect this technique to be a **partial** fit for my project because:

This week fits my project because I can still compare two different model families on the same binary electrical fault task. However, the exact Part A setup could not be copied over directly because `MultinomialNB` is designed for count-based data such as bag-of-words text, while my capstone data contains continuous current and voltage measurements, including negative values. Because of that mismatch, I adapted the Naive Bayes side to use `GaussianNB`, which is a more reasonable choice for continuous numeric features.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** Raw numeric sensor values `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc`, with a binary proxy target of `1` for any fault and `0` for no fault.
* **Why this representation was reasonable for this week:**

This representation kept the comparison focused on model-family behavior instead of adding extra complexity from multiclass or multilabel fault-type prediction. It also matched the assumptions of both models reasonably well: kNN can compare nearby numeric measurement patterns, and `GaussianNB` can model each numeric feature with a class-conditional Gaussian distribution.

---

## 4. What Was Attempted

I implemented a Part B version of the Week 7 workflow using my electrical fault dataset. The main steps were:

1. Load the CSV data and define the binary target as `any fault` vs. `no fault`.
2. Separate the four fault-indicator columns from the measurement features.
3. Build a reproducible `60/20/20` train/validation/test split using stratified random splitting with `random_state=4320`.
4. Train a `GaussianNB` model on standardized numeric features.
5. Train multiple kNN models over several `k` values on the same standardized features.
6. Compare the models on validation F1, accuracy, precision, and recall.
7. Select the better model family using validation F1 only.
8. Perform error analysis on validation mistakes.
9. Evaluate the selected winner once on the held-out test set.

What I intentionally did not attempt:

* I did not use `MultinomialNB`, because that would not be a sound match for continuous sensor values.
* I did not move to multiclass or multilabel fault-type prediction yet, because I wanted to keep the family comparison scoped to one binary task.
* I did not add extensive feature engineering, since the assignment focus was model-family comparison rather than inventing a new representation.

Constraints encountered:

* The dataset does not come with instructor-provided fixed splits, so I had to create my own reproducible split.
* The binary `any fault` target is simpler than the eventual full fault-type identification task I want to work toward.
* Very strong kNN performance makes interpretation harder, because I have to ask whether the result reflects true separability or just an unusually easy binary target.

---

## 5. Results or Observations

The biggest observation from this week is that kNN performed almost perfectly on validation and then achieved a perfect score on the final test evaluation. In contrast, the Naive Bayes side was still strong, but clearly weaker than the best kNN result.

Results snapshot from the run:

* `GaussianNB` validation metrics: `accuracy=0.9739`, `precision=1.0000`, `recall=0.9627`, `f1=0.9810`
* kNN validation metrics:
  * `k=1`: `accuracy=1.0000`, `precision=1.0000`, `recall=1.0000`, `f1=1.0000`
  * `k=3`: `accuracy=1.0000`, `precision=1.0000`, `recall=1.0000`, `f1=1.0000`
  * `k=5`: `accuracy=0.9994`, `precision=1.0000`, `recall=0.9991`, `f1=0.9995`
  * `k=7`: `accuracy=0.9994`, `precision=1.0000`, `recall=0.9991`, `f1=0.9995`
  * `k=11`: `accuracy=0.9981`, `precision=0.9982`, `recall=0.9991`, `f1=0.9986`
  * `k=100`: `accuracy=0.9625`, `precision=1.0000`, `recall=0.9463`, `f1=0.9724`
* Best validation choice: `k=1`
* Final selected winner: `kNN (k=1)`
* Final test metrics for the winner: `accuracy=1.0000`, `precision=1.0000`, `recall=1.0000`, `f1=1.0000`

Additional observations:

* `GaussianNB` is a reasonable match for the data type, but its independence and Gaussian-shape assumptions are still pretty strong.
* kNN was tested across several `k` values, and even moderate values like `5` and `7` were still almost perfect.
* All preprocessing was done inside a pipeline, so scaling was fit on training data only and then reused safely on validation and test data.
* Error analysis was included so the output was not just numeric, but also showed representative mistakes.
* `GaussianNB` made `41` validation errors, while the selected kNN configuration made none on the validation split.

One important behavior is that the kNN results were so strong that I had to think carefully about leakage risk. After checking the code logic, there is no obvious direct leakage in the workflow itself: the target columns are removed from the feature matrix, the split happens before fitting the pipeline, and the winner is selected on validation before touching the test set. That suggests the perfect result is more likely due to the binary target being highly separable in the measurement space than to a simple coding bug.

---

## 6. Interpretation and Judgment

This week was useful because it forced me to think about whether a model family can be transferred directly from one domain to another. The answer here was "not exactly." The overall comparison idea from Part A still made sense, but the specific Naive Bayes variant had to change because the electrical fault data is numeric and continuous rather than count-based. That adaptation itself was an important result, because it showed that applying a model family responsibly means checking whether its assumptions actually match the data representation.

The comparison also says something interesting about the dataset. Since kNN reached perfect validation performance at `k=1` and perfect final test performance as well, local neighborhoods in the sensor feature space are likely very cleanly separated for the binary `any fault` target. That could mean the task is genuinely easy in this representation, or it could mean that this binary proxy is much simpler than the eventual full project goal. Either way, the result suggests that future work should not only ask which model is better, but also whether the task definition is still challenging enough to be informative for the real project.

---

## 7. Forward-Looking Adjustment

The next change I would make is to keep the same leakage-safe family-comparison workflow, but move beyond the binary `any fault` target:

1. Compare model families on a more specific target, such as one-vs-rest prediction for individual fault indicators.
2. Measure whether kNN is still dominant once the label space becomes more detailed.
3. Add modest feature engineering only after the model-family comparison is established on the harder target.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

There was a partial mismatch this week. The assignment theme of comparing model families was relevant, but the exact `MultinomialNB` vs. kNN pairing from the text assignment was not directly transferable to my capstone data. The value of the attempt was that it made the representation issue explicit: once the data type changed from sparse word counts to continuous sensor measurements, I had to swap in `GaussianNB` to preserve a valid comparison.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code file included below: `cs-4320-hw7-part-b.py`


```python
# CS 4320 - Assignment 7 (Part B)
# Comparing model families on electrical fault detection: GaussianNB vs. kNN.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FAULT_COLS = ["G", "C", "B", "A"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electrical_fault_data.csv"
RANDOM_STATE = 4320
K_VALUES = [1, 3, 5, 7, 11, 100]


# Load the capstone dataset and build the binary target used in earlier assignments.
def load_data(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path)
    X = df.drop(columns=FAULT_COLS)
    y = (df[FAULT_COLS].sum(axis=1) > 0).astype(int).to_numpy()
    row_ids = df.index.to_numpy()
    return df, X, y, row_ids


# Compute the classification metrics used to compare the model families.
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# Format a metric dictionary into a single printable line.
def format_metrics(metrics):
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )


# Print split sizes and positive-class rates for the reproducible train/val/test partition.
def print_split_info(y_train, y_val, y_test):
    print("Split sizes:")
    print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print("Class balance (positive = any fault):")
    print(f"  train: positives={int(y_train.sum())}, rate={y_train.mean():.3f}")
    print(f"  val:   positives={int(y_val.sum())}, rate={y_val.mean():.3f}")
    print(f"  test:  positives={int(y_test.sum())}, rate={y_test.mean():.3f}")


# Show a few representative false negatives and false positives for qualitative discussion.
def print_error_analysis(model_name, model, X, y, row_ids):
    y_pred = model.predict(X)
    wrong_positions = np.where(y_pred != y)[0]
    if len(wrong_positions) == 0:
        print(f"No mistakes for {model_name} on this split.")
        return

    wrong_ids = row_ids[wrong_positions]
    wrong_df = X.iloc[wrong_positions].copy()
    wrong_df.insert(0, "row_id", wrong_ids)
    wrong_df["y_true"] = y[wrong_positions]
    wrong_df["y_pred"] = y_pred[wrong_positions]

    print(f"\n{model_name} error analysis: {len(wrong_positions)} misclassified rows")

    fn = wrong_df[(wrong_df["y_true"] == 1) & (wrong_df["y_pred"] == 0)].head(3)
    fp = wrong_df[(wrong_df["y_true"] == 0) & (wrong_df["y_pred"] == 1)].head(3)

    if len(fn):
        print("  False negatives (fault predicted no fault):")
        for row in fn.itertuples(index=False):
            print(
                "    "
                f"row={row.row_id}, Ia={row.Ia:.3f}, Ib={row.Ib:.3f}, Ic={row.Ic:.3f}, "
                f"Va={row.Va:.3f}, Vb={row.Vb:.3f}, Vc={row.Vc:.3f}"
            )
    if len(fp):
        print("  False positives (no fault predicted fault):")
        for row in fp.itertuples(index=False):
            print(
                "    "
                f"row={row.row_id}, Ia={row.Ia:.3f}, Ib={row.Ib:.3f}, Ic={row.Ic:.3f}, "
                f"Va={row.Va:.3f}, Vb={row.Vb:.3f}, Vc={row.Vc:.3f}"
            )


def main():
    # Load the capstone data and keep the same binary target definition as HW5/HW6 Part B.
    df, X, y, row_ids = load_data(DATA_PATH)

    # Create a reproducible 60/20/20 stratified split because this dataset does not ship with fixed ids.
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print("HW7 Part B: Comparing GaussianNB and kNN on Electrical Fault Data")
    print("=================================================================")
    print(f"Rows: {len(df)}")
    print("Naive Bayes note: MultinomialNB is not appropriate here because these features are")
    print("continuous sensor readings that include negative values, so GaussianNB is the right NB variant.")
    print_split_info(y_train, y_val, y_test)

    # Train GaussianNB on standardized numeric features.
    gnb_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("gnb", GaussianNB()),
        ]
    )
    gnb_model.fit(X_train, y_train)

    y_val_pred_gnb = gnb_model.predict(X_val)
    gnb_val_metrics = evaluate(y_val, y_val_pred_gnb)
    print("\nGaussianNB validation metrics:")
    print(format_metrics(gnb_val_metrics))
    print("Why GaussianNB? Each feature is continuous, so the model assumes class-conditional Gaussian distributions.")

    # Try several neighborhood sizes for kNN on the same standardized features.
    knn_results = []

    for k in K_VALUES:
        knn_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=k,
                        metric="euclidean",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        knn_model.fit(X_train, y_train)
        y_val_pred_knn = knn_model.predict(X_val)
        metrics_knn = evaluate(y_val, y_val_pred_knn)
        knn_results.append({"k": k, **metrics_knn})

    knn_df = pd.DataFrame(knn_results)
    print("\nValidation results for kNN:")
    print(knn_df.to_string(index=False))

    # Select the best k using validation F1 so model selection stays off the test split.
    best_row = knn_df.loc[knn_df["f1"].idxmax()]
    best_k = int(best_row["k"])
    best_knn_val_metrics = {
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }
    print(f"Best k selected by validation F1: k={best_k}")

    # Refit the best kNN model on the original training split for the final comparison.
    best_knn = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=best_k,
                    metric="euclidean",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    best_knn.fit(X_train, y_train)

    # Print the validation comparison before looking at qualitative errors.
    print("\nValidation comparison:")
    print("model, key_hyperparams, accuracy, precision, recall, f1")
    print("GaussianNB, standardized numeric features, " + format_metrics(gnb_val_metrics))
    print(f"kNN (k={best_k}, euclidean), standardized numeric features, " + format_metrics(best_knn_val_metrics))

    print_error_analysis("GaussianNB", gnb_model, X_val, y_val, ids_val)
    print_error_analysis("kNN", best_knn, X_val, y_val, ids_val)

    # Choose the final model from validation performance only, then evaluate once on test.
    winner_name = "GaussianNB"
    winner_model = gnb_model
    winner_reason = (
        "GaussianNB had the higher validation F1, suggesting the class-conditional distribution assumption "
        "fit the sensor measurements better than local-neighbor voting."
    )

    if best_knn_val_metrics["f1"] > gnb_val_metrics["f1"]:
        winner_name = f"kNN (k={best_k})"
        winner_model = best_knn
        winner_reason = (
            "kNN had the higher validation F1, suggesting nearby sensor patterns were more informative "
            "than the Gaussian independence assumptions."
        )

    print(f"\nWinner selected by validation F1: {winner_name}")
    print(f"Selection note: {winner_reason}")
    y_test_pred = winner_model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred)
    print("Final test metrics for winner:")
    print(format_metrics(test_metrics))


if __name__ == "__main__":
    main()
```
