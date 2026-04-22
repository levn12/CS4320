## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection
* **Data Modality:** Tabular
* **Task Type:** Classification
* **One-Sentence Goal:** Using current and voltage measurements from a simulated three-phase grid, detect whether a fault happened and build toward identifying which line combinations were involved.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Classification using logistric regression with evaluation and metrics (accuracy, precision, recall, F1, confusion matrix, ROC AUC, PR AUC, threshold selection).
* **Key Assumptions of This Technique:**
  * Predicted probabilities are meaningful enough to rank examples and support threshold tuning.
  * Train/validation/test splits are representative enough that metric comparisons transfer to held-out data.

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for my project because:

This week fits my project well because the hard part is not only training a model, but deciding how to evaluate it in a way that makes sense for fault detection. Missing real faults and raising false alarms are both important, and looking at threshold-based metrics helped make that tradeoff visible. Instead of only reporting one score, I could look at the model behavior in a more realistic way.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** Raw numeric phase measurements `Ia, Ib, Ic, Va, Vb, Vc`, with a binary target: `1` if any of `G/C/B/A` indicates fault, else `0`.
* **Why this representation was reasonable for this week:**
This was a good fit for the assignment because the focus was on evaluation and decision thresholds, not building a highly complex model. Keeping the target binary made it easier to interpret confusion matrices and understand how threshold changes impacted false positives and false negatives.

---

## 4. What Was Attempted

Using the same workflow as in part A of this assignment, I implemented a logistic regression classification model and evaluated its performance using metrics with the following process:

* Loaded data and built the binary target (any fault vs no fault).
* Used a stratified 60/20/20 train/validation/test split via a two-step split.
* Applied no-leakage preprocessing fit only on train (median imputation and StandardScaler).
* Trained logistic regression as a simple classifier baseline.
* Evaluated validation metrics at threshold 0.50 and compared against a tuned threshold.
* Tried multiple threshold-selection strategies on validation (starting with F1, then trying alternatives when confusion matrix behavior looked too one-sided).
* Final threshold policy: select the threshold by balancing **accuracy** and **balanced accuracy**, so one metric is not strong while the other is weak.
* Reported confusion matrices and final test metrics with the locked threshold.

What I intentionally did not attempt:

* No multiclass/multilabel line-combination classifier this week; even though I eventaully want to be able to predict fault types, this week's focus was more on model evaluation metrics, so I opted to keep the model behavior simpler to make the metrics easier to conceptually analyze.
* No complex models or hyperparameter sweeps; I kept model capacity fixed so metric interpretation stayed clear.

Constraints encountered:

* The dataset is simulated; real deployment data may have drift/noise patterns not represented here.
* Class imbalance can make default-threshold accuracy look better than the true error tradeoff, but to be fair, the data I'm working with isn't very imbalanced. However, the real-world data associated with my objective would be very imbalanced, so I tried to to take an approach that could handle class imbalance if necessary.

---

## 5. Results or Observations

Primary observations from this week are evaluation-oriented:

* Separating model training from threshold choice made the workflow much clearer.
* Different threshold objectives produced very different confusion matrices, even with the same model. In part A, choosing the threshold by maximizing F1 was sufficient to produce acceptable results, but on my data, maximizing F1 really imbalanced the confusion matrix. I also tried choosing threshold according to MCC and balanced accuracy, which are two other metrics I found while researching what the best option may be. I ended up choosing to minimize the difference between balanced accuracy and accuracy so no part of the matrix was given significantly higher priority than the other quadrants.
* Comparing accuracy with balanced accuracy helped avoid threshold choices that looked good on one metric but were poor in class balance.
* ROC AUC and PR AUC were still useful as threshold-independent checks.
* Final test metrics were computed only once after threshold selection on validation, which reduced leakage risk.

Results snapshot from the latest run:

* At threshold `0.71`, the test confusion matrix was `[[261, 212],[521, 579]]` (`TN=261`, `FP=212`, `FN=521`, `TP=579`). This is not ideal, but it is more balanced than earlier settings where one error type dominated. The final evaluation metrics at this threshold were `accuracy: 0.5340, balanced_acc: 0.5391, precision: 0.7320, recall: 0.5264, Test ROC AUC: 0.5637, Test PR AUC: 0.7896`

Operationally relevant pattern:

* For grid fault screening, recall matters because missed faults are costly. But pushing only for recall can hurt non-fault detection too much. Balancing accuracy and balanced accuracy gave me a more reasonable middle ground.

---

## 6. Interpretation and Judgment

This week was helpful because it shifted my focus from just "did the model classify" to "how is the model making mistakes." Trying multiple threshold strategies showed that threshold choice can completely change the confusion matrix behavior, even when the base model is unchanged. That made threshold tuning feel like a core design decision, not just a final tweak.

For this binary version, the assumptions mostly held: numeric features worked, and the split strategy allowed fair validation and testing. I ultimately used the accuracy vs balanced-accuracy threshold strategy because it gave a less skewed error pattern. The main limitation is that everything is still based on simulated data, so this is a baseline operating point rather than a final deployment setting.

---

## 7. Forward-Looking Adjustment

Next iteration, I would keep the same metric protocol and extend the task to also categorize fault types, not just binary fault detection:

1. Move from binary `any fault` to one-vs-rest labels for `G`, `C`, `B`, and `A` (or multilabel setup).
2. Evaluate per-label precision/recall/F1 and macro vs weighted summaries to capture imbalance.
3. Compare threshold policies for each label instead of using a single global threshold.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

This week's technique focus (classification metrics and threshold reasoning) is directly aligned with my capstone objective, but I question whether logistic regression was the best way to do it. My final accuracy score was much too low for me to happy with this model as the final result. Even though my target is classification, I think there must be a better method out there that works with the specific data I'm working with.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: **optional unless explicitly requested**
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results

The code I implemented this week is as follows:

```python
"""
CS 4320 - Assignment 5 (Part B)

Capstone assignment focused on classification evaluation and metric tradeoffs.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FAULT_COLS = ["G", "C", "B", "A"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"
RANDOM_STATE = 4320


def evaluate_at_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    # Convert probabilities into class labels using a candidate decision threshold.
    y_pred = (y_prob >= threshold).astype(int)
    return {
        # Standard accuracy (can be optimistic under class imbalance).
        "accuracy": accuracy_score(y_true, y_pred),
        # Balanced accuracy gives equal weight to positive and negative recall.
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        # Precision quantifies false-alarm control among predicted positives.
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # Recall quantifies missed-fault control among actual positives.
        "recall": recall_score(y_true, y_pred, zero_division=0),
        # F1 summarizes precision/recall tradeoff for positive class.
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def choose_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, str]:
    # Sweep thresholds on validation data; keep test data untouched for final evaluation only.
    threshold_grid = np.arange(0.10, 0.91, 0.01)
    # (threshold, accuracy, balanced_acc, precision, recall, f1, balance_score, gap)
    candidates: list[tuple[float, float, float, float, float, float, float, float]] = []

    for thr in threshold_grid:
        # Score each threshold by converting probabilities to hard labels.
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # Require both accuracy metrics to be strong, not just one of them.
        min_metric = min(float(acc), float(bal_acc))
        # Penalize thresholds where accuracy and balanced accuracy disagree heavily.
        gap = abs(float(acc) - float(bal_acc))
        # Higher is better: reward strong shared performance and penalize disagreement.
        balance_score = min_metric - gap
        candidates.append(
            (
                float(thr),
                float(acc),
                float(bal_acc),
                float(prec),
                float(rec),
                float(f1),
                float(balance_score),
                float(gap),
            )
        )

    # Pick the threshold with best combined balance; use F1 as a tiebreaker.
    chosen = max(candidates, key=lambda row: (row[6], row[5]))
    return chosen[0], (
        "Selected threshold to balance accuracy and balanced accuracy "
        f"(acc={chosen[1]:.4f}, balanced_acc={chosen[2]:.4f}, gap={chosen[7]:.4f}, "
        f"precision={chosen[3]:.4f}, recall={chosen[4]:.4f}, f1={chosen[5]:.4f})."
    )


def print_metric_block(title: str, metrics: dict[str, float]) -> None:
    # Pretty-print metric dictionaries in a consistent block format.
    print(f"\n{title}")
    print("-" * len(title))
    for metric_name, value in metrics.items():
        print(f"{metric_name:10s}: {value:.4f}")


def print_confusion(title: str, y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> None:
    # Generate confusion matrix at the requested threshold for error-type inspection.
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{title}")
    print("-" * len(title))
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def main() -> None:
    # Load capstone dataset from local project directory.
    df = pd.read_csv(DATA_PATH)

    # Build binary target: 1 if any fault flag is active, else 0.
    X = df.drop(columns=FAULT_COLS)
    y = (df[FAULT_COLS].sum(axis=1) > 0).astype(int)

    # Two-step stratified split to produce train/val/test = 60/20/20.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # Split remaining 80% into 75/25 to get final 60/20/20 proportions.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    numeric_features = X.columns.tolist()

    # Numeric preprocessing: median imputation then standardization.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # ColumnTransformer keeps preprocessing explicit and leakage-safe.
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features)],
        remainder="drop",
    )

    # Baseline linear classifier; thresholding policy handles operating-point tradeoff.
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    # Fit model using train split only.
    clf.fit(X_train, y_train)

    # Validation probabilities drive threshold tuning and comparison.
    val_probs = clf.predict_proba(X_val)[:, 1]
    # Keep 0.50 as a reference operating point.
    val_metrics_05 = evaluate_at_threshold(y_val, val_probs, threshold=0.50)

    # Select threshold balancing my objective on validation.
    chosen_threshold, threshold_reason = choose_threshold(y_val, val_probs)
    val_metrics_chosen = evaluate_at_threshold(y_val, val_probs, threshold=chosen_threshold)

    # Threshold-independent ranking metrics on validation.
    val_roc_auc = roc_auc_score(y_val, val_probs)
    val_precision_curve, val_recall_curve, _ = precision_recall_curve(y_val, val_probs)
    val_pr_auc = auc(val_recall_curve, val_precision_curve)

    print("Assignment 5 Part B: Classification Metrics for Capstone")
    print("========================================================")
    print(f"Rows: {len(df)}")
    print(
        f"Split proportions: train={len(X_train)/len(df):.2%}, "
        f"val={len(X_val)/len(df):.2%}, test={len(X_test)/len(df):.2%}"
    )
    print(
        f"Class balance (positive=any fault): train={y_train.mean():.2%}, "
        f"val={y_val.mean():.2%}, test={y_test.mean():.2%}"
    )

    print_metric_block("Validation metrics @ threshold=0.50", val_metrics_05)
    print(f"Validation ROC AUC : {val_roc_auc:.4f}")
    print(f"Validation PR AUC  : {val_pr_auc:.4f}")
    print_confusion(
        "Validation confusion matrix @ threshold=0.50",
        y_val,
        val_probs,
        threshold=0.50,
    )

    print(f"\nChosen threshold: {chosen_threshold:.2f}")
    print(threshold_reason)

    print_metric_block(
        f"Validation metrics @ chosen threshold={chosen_threshold:.2f}",
        val_metrics_chosen,
    )
    print_confusion(
        f"Validation confusion matrix @ threshold={chosen_threshold:.2f}",
        y_val,
        val_probs,
        threshold=chosen_threshold,
    )

    # Apply the locked threshold once on test for unbiased final reporting.
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_probs, threshold=chosen_threshold)
    # Include ranking metrics on test for completeness.
    test_roc_auc = roc_auc_score(y_test, test_probs)
    test_precision_curve, test_recall_curve, _ = precision_recall_curve(y_test, test_probs)
    test_pr_auc = auc(test_recall_curve, test_precision_curve)

    print_metric_block(
        f"Final TEST metrics @ threshold={chosen_threshold:.2f}",
        test_metrics,
    )
    print(f"Test ROC AUC      : {test_roc_auc:.4f}")
    print(f"Test PR AUC       : {test_pr_auc:.4f}")
    print_confusion(
        f"Test confusion matrix @ threshold={chosen_threshold:.2f}",
        y_test,
        test_probs,
        threshold=chosen_threshold,
    )

    print(
        "\nInterpretation: For this safety-oriented use case, higher recall reduces missed faults "
        "(false negatives), while higher precision reduces false alarms. The threshold policy should "
        "reflect operational costs of those two error types."
    )


if __name__ == "__main__":
    main()

```
