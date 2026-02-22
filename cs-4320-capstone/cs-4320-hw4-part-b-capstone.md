## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection
* **Data Modality:** Tabular
* **Task Type:** Classification
* **One-Sentence Goal:** Using current and voltage values of a simulated electrical grid, predict whether an electrical fault occurred, and eventually classify the faulted line(s).

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Logistic regression baseline trained with explicit gradient descent (regression-style linear model + optimization).
* **Key Assumptions of This Technique:**
  * The relationship between features and fault probability is approximately linear in log-odds space.
  * Rows are independent and identically distributed across train/validation/test.
  * Scaled numeric features are sufficient for a first-pass baseline.

**Fit Assessment (required):**

> I expect this technique to be a **partial** fit for my project because:

My true capstone goal is classification, so classical linear regression is not the right objective by itself. I used a closely related regression-style baseline (logistic regression) that still uses a linear predictor, explicit loss, and gradient-based optimization. This gave me a simple, interpretable model to test optimization behavior and establish a baseline before trying more expressive models.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** Numeric feature vectors using raw phase measurements: `Ia, Ib, Ic, Va, Vb, Vc`.
* **Why this representation was reasonable for this week:**  
The assignment focus was optimization and loss behavior, not feature engineering complexity. These six values are direct physical measurements related to grid behavior and fault conditions. I also used a simplified proxy target for this week: `fault vs no-fault`, where the label is `1` if any of `G/C/B/A` is faulted and `0` otherwise.

---

## 4. What Was Attempted

This week I implemented a full baseline pipeline in `cs-4320-hw4-part-b.py`:

* Loaded `electrical_fault_data.csv` (7,861 rows)
* Built binary target from fault columns (`G, C, B, A`) to represent **any fault**
* Split data into train/validation/test with reproducible seed (`70/15/15` equivalent using two-step split)
* Preprocessed with scikit-learn on train only:
  * `SimpleImputer(strategy="median")`
  * `StandardScaler()`
* Implemented logistic regression training manually with batch gradient descent:
  * Sigmoid output
  * Binary cross-entropy loss
  * Vectorized gradient update
* Tracked train and validation loss over epochs and saved a loss plot
* Evaluated on held-out test set using accuracy, precision, recall, and F1

What I intentionally did not attempt:

* No advanced model families (trees, ensembles, neural nets)
* No hyperparameter search/tuning sweep
* No direct multiclass/multilabel fault-type model in final version
* No additional engineered features (kept representation simple on purpose)

Constraints encountered:

* The dataset is simulated and may not reflect full real-world noise behavior
* Class imbalance exists (`fault` rows > `no-fault` rows), which can bias threshold-based behavior
* Time/scope constraints favored a clear baseline over model complexity

---

## 5. Results or Observations

Observed optimization behavior:

* Training remained stable (no divergence/exploding loss)
* Binary cross-entropy decreased early and then plateaued:
  * Around epoch 100: train `0.60995`, val `0.60992`
  * Around epoch 1200: train `0.60092`, val `0.60223`
* This indicates convergence/stagnation rather than instability

Test metrics:

* Accuracy: `0.6992`
* Precision: `0.6992`
* Recall: `1.0000`
* F1: `0.8229`

Qualitative observation:

* The model strongly favors predicting the positive (`fault`) class, consistent with high recall and moderate precision. This is informative as a baseline but not sufficient by itself for detailed line-level diagnosis.

---

## 6. Interpretation and Judgment

This baseline was informative for the assignment goals even though it is simple. I was able to clearly define an objective (binary cross-entropy), observe optimization behavior over epochs, and verify that gradient-based updates converged. The learning curve shape (initial drop followed by flattening) suggests the optimizer reached the limit of what this linear boundary can capture under current features and setup.

The regression-style assumptions were partially valid. Numeric feature representation and scaling worked cleanly, and optimization behaved as expected. However, the task itself is fundamentally classification and likely contains nonlinear structure and correlated line-fault patterns that a single linear-logistic baseline cannot fully represent. The result is a useful reference point, not a final solution.

---

## 7. Forward-Looking Adjustment

Before the next assignment, I will keep the same no-leakage preprocessing and train/val/test protocol, but change model scope in one of two directions:

1. Expand from binary fault detection to line-level prediction (`G/C/B/A`) using one-vs-rest logistic baselines.
2. Compare this baseline against a nonlinear model to see whether the plateau is a model-capacity limit rather than an optimization issue.

I will also evaluate class imbalance effects more explicitly (for example, threshold sensitivity and class-distribution-aware metrics).

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

Classical regression is a poor direct fit for my task because the output is categorical fault state rather than a continuous quantity. I used logistic regression as an analogous regression-style baseline because it preserves the core weekly requirements (explicit loss and gradient-based optimization) while remaining aligned with binary labels. This mismatch clarification still provided value by confirming that optimization was stable and by establishing a transparent baseline for future model comparisons.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code or notebooks: **optional unless explicitly requested**
* Performance is **not** graded competitively
* Clear reasoning and honest reflection matter more than results
