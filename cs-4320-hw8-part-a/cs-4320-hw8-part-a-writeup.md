# Assignment 8 - Part A: SVMs and Interpretability

## 1. Shared Split and Pipeline Preprocessing

I used the instructor-provided churn dataset in [cs-4320-hw8-part-a.py](/c:/Users/levid/School_Programming/CS4320/cs-4320-hw8-part-a/cs-4320-hw8-part-a.py). Before splitting, I dropped the 195 rows with missing target labels because they cannot be used for supervised training. That left 2,591 labeled rows.

I created a stratified `60/20/20` train/validation/test split with `random_state=4320`:

| Split | Rows | Positive rate |
| --- | ---: | ---: |
| Train | 1554 | 0.436 |
| Validation | 518 | 0.436 |
| Test | 519 | 0.435 |

Preprocessing was done entirely inside a `Pipeline` using a `ColumnTransformer`:

* Numeric columns: median imputation + `StandardScaler`
* Categorical columns: most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`
* Excluded identifier: `customer_id`

I kept `F1` as the primary model-selection metric to stay consistent with Assignment 7.

## 2. Linear SVM Baseline

I trained a linear SVM with three values of `C`. The parameter `C` controls soft-margin behavior: smaller `C` allows a wider margin with more training violations, while larger `C` penalizes violations more heavily and pushes the model toward a tighter fit.

| Kernel | C | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 0.1 | 0.5637 | 0.0000 | 0.0000 | 0.0000 |
| Linear | 1.0 | 0.5502 | 0.2308 | 0.0133 | 0.0251 |
| Linear | 10.0 | 0.5502 | 0.2308 | 0.0133 | 0.0251 |

Pattern summary:

* At `C=0.1`, the classifier effectively collapsed to predicting the majority class on validation.
* Increasing `C` from `0.1` to `1.0` slightly increased recall, but F1 remained extremely low.
* Increasing `C` again to `10.0` produced no further gain, which suggests the main limitation was the linear decision boundary rather than under-tuning `C`.

Using the fitted linear model, the largest positive coefficient was `plan_type=Enterprise`, while `plan_type=Basic`, `Standard`, and `Pro` pushed predictions toward retention relative to that reference structure. That gives a simple coefficient-level explanation, but only within the assumptions of a linear boundary.

## 3. Kernelized SVM

I trained an RBF SVM and kept the search intentionally small by tuning only `C` and `gamma`.

| Kernel | C | Gamma | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RBF | 0.5 | scale | 0.5695 | 0.6154 | 0.0354 | 0.0669 |
| RBF | 0.5 | 0.1 | 0.5695 | 0.6000 | 0.0398 | 0.0747 |
| RBF | 0.5 | 1.0 | 0.5637 | 0.0000 | 0.0000 | 0.0000 |
| RBF | 1.0 | scale | 0.5347 | 0.4000 | 0.1327 | 0.1993 |
| RBF | 1.0 | 0.1 | 0.5405 | 0.4250 | 0.1504 | 0.2222 |
| RBF | 1.0 | 1.0 | 0.5541 | 0.4359 | 0.0752 | 0.1283 |
| RBF | 5.0 | scale | 0.5270 | 0.4457 | 0.3451 | 0.3890 |
| RBF | 5.0 | 0.1 | 0.5347 | 0.4581 | 0.3628 | 0.4049 |
| RBF | 5.0 | 1.0 | 0.5251 | 0.3611 | 0.1150 | 0.1745 |

Best validation model:

* Kernel: `RBF`
* `C=5.0`
* `gamma=0.1`
* Validation F1: `0.4049`

Tradeoff summary:

* Compared with the linear SVM, the RBF kernel captured non-linear structure and improved validation F1 substantially.
* Higher `C` helped more than lower `C` in this grid, which suggests the best model needed a somewhat tighter fit.
* Very large `gamma` (`1.0`) hurt performance, likely because the boundary became too local and did not generalize well.

## 4. Visualization and Interpretation

I created a 2D PCA projection of the validation data for visualization only and colored the points by the best model's decision score. The PCA fit was learned on the training data only, so the visualization did not leak validation or test information into training.

Plot:

* [hw8_part_a_decision_scores.png](/c:/Users/levid/School_Programming/CS4320/cs-4320-hw8-part-a/hw8_part_a_decision_scores.png)

Short interpretation:

* The first two PCA axes capture visible structure, but not enough to cleanly separate the classes by themselves.
* Points with large positive or negative decision scores are farther from the boundary, while points near zero are ambiguous cases.
* The model used 1,357 support vectors, which is a large share of the training set. That suggests the classes overlap substantially and the decision surface depends on many borderline examples rather than a small, simple separating boundary.

## 5. Interpretability Reflection

The linear SVM is easier to explain to a stakeholder because each feature has a signed coefficient. I can say that some inputs push the prediction toward churn and others push it toward retention, and I can rank those effects within the model. What I cannot claim is that the coefficient itself proves a real-world cause of churn. The weight only shows how the model uses that feature inside this particular representation and preprocessing pipeline.

The kernelized SVM is harder to explain directly because the decision is based on similarities in a transformed feature space rather than one transparent weight per original feature. I can explain that the model found non-linear combinations of behavior, billing, and plan attributes that better separated churn outcomes, and I can visualize which points are near or far from the boundary. What I cannot give as cleanly is a single global coefficient table with the same meaning as the linear model. In both models, "feature importance" is not causality: a predictive pattern can reflect correlation, confounding, or dataset-specific behavior without proving that changing the feature would change churn.

## 6. Final Test Evaluation

I selected the RBF SVM because it had the strongest validation F1. After that selection was fixed, I refit the chosen configuration on the combined train+validation data and evaluated once on the held-out test set.

Final test metrics for the selected model:

| Selected model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| RBF SVM (`C=5.0`, `gamma=0.1`) | 0.5067 | 0.4157 | 0.3274 | 0.3663 |

The test F1 is lower than the validation F1, so the chosen kernelized model generalized only moderately well. Even so, it remained clearly stronger than the linear SVM, which supports the conclusion that non-linear decision behavior mattered more than a purely linear margin on this dataset.

## Deliverables Included

* Code: [cs-4320-hw8-part-a.py](/c:/Users/levid/School_Programming/CS4320/cs-4320-hw8-part-a/cs-4320-hw8-part-a.py)
* Writeup: [cs-4320-hw8-part-a-writeup.md](/c:/Users/levid/School_Programming/CS4320/cs-4320-hw8-part-a/cs-4320-hw8-part-a-writeup.md)
* Visualization: [hw8_part_a_decision_scores.png](/c:/Users/levid/School_Programming/CS4320/cs-4320-hw8-part-a/hw8_part_a_decision_scores.png)
