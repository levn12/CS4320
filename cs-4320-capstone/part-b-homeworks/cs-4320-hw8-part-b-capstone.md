## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection
* **Data Modality:** Tabular
* **Task Type:** Classification
* **One-Sentence Goal:** Use current and voltage measurements from a simulated three-phase electrical system to detect whether any fault occurred.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Support Vector Machines, including both linear SVMs and kernelized RBF SVMs, plus interpretability through coefficients and visualization.
* **Key Assumptions of This Technique:**
  * A separating margin exists that can distinguish the classes reasonably well in the chosen feature space.
  * Scaling matters because SVMs depend on distances and margin width/shape.
  * A non-linear kernel can help when a linear decision boundary is not flexible enough.

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for my project because:

My capstone data is fully numeric, fairly clean, and already represented as direct sensor measurements, which makes it a strong match for SVMs. The main question this week is whether the binary `any fault` target is better handled by a simple linear boundary or whether a kernelized boundary adds meaningful value. That makes the technique a good fit because SVMs let me test both options in the same leakage-safe pipeline while still keeping one model version interpretable.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** Raw numeric sensor values `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc`, with a binary proxy target of `1` for any fault and `0` for no fault.
* **Why this representation was reasonable for this week:**

This representation keeps the assignment focused on the model family itself rather than changing the project target. It also fits SVM assumptions well because the features are continuous, low-dimensional, and easy to standardize. I kept the same binary `any fault` target from the earlier capstone assignments so the new results stay comparable to prior weeks.

---

## 4. What Was Attempted

I implemented a Part B version of the Week 8 workflow using my electrical fault dataset. The main steps were:

1. Load the CSV data and define the binary target as `any fault` vs. `no fault`.
2. Separate the four fault-indicator columns from the six measurement features.
3. Build a reproducible `60/20/20` train/validation/test split using stratified random splitting with a seed `4320`.
4. Build a preprocessing pipeline with median imputation and standard scaling for the numeric features.
5. Train several linear SVM models across multiple `C` values and `class_weight` settings.
6. Print a coefficient snapshot from the best linear model for interpretation.
7. Train a small RBF SVM grid across `C`, `gamma`, and `class_weight`.
8. Select the winning model using validation `balanced_accuracy`.
9. Create a 2D visualization with phase a values from the dataset to inspect the boundary. In principle, each phase should produce a similar plot, so choosing the two features to plot is pretty straightforward. 
10. Fit the chosen model on train/validation and evaluate once on the held-out test set.

What I intentionally did not attempt:

* I did not move beyond the binary `any fault` target into multiclass or multilabel fault-type prediction.
* I did not run a large hyperparameter search, because I wanted the search to stay small and readable like the main assignment. The search that I did do took a fair amount of time to run, and it produced great results, so there was no need to go further.
* I did not treat the plot as part of model selection. It was used only for interpretation after the main comparison.

Constraints encountered:

* The binary `any fault` target appears to be much easier than a more detailed fault-classification task, so some models (like this one) can perform almost perfectly.
* Any 2D projection for visualization throws away some information from the full six-feature space, but for this data, looking at 2 features was informative enough to get an idea of what's going on.

---

## 5. Results or Observations

The main result this week is that the RBF SVM clearly outperformed the linear SVM on the electrical fault task, and the final selected model generalized extremely well on the test split.

Results snapshot from the run:

* Dataset rows: `7,861`
* Split sizes: `train=4716`, `val=1572`, `test=1573`
* Positive class rate across splits: about `0.699`
* Best linear validation setting: `C=1.0`, `class_weight="balanced"`
* Best linear validation balanced accuracy: `0.7102`
* Best RBF validation setting: `C=10.0`, `gamma=1.0`, `class_weight=None`
* Best RBF validation metrics: `accuracy=1.0000`, `balanced_accuracy=1.0000`, `precision=1.0000`, `recall=1.0000`, `f1=1.0000`
* Final selected winner: `RBF SVM`
* Final test metrics for the winner: `accuracy=0.9968`, `balanced_accuracy=0.9977`, `precision=1.0000`, `recall=0.9955`, `f1=0.9977`

Additional observations:

* The winning SVM performed spectacularly for this data. This tells me my data is highly seperable, and there is a clear distinciton in sensor readings when there is a fault in the circuit. This strategy worked much better on my data than it did on the given data for part a.
* The linear SVM was still useful for interpretation, but it was much weaker than the best RBF model on the main validation metric.
* Small `gamma` values often behaved like underpowered models and sometimes moved toward majority-class behavior.
* Moderate-to-large `gamma` values let the RBF SVM capture the structure of the electrical data very effectively.

Visualization observations:

* The 2D visualization used the fixed feature pair `Ia` and `Va`.
* The 2D visualization model reached validation `F1=0.8736`.
* The 2D visualization model used `1,484` support vectors.
* This plot was still useful for interpretation, but it captured less of the full six-feature structure than the complete model.
---

## 6. Interpretation and Judgment

This week was useful because it showed that the choice between a linear boundary and a non-linear boundary really matters for this project, even on the simplified binary `any fault` target. The linear SVM offered a cleaner interpretation through coefficients, but that simpler explanation came with a significant performance cost. The RBF SVM, while less transparent, fit the structure of the electrical measurements much better and produced nearly perfect validation and test performance.

That result says something important about the dataset itself. The binary electrical fault task appears to be highly separable in this feature space, especially once the SVM can use a non-linear boundary. This is encouraging in one sense, because it means the measurements do contain very strong signal about whether a fault occurred. At the same time, it also suggests the current project framing may be easier than the more detailed problem I eventually want to investigate. In other words, the technique worked very well, but part of that success may come from the simplicity of the binary proxy target rather than from solving the hardest version of the real problem.

The visualization was also useful. It gave me a principled 2D summary of the six-feature space. Even so, the plot is still only a compressed view, so it only helps interpretation without replacing the full model.

---

## 7. Forward-Looking Adjustment

The next change I would make is to keep the same leakage-safe SVM workflow, but move to a harder and more informative target:

1. Shift from binary `any fault` detection to one-vs-rest prediction for the specific fault indicators.
2. Check whether the RBF advantage still holds once the task becomes more detailed.
3. Keep the visualization idea, but use it only as an interpretation tool rather than a performance summary.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

There was hardly any mismatch this week. The general Week 8 assignment transferred well to my capstone data, but the interpretability discussion had to be adjusted because my data is already numeric and low-dimensional. That made the linear model easier to read than the churn example from Part A, while also making the binary electrical task much easier overall.

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code file included below: `cs-4320-hw8-part-b.py`


```python
# CS 4320 - Assignment 8 Part B
# Training linear and RBF kernel SVMs on the electrical fault dataset.

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


matplotlib.use("Agg")


FAULT_COLS = ["G", "C", "B", "A"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"
PLOT_PATH = BASE_DIR / "hw8_part_b_decision_boundary.png"

# Keep the split and search reproducible across runs.
RANDOM_STATE = 4320
# Use balanced accuracy as the main model-selection metric because the classes are imbalanced.
PRIMARY_METRIC = "balanced_accuracy"
LINEAR_C_VALUES = [0.1, 1.0, 10.0, 100.0]
RBF_C_VALUES = [0.1, 1.0, 10.0, 100.0]
RBF_GAMMA_VALUES = [0.001, 0.01, 0.1, 1.0, "scale"]
CLASS_WEIGHT_VALUES = [None, "balanced"]
# Keep one fixed 2D view for the visualization so the plot stays simple.
VIS_FEATURES = ["Ia", "Va"]


def load_data(data_path: Path = DATA_PATH):
    # Read the electrical fault dataset from disk and build the binary any-fault target.
    df = pd.read_csv(data_path)
    y = (df[FAULT_COLS].sum(axis=1) > 0).astype(int).to_numpy()
    X = df.drop(columns=FAULT_COLS)
    return df, X, y


def build_preprocessor(X: pd.DataFrame):
    # The electrical measurements are all numeric, so median imputation + scaling are enough here.
    numeric_cols = X.columns.tolist()
    return ColumnTransformer(
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
            )
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(preprocessor, *, kernel: str, c_value: float, gamma="scale", class_weight=None):
    # Store the SVM settings before building the full preprocessing + model pipeline.
    model_kwargs = {
        "kernel": kernel,
        "C": c_value,
        "class_weight": class_weight,
        "random_state": RANDOM_STATE,
    }
    if kernel == "rbf":
        model_kwargs["gamma"] = gamma

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", SVC(**model_kwargs)),
        ]
    )


def evaluate(model, X, y):
    # Compute the assignment's classification metrics for a set of predictions.
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def format_metrics(metrics):
    # Format metric values into one short printable line.
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )


def print_split_info(y_train, y_val, y_test):
    # Print the number of examples in each split and the fault rate.
    print("Split sizes:")
    print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print("Class balance (positive = any fault):")
    print(f"  train: positives={int(y_train.sum())}, rate={y_train.mean():.3f}")
    print(f"  val:   positives={int(y_val.sum())}, rate={y_val.mean():.3f}")
    print(f"  test:  positives={int(y_test.sum())}, rate={y_test.mean():.3f}")


def print_linear_interpretation(linear_model, X_train):
    # Show the largest positive and negative linear coefficients for interpretation.
    preprocessor = linear_model.named_steps["preprocessor"]
    svm = linear_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out(X_train.columns)
    coefficients = svm.coef_.ravel()

    coef_df = pd.DataFrame({"feature": feature_names, "weight": coefficients})
    positive = coef_df[coef_df["weight"] > 0].sort_values("weight", ascending=False).head(5)
    negative = coef_df[coef_df["weight"] < 0].sort_values("weight", ascending=True).head(5)

    print("\nLinear SVM coefficient snapshot:")
    print("  Strongest push toward fault=1:")
    if len(positive):
        print(positive.to_string(index=False))
    else:
        print("  No positive coefficients in this fitted model.")
    print("  Strongest push toward fault=0:")
    if len(negative):
        print(negative.to_string(index=False))
    else:
        print("  No negative coefficients in this fitted model.")


def save_visualization(best_model, X_train, y_train, X_val, y_val):
    # Save a 2D decision-boundary plot using one current and one voltage feature.
    svm = best_model.named_steps["model"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_pair = imputer.fit_transform(X_train[VIS_FEATURES])
    X_val_pair = imputer.transform(X_val[VIS_FEATURES])
    X_train_2d = scaler.fit_transform(X_train_pair)
    X_val_2d = scaler.transform(X_val_pair)

    viz_params = {
        "kernel": svm.kernel,
        "C": svm.C,
        "class_weight": svm.class_weight,
        "random_state": RANDOM_STATE,
    }
    if svm.kernel == "rbf":
        viz_params["gamma"] = svm.gamma

    viz_model = SVC(**viz_params)
    viz_model.fit(X_train_2d, y_train)

    x_min = min(X_train_2d[:, 0].min(), X_val_2d[:, 0].min()) - 0.6
    x_max = max(X_train_2d[:, 0].max(), X_val_2d[:, 0].max()) + 0.6
    y_min = min(X_train_2d[:, 1].min(), X_val_2d[:, 1].min()) - 0.6
    y_max = max(X_train_2d[:, 1].max(), X_val_2d[:, 1].max()) + 0.6
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    decision_surface = viz_model.decision_function(grid).reshape(xx.shape)
    predicted_regions = viz_model.predict(grid).reshape(xx.shape)
    support_points_2d = viz_model.support_vectors_

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.contourf(
        xx,
        yy,
        predicted_regions,
        levels=[-0.5, 0.5, 1.5],
        alpha=0.08,
    )
    ax.contour(
        xx,
        yy,
        decision_surface,
        levels=[-1, 0, 1],
        colors=["#7a7a7a", "#000000", "#7a7a7a"],
        linestyles=[":", "-", ":"],
        linewidths=[0.9, 2.8, 0.9],
    )
    ax.scatter(
        X_val_2d[y_val == 0, 0],
        X_val_2d[y_val == 0, 1],
        c="tab:blue",
        marker="o",
        s=32,
        alpha=0.7,
        label="No fault (0)",
    )
    ax.scatter(
        X_val_2d[y_val == 1, 0],
        X_val_2d[y_val == 1, 1],
        c="tab:orange",
        marker="^",
        s=36,
        alpha=0.7,
        label="Any fault (1)",
    )
    ax.scatter(
        support_points_2d[:, 0],
        support_points_2d[:, 1],
        facecolors="none",
        edgecolors="black",
        linewidths=0.5,
        s=28,
        alpha=0.45,
        label="Support vectors",
    )
    ax.grid(True, alpha=0.15)
    ax.set_xlabel(f"{VIS_FEATURES[0]} (standardized)")
    ax.set_ylabel(f"{VIS_FEATURES[1]} (standardized)")
    ax.set_title("Electrical fault SVM decision boundary using two selected features")
    ax.legend(loc="best", frameon=True)
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    pair_score = f1_score(y_val, viz_model.predict(X_val_2d), zero_division=0)
    return pair_score, len(support_points_2d)


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

    print("Assignment 8 Part B - SVMs on Electrical Fault Data")
    print("====================================================")
    print(f"Rows: {len(df)}")
    print_split_info(y_train, y_val, y_test)
    print(f"Primary validation metric: {PRIMARY_METRIC}")

    linear_results = []
    best_linear_model = None
    best_linear_metrics = None
    best_linear_params = None

    # Try several C values and class-weight settings for the linear SVM.
    for c_value in LINEAR_C_VALUES:
        for class_weight in CLASS_WEIGHT_VALUES:
            model = make_pipeline(
                build_preprocessor(X_train),
                kernel="linear",
                c_value=c_value,
                class_weight=class_weight,
            )
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val)
            linear_results.append(
                {
                    "kernel": "linear",
                    "C": c_value,
                    "class_weight": class_weight,
                    **metrics,
                }
            )

            if best_linear_metrics is None or metrics[PRIMARY_METRIC] > best_linear_metrics[PRIMARY_METRIC]:
                best_linear_model = model
                best_linear_metrics = metrics
                best_linear_params = {"C": c_value, "class_weight": class_weight}

    linear_df = pd.DataFrame(linear_results)
    print("\nLinear SVM validation results:")
    print(linear_df.to_string(index=False))
    print(
        "\nSoft-margin note: smaller C allows a wider margin with more violations, "
        "while larger C penalizes violations more and pushes the model toward a tighter fit."
    )
    print(
        f"Best linear setting by validation {PRIMARY_METRIC}: "
        f"C={best_linear_params['C']}, class_weight={best_linear_params['class_weight']}"
    )
    print_linear_interpretation(best_linear_model, X_train)

    rbf_results = []
    best_rbf_model = None
    best_rbf_metrics = None
    best_rbf_params = None

    # Try a small RBF grid over C, gamma, and class_weight to compare a non-linear decision boundary.
    for c_value in RBF_C_VALUES:
        for gamma_value in RBF_GAMMA_VALUES:
            for class_weight in CLASS_WEIGHT_VALUES:
                model = make_pipeline(
                    build_preprocessor(X_train),
                    kernel="rbf",
                    c_value=c_value,
                    gamma=gamma_value,
                    class_weight=class_weight,
                )
                model.fit(X_train, y_train)
                metrics = evaluate(model, X_val, y_val)
                rbf_results.append(
                    {
                        "kernel": "rbf",
                        "C": c_value,
                        "gamma": gamma_value,
                        "class_weight": class_weight,
                        **metrics,
                    }
                )

                if best_rbf_metrics is None or metrics[PRIMARY_METRIC] > best_rbf_metrics[PRIMARY_METRIC]:
                    best_rbf_model = model
                    best_rbf_metrics = metrics
                    best_rbf_params = {"C": c_value, "gamma": gamma_value, "class_weight": class_weight}

    rbf_df = pd.DataFrame(rbf_results)
    print("\nRBF SVM validation results:")
    print(rbf_df.to_string(index=False))
    print(
        f"Best RBF setting by validation {PRIMARY_METRIC}: "
        f"C={best_rbf_params['C']}, gamma={best_rbf_params['gamma']}, "
        f"class_weight={best_rbf_params['class_weight']}"
    )

    winner_name = "linear"
    winner_model = best_linear_model
    winner_metrics = best_linear_metrics

    # Switch to the RBF model if it beats the linear model on the validation metric.
    if best_rbf_metrics[PRIMARY_METRIC] > best_linear_metrics[PRIMARY_METRIC]:
        winner_name = "rbf"
        winner_model = best_rbf_model
        winner_metrics = best_rbf_metrics

    print("\nValidation winner:")
    print(f"  kernel={winner_name}, {format_metrics(winner_metrics)}")

    pair_score, support_count = save_visualization(winner_model, X_train, y_train, X_val, y_val)
    print(f"Saved visualization: {PLOT_PATH}")
    print(
        "Visualization note: the plot uses a simple two-feature view with "
        f"{VIS_FEATURES[0]} and {VIS_FEATURES[1]}."
    )
    print(f"  2D visualization F1 with this feature pair: {pair_score:.4f}")
    print(f"  support vectors in the 2D visualization model: {support_count}")

    # Refit the chosen configuration on train+validation data, then evaluate once on test.
    if winner_name == "linear":
        final_model = make_pipeline(
            build_preprocessor(X_train_val),
            kernel="linear",
            c_value=best_linear_params["C"],
            class_weight=best_linear_params["class_weight"],
        )
    else:
        final_model = make_pipeline(
            build_preprocessor(X_train_val),
            kernel="rbf",
            c_value=best_rbf_params["C"],
            gamma=best_rbf_params["gamma"],
            class_weight=best_rbf_params["class_weight"],
        )

    final_model.fit(X_train_val, y_train_val)
    test_metrics = evaluate(final_model, X_test, y_test)

    print("\nFinal test evaluation:")
    print(f"  selected kernel={winner_name}")
    if winner_name == "linear":
        print(
            f"  selected C={best_linear_params['C']}, "
            f"class_weight={best_linear_params['class_weight']}"
        )
    else:
        print(
            f"  selected C={best_rbf_params['C']}, gamma={best_rbf_params['gamma']}, "
            f"class_weight={best_rbf_params['class_weight']}"
        )
    print(f"  {format_metrics(test_metrics)}")


if __name__ == "__main__":
    main()

```
