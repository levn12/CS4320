# CS 4320 - Assignment 8 Part A
# Training linear and RBF kernel SVMs on the churn dataset.

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "assignment8_svm_dataset.csv"
PLOT_PATH = BASE_DIR / "hw8_part_a_decision_scores.png"

# Keep the split and search reproducible across runs.
RANDOM_STATE = 4320
# Use balanced accuracy as the main model-selection metric.
PRIMARY_METRIC = "balanced_accuracy"
LINEAR_C_VALUES = [0.1, 1.0, 10.0, 100.0]
RBF_C_VALUES = [1.0, 10.0, 100.0]
RBF_GAMMA_VALUES = [0.001, 0.01, 0.1, 1.0, "scale"]
CLASS_WEIGHT_VALUES = [None, "balanced"]
# Keep one fixed 2D view for the visualization so the plot stays simple.
VIS_FEATURES = ["engagement_score", "tenure_months"]


def load_data(data_path: Path = DATA_PATH):
    # Read the churn dataset from disk.
    df = pd.read_csv(data_path)

    # Drop rows with missing labels and remove the identifier from the feature set.
    df = df.dropna(subset=["churned_next_month"]).reset_index(drop=True)
    y = df["churned_next_month"].astype(int).to_numpy()
    X = df.drop(columns=["customer_id", "churned_next_month"])

    return df, X, y


def build_preprocessor(X: pd.DataFrame):
    # Split the columns by type so numeric and categorical data are handled appropriately.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # Keep all preprocessing inside the pipeline so imputation and scaling are learned on training data only.
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
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
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


# Compute the assignment's classification metrics for a set of predictions.
def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


# Format metric values into one short printable line.
def format_metrics(metrics):
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )


# Print the number of examples in each split and the churn rate.
def print_split_info(y_train, y_val, y_test):
    print("Split sizes:")
    print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print("Class balance (positive = churn next month):")
    print(f"  train: positives={int(y_train.sum())}, rate={y_train.mean():.3f}")
    print(f"  val:   positives={int(y_val.sum())}, rate={y_val.mean():.3f}")
    print(f"  test:  positives={int(y_test.sum())}, rate={y_test.mean():.3f}")


def get_feature_names(preprocessor, X: pd.DataFrame):
    # Recover the transformed feature names after one-hot encoding.
    return preprocessor.get_feature_names_out(X.columns)


# Show the largest positive and negative linear coefficients for interpretability discussion.
def print_linear_interpretation(linear_model, X_train):
    # Pull the fitted preprocessing and model steps out of the pipeline.
    preprocessor = linear_model.named_steps["preprocessor"]
    svm = linear_model.named_steps["model"]
    feature_names = get_feature_names(preprocessor, X_train)
    coefficients = svm.coef_.ravel()

    # Pair each transformed feature with its learned linear weight.
    coef_df = pd.DataFrame({"feature": feature_names, "weight": coefficients})
    positive = coef_df.sort_values("weight", ascending=False).head(5)
    negative = coef_df.sort_values("weight", ascending=True).head(5)

    print("\nLinear SVM coefficient snapshot:")
    print("  Strongest push toward churn=1:")
    print(positive.to_string(index=False))
    print("  Strongest push toward churn=0:")
    print(negative.to_string(index=False))


# Save a 2D decision-boundary plot using the best numeric feature pair and highlight support vectors.
def save_visualization(best_model, X_train, y_train, X_val, y_val):
    # Reuse the winning SVM settings for the simpler 2D visualization model.
    svm = best_model.named_steps["model"]

    # Use a fixed two-feature view so the plot stays simple and easy to explain.
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

    # Fit a second SVM on the selected feature pair so we can draw a true 2D boundary.
    viz_model = SVC(**viz_params)
    viz_model.fit(X_train_2d, y_train)

    # Build a dense grid so contour lines can show the 2D decision regions.
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

    # Draw the decision regions and boundary first so the points stay visible on top.
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
    # Plot the validation examples using their true labels.
    ax.scatter(
        X_val_2d[y_val == 0, 0],
        X_val_2d[y_val == 0, 1],
        c="tab:blue",
        marker="o",
        s=32,
        alpha=0.7,
        label="Retained (0)",
    )
    ax.scatter(
        X_val_2d[y_val == 1, 0],
        X_val_2d[y_val == 1, 1],
        c="tab:orange",
        marker="^",
        s=36,
        alpha=0.7,
        label="Churned (1)",
    )
    # Highlight support vectors with open circles so they stand out from the class markers.
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
    ax.set_title("SVM decision boundary using two selected features")
    ax.legend(loc="best", frameon=True)
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    # Score the simple 2D model so we can report how informative the view is.
    pair_score = f1_score(y_val, viz_model.predict(X_val_2d), zero_division=0)
    return pair_score, len(support_points_2d)


def main():
    # Load the churn dataset and create a reproducible 60/20/20 stratified split.
    df, X, y = load_data(DATA_PATH)

    # Keep the test split untouched until final model selection is complete.
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

    # Print the dataset and split information before training models.
    print("Assignment 8 Part A - SVMs and Interpretability")
    print("===============================================")
    print(f"Rows after dropping missing targets: {len(df)}")
    print_split_info(y_train, y_val, y_test)
    print(f"Primary validation metric: {PRIMARY_METRIC}")

    # Try several C values and class-weight settings for the linear SVM.
    linear_results = []
    best_linear_model = None
    best_linear_metrics = None
    best_linear_params = None

    for c_value in LINEAR_C_VALUES:
        for class_weight in CLASS_WEIGHT_VALUES:
            # Train one linear SVM configuration from the small search list.
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

            # Keep the best linear model according to validation balanced accuracy.
            if best_linear_metrics is None or metrics[PRIMARY_METRIC] > best_linear_metrics[PRIMARY_METRIC]:
                best_linear_model = model
                best_linear_metrics = metrics
                best_linear_params = {"C": c_value, "class_weight": class_weight}

    # Turn the collected linear-model results into a readable table.
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

    # Try a small RBF grid over C, gamma, and class_weight to compare a non-linear decision boundary.
    rbf_results = []
    best_rbf_model = None
    best_rbf_metrics = None
    best_rbf_params = None

    for c_value in RBF_C_VALUES:
        for gamma_value in RBF_GAMMA_VALUES:
            for class_weight in CLASS_WEIGHT_VALUES:
                # Train one RBF SVM configuration from the small search grid.
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

                # Keep the best RBF model according to validation balanced accuracy.
                if best_rbf_metrics is None or metrics[PRIMARY_METRIC] > best_rbf_metrics[PRIMARY_METRIC]:
                    best_rbf_model = model
                    best_rbf_metrics = metrics
                    best_rbf_params = {"C": c_value, "gamma": gamma_value, "class_weight": class_weight}

    # Turn the collected RBF-model results into a readable table.
    rbf_df = pd.DataFrame(rbf_results)
    print("\nRBF SVM validation results:")
    print(rbf_df.to_string(index=False))
    print(
        f"Best RBF setting by validation {PRIMARY_METRIC}: "
        f"C={best_rbf_params['C']}, gamma={best_rbf_params['gamma']}, "
        f"class_weight={best_rbf_params['class_weight']}"
    )

    # Choose the best kernel by validation's primary metric before touching the test set.
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

    # Save one visualization to support the written interpretation.
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
        # Rebuild the chosen RBF settings on all non-test data before the final test step.
        final_model = make_pipeline(
            build_preprocessor(X_train_val),
            kernel="rbf",
            c_value=best_rbf_params["C"],
            gamma=best_rbf_params["gamma"],
            class_weight=best_rbf_params["class_weight"],
        )

    # Fit once on train+validation and evaluate once on the untouched test split.
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
