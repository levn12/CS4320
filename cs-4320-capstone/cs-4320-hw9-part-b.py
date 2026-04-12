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
DATA_PATH = BASE_DIR / "electrical_fault_data.csv"

# Keep the split and model comparisons reproducible across runs.
RANDOM_STATE = 4320
# Use balanced accuracy as the main model-selection metric because the classes are somewhat imbalanced.
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
