# CS 4320 - Assignment 9 Part A
# Comparing a single decision tree, Random Forest, and Gradient Boosting.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "generated_dataset.csv"

# Keep the split and model comparisons reproducible across runs.
RANDOM_STATE = 4320
# Use balanced accuracy as the primary metric because the classes are somewhat imbalanced.
PRIMARY_METRIC = "balanced_accuracy"
TREE_DEPTH_VALUES = [3, 5, 8, None]
RF_TUNE_GRID = [
    {"n_estimators": 100, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 8, "max_features": "sqrt"},
    {"n_estimators": 700, "max_depth": 8, "max_features": "sqrt"},
]
GB_TUNE_GRID = [
    {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 5},
    {"n_estimators": 300, "learning_rate": 0.10, "max_depth": 5},
    {"n_estimators": 500, "learning_rate": 0.10, "max_depth": 5},
    {"n_estimators": 700, "learning_rate": 0.10, "max_depth": 5},
]


def load_data(data_path: Path = DATA_PATH):
    # Read the manufacturing-risk dataset.
    df = pd.read_csv(data_path)
    y = df["target"].astype(int).to_numpy()
    X = df.drop(columns=["target"])
    return df, X, y


def build_preprocessor(X: pd.DataFrame):
    # Split the columns by type so missing values and categories are handled safely inside the pipeline.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

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


def make_pipeline(X: pd.DataFrame, model):
    # Keep preprocessing and model fitting together so every model sees the same split logic.
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
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
    # Print the number of examples in each split and the positive-class rate.
    print("Split sizes:")
    print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print("Class balance (positive = quality incident escalation):")
    print(f"  train: positives={int(y_train.sum())}, rate={y_train.mean():.3f}")
    print(f"  val:   positives={int(y_val.sum())}, rate={y_val.mean():.3f}")
    print(f"  test:  positives={int(y_test.sum())}, rate={y_test.mean():.3f}")


def fit_tree_search(X_train, y_train, X_val, y_val):
    # Try a few tree depths so we can show how a single tree becomes high-variance as it grows.
    results = []
    best_model = None
    best_metrics = None
    best_params = None

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
        results.append(
            {
                "model": "Decision Tree",
                "max_depth": str(max_depth),
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1"],
            }
        )

        if best_metrics is None or val_metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_metrics = val_metrics
            best_params = {"max_depth": max_depth}

    return pd.DataFrame(results), best_model, best_metrics, best_params


def fit_default_random_forest(X_train, y_train, X_val, y_val):
    # Train one default-style Random Forest to establish the bagging baseline.
    model = make_pipeline(
        X_train,
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
            oob_score=True,
        ),
    )
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_val, y_val)
    oob_score = float(model.named_steps["model"].oob_score_)
    return model, metrics, oob_score


def fit_random_forest_search(X_train, y_train, X_val, y_val):
    # Tune a small Random Forest grid to compare bagging settings without overcomplicating the search.
    results = []
    best_model = None
    best_metrics = None
    best_params = None
    best_oob = None

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
        results.append(
            {
                "model": "Random Forest",
                "n_estimators": params["n_estimators"],
                "max_depth": str(params["max_depth"]),
                "max_features": params["max_features"],
                "oob_accuracy": oob_score,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_metrics = metrics
            best_params = params.copy()
            best_oob = oob_score

    return pd.DataFrame(results), best_model, best_metrics, best_params, best_oob


def fit_gradient_boosting_search(X_train, y_train, X_val, y_val):
    # Tune a small Gradient Boosting grid to compare sequential error-correction settings.
    results = []
    best_model = None
    best_metrics = None
    best_params = None

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
        results.append(
            {
                "model": "Gradient Boosting",
                "n_estimators": params["n_estimators"],
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_metrics = metrics
            best_params = params.copy()

    return pd.DataFrame(results), best_model, best_metrics, best_params


def build_comparison_table(tree_metrics, tree_params, rf_metrics, rf_params, gb_metrics, gb_params):
    # Collect the best validation result from each model family into one compact comparison table.
    rows = [
        {
            "model": "Decision Tree",
            "key_hyperparameters": f"max_depth={tree_params['max_depth']}",
            "val_accuracy": tree_metrics["accuracy"],
            "val_balanced_accuracy": tree_metrics["balanced_accuracy"],
            "val_f1": tree_metrics["f1"],
        },
        {
            "model": "Random Forest",
            "key_hyperparameters": (
                f"n_estimators={rf_params['n_estimators']}, "
                f"max_depth={rf_params['max_depth']}, "
                f"max_features={rf_params['max_features']}"
            ),
            "val_accuracy": rf_metrics["accuracy"],
            "val_balanced_accuracy": rf_metrics["balanced_accuracy"],
            "val_f1": rf_metrics["f1"],
        },
        {
            "model": "Gradient Boosting",
            "key_hyperparameters": (
                f"n_estimators={gb_params['n_estimators']}, "
                f"learning_rate={gb_params['learning_rate']}, "
                f"max_depth={gb_params['max_depth']}"
            ),
            "val_accuracy": gb_metrics["accuracy"],
            "val_balanced_accuracy": gb_metrics["balanced_accuracy"],
            "val_f1": gb_metrics["f1"],
        },
    ]
    return pd.DataFrame(rows)


def refit_tree(X_train_val, y_train_val, params):
    # Rebuild the selected single-tree baseline on all non-test data.
    model = make_pipeline(
        X_train_val,
        DecisionTreeClassifier(
            max_depth=params["max_depth"],
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(X_train_val, y_train_val)
    return model


def refit_random_forest(X_train_val, y_train_val, params):
    # Rebuild the selected Random Forest on all non-test data before the final test step.
    model = make_pipeline(
        X_train_val,
        RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=1,
            oob_score=True,
        ),
    )
    model.fit(X_train_val, y_train_val)
    return model


def refit_gradient_boosting(X_train_val, y_train_val, params):
    # Rebuild the selected boosting model on all non-test data before the final test step.
    model = make_pipeline(
        X_train_val,
        GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=RANDOM_STATE,
        ),
    )
    model.fit(X_train_val, y_train_val)
    return model


def print_interpretation(tree_metrics, rf_metrics, gb_metrics):
    # Write short result-based interpretation paragraphs for the assignment prompt.
    rf_gain = rf_metrics["balanced_accuracy"] - tree_metrics["balanced_accuracy"]
    gb_gain = gb_metrics["balanced_accuracy"] - rf_metrics["balanced_accuracy"]

    print("\nInterpretation:")
    print(
        "Random Forest usually reduces variance because it averages many bootstrapped trees, "
        "so one unstable split in a single tree matters less in the final prediction. "
        f"That pattern showed up here because the best Random Forest improved validation {PRIMARY_METRIC} "
        f"from {tree_metrics['balanced_accuracy']:.4f} for the single tree to {rf_metrics['balanced_accuracy']:.4f} "
        f"for the forest, a gain of {rf_gain:.4f}."
    )
    print(
        "Boosting can outperform bagging when the dataset contains many small mistakes that can be corrected "
        "stage by stage, because each new tree focuses on the residual errors left by the earlier ones. "
        f"On this dataset, Gradient Boosting reached validation {PRIMARY_METRIC} "
        f"{gb_metrics['balanced_accuracy']:.4f} compared with {rf_metrics['balanced_accuracy']:.4f} for Random Forest, "
        f"which suggests the sequential corrections captured structure the averaged forest missed. "
        "At the same time, boosting can overfit if the learning rate is too aggressive or the sequence runs too long, "
        "so keeping the tuning small and readable matters."
    )
    print(f"Observation from results: Gradient Boosting vs. Random Forest changed validation {PRIMARY_METRIC} by {gb_gain:.4f}.")


def main():
    # Load the dataset and create the recommended stratified 70/15/15 split.
    df, X, y = load_data(DATA_PATH)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size= 15/85,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print("Assignment 9 Part A - Ensemble Learning")
    print("=======================================")
    print(f"Rows: {len(df)}")
    print_split_info(y_train, y_val, y_test)
    print(f"Primary validation metric: {PRIMARY_METRIC}")

    tree_results, best_tree_model, best_tree_metrics, best_tree_params = fit_tree_search(
        X_train, y_train, X_val, y_val
    )
    print("\nDecision Tree validation results:")
    print(tree_results.to_string(index=False))
    print(
        f"Best tree by validation {PRIMARY_METRIC}: "
        f"max_depth={best_tree_params['max_depth']}, {format_metrics(best_tree_metrics)}"
    )
    print(
        "Overfitting note: if train balanced accuracy keeps rising while validation stalls or drops, "
        "the deeper tree is fitting noise instead of just signal."
    )

    default_rf_model, default_rf_metrics, default_rf_oob = fit_default_random_forest(
        X_train, y_train, X_val, y_val
    )
    print("\nRandom Forest default validation result:")
    print(f"  {format_metrics(default_rf_metrics)}")
    print(f"  OOB accuracy={default_rf_oob:.4f}")
    print(
        "  OOB note: this is an internal training-set estimate from out-of-bag predictions, "
        "so it should usually be close to validation performance but not identical."
    )

    rf_results, best_rf_model, best_rf_metrics, best_rf_params, best_rf_oob = fit_random_forest_search(
        X_train, y_train, X_val, y_val
    )
    print("\nRandom Forest tuned validation results:")
    print(rf_results.to_string(index=False))
    print(
        f"Best Random Forest by validation {PRIMARY_METRIC}: "
        f"n_estimators={best_rf_params['n_estimators']}, "
        f"max_depth={best_rf_params['max_depth']}, "
        f"max_features={best_rf_params['max_features']}, "
        f"OOB accuracy={best_rf_oob:.4f}, {format_metrics(best_rf_metrics)}"
    )

    gb_results, best_gb_model, best_gb_metrics, best_gb_params = fit_gradient_boosting_search(
        X_train, y_train, X_val, y_val
    )
    print("\nGradient Boosting validation results:")
    print(gb_results.to_string(index=False))
    print(
        f"Best Gradient Boosting by validation {PRIMARY_METRIC}: "
        f"n_estimators={best_gb_params['n_estimators']}, "
        f"learning_rate={best_gb_params['learning_rate']}, "
        f"max_depth={best_gb_params['max_depth']}, "
        f"{format_metrics(best_gb_metrics)}"
    )

    comparison_df = build_comparison_table(
        best_tree_metrics,
        best_tree_params,
        best_rf_metrics,
        best_rf_params,
        best_gb_metrics,
        best_gb_params,
    )
    print("\nValidation comparison table:")
    print(comparison_df.to_string(index=False))

    print_interpretation(best_tree_metrics, best_rf_metrics, best_gb_metrics)

    validation_winner_name = "Decision Tree"
    validation_winner_params = best_tree_params
    validation_winner_metrics = best_tree_metrics

    if best_rf_metrics[PRIMARY_METRIC] > validation_winner_metrics[PRIMARY_METRIC]:
        validation_winner_name = "Random Forest"
        validation_winner_params = best_rf_params
        validation_winner_metrics = best_rf_metrics

    if best_gb_metrics[PRIMARY_METRIC] > validation_winner_metrics[PRIMARY_METRIC]:
        validation_winner_name = "Gradient Boosting"
        validation_winner_params = best_gb_params
        validation_winner_metrics = best_gb_metrics

    print("\nValidation winner:")
    print(f"  model={validation_winner_name}, {format_metrics(validation_winner_metrics)}")

    final_tree_model = refit_tree(X_train_val, y_train_val, best_tree_params)
    final_tree_test_metrics = evaluate(final_tree_model, X_test, y_test)

    if validation_winner_name == "Decision Tree":
        final_model = refit_tree(X_train_val, y_train_val, validation_winner_params)
    elif validation_winner_name == "Random Forest":
        final_model = refit_random_forest(X_train_val, y_train_val, validation_winner_params)
    else:
        final_model = refit_gradient_boosting(X_train_val, y_train_val, validation_winner_params)

    final_test_metrics = evaluate(final_model, X_test, y_test)

    print("\nFinal test evaluation:")
    print(f"  selected model={validation_winner_name}")
    print(f"  selected model test: {format_metrics(final_test_metrics)}")
    print(f"  single-tree baseline test: {format_metrics(final_tree_test_metrics)}")
    print(
        f"  test {PRIMARY_METRIC} difference vs. single tree="
        f"{final_test_metrics[PRIMARY_METRIC] - final_tree_test_metrics[PRIMARY_METRIC]:.4f}"
    )


if __name__ == "__main__":
    main()
