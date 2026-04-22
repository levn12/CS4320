"""
CS 4320 - Random Forest vs MLP comparison on raw and proxy electrical features.

This script:
1. Loads the electrical fault dataset.
2. Builds a multiclass fault-pattern target from G/C/B/A.
3. Compares Random Forest and MLP on:
   - raw_only
   - raw_plus_best_proxy
4. Tunes both model families on a shared train/validation split.
5. Refits the best configuration on train+validation and evaluates once on test.
6. Saves summaries and plots inside this experiment folder.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_PATH = EXPERIMENT_DIR.parent.parent / "electrical_fault_data.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

FAULT_COLS = ["G", "C", "B", "A"]
CURRENT_COLS = ["Ia", "Ib", "Ic"]
VOLTAGE_COLS = ["Va", "Vb", "Vc"]
RAW_FEATURE_COLS = CURRENT_COLS + VOLTAGE_COLS
PHASE_NAMES = ["A", "B", "C"]

RANDOM_STATE = 4320
PRIMARY_METRIC = "balanced_accuracy"
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN_VAL = 0.25
EPSILON = 1e-6

RF_TUNE_GRID = [
    {"n_estimators": 300, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": 20, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 500, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 1},
]

MLP_TUNE_GRID = [
    {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (256, 128), "alpha": 1e-4, "learning_rate_init": 1e-3},
    {"hidden_layer_sizes": (128, 64), "alpha": 5e-4, "learning_rate_init": 5e-4},
    {"hidden_layer_sizes": (256, 128), "alpha": 5e-4, "learning_rate_init": 5e-4},
]


def safe_ratio(numerator, denominator):
    return numerator / np.maximum(denominator, EPSILON)


def ensure_output_dir(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except PermissionError:
        print(f"Warning: could not create {path}. Falling back to {EXPERIMENT_DIR}")
        return EXPERIMENT_DIR


def save_dataframe(df: pd.DataFrame, path: Path):
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print(f"Warning: could not save CSV output to {path}")


def save_json(data, path: Path):
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
    except PermissionError:
        print(f"Warning: could not save JSON output to {path}")


def save_text(text: str, path: Path):
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        print(f"Warning: could not save text output to {path}")


def save_figure(fig, path: Path):
    try:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    except PermissionError:
        print(f"Warning: could not save plot output to {path}")
    plt.close(fig)


def load_data():
    df = pd.read_csv(DATA_PATH)
    y = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1).to_numpy()
    X_raw = df[RAW_FEATURE_COLS].copy()
    return df, X_raw, y


def build_best_proxy_features(X_raw: pd.DataFrame):
    proxy = pd.DataFrame(index=X_raw.index)

    abs_currents = X_raw[CURRENT_COLS].abs()
    abs_voltages = X_raw[VOLTAGE_COLS].abs()
    abs_current_sum = abs_currents.sum(axis=1)
    abs_voltage_sum = abs_voltages.sum(axis=1)

    for current_col, voltage_col, phase_name in zip(CURRENT_COLS, VOLTAGE_COLS, PHASE_NAMES):
        abs_current = X_raw[current_col].abs()
        abs_voltage = X_raw[voltage_col].abs()
        proxy[f"abs_{current_col}"] = abs_current
        proxy[f"abs_{voltage_col}"] = abs_voltage
        proxy[f"z_proxy_{phase_name}"] = safe_ratio(abs_voltage, abs_current)

    current_matrix = X_raw[CURRENT_COLS].to_numpy()
    voltage_matrix = X_raw[VOLTAGE_COLS].to_numpy()

    proxy["abs_current_sum"] = abs_current_sum
    proxy["abs_voltage_sum"] = abs_voltage_sum
    proxy["current_sum"] = X_raw[CURRENT_COLS].sum(axis=1)
    proxy["abs_current_sum_signed"] = proxy["current_sum"].abs()

    proxy["current_vector_mag"] = np.linalg.norm(current_matrix, axis=1)
    proxy["voltage_vector_mag"] = np.linalg.norm(voltage_matrix, axis=1)
    proxy["z_proxy_3ph"] = safe_ratio(proxy["voltage_vector_mag"], proxy["current_vector_mag"])

    proxy["current_abs_mean"] = abs_currents.mean(axis=1)
    proxy["current_abs_std"] = abs_currents.std(axis=1, ddof=0)
    proxy["current_abs_max"] = abs_currents.max(axis=1)
    proxy["voltage_abs_mean"] = abs_voltages.mean(axis=1)
    proxy["voltage_abs_std"] = abs_voltages.std(axis=1, ddof=0)
    proxy["current_unbalance"] = safe_ratio(proxy["current_abs_std"], proxy["current_abs_mean"])
    proxy["voltage_unbalance"] = safe_ratio(proxy["voltage_abs_std"], proxy["voltage_abs_mean"])

    dot_product = np.sum(current_matrix * voltage_matrix, axis=1)
    proxy["instantaneous_power_proxy"] = dot_product
    proxy["abs_instantaneous_power_proxy"] = np.abs(dot_product)
    proxy["v_i_alignment"] = safe_ratio(
        dot_product,
        proxy["current_vector_mag"] * proxy["voltage_vector_mag"],
    )

    phase_pairs = [("A", "B"), ("B", "C"), ("C", "A")]
    current_map = dict(zip(PHASE_NAMES, CURRENT_COLS))
    voltage_map = dict(zip(PHASE_NAMES, VOLTAGE_COLS))
    for left_phase, right_phase in phase_pairs:
        current_diff = X_raw[current_map[left_phase]] - X_raw[current_map[right_phase]]
        voltage_diff = X_raw[voltage_map[left_phase]] - X_raw[voltage_map[right_phase]]
        proxy[f"z_proxy_{left_phase}{right_phase}"] = safe_ratio(voltage_diff.abs(), current_diff.abs())

    return proxy


def build_feature_sets(X_raw: pd.DataFrame):
    proxy = build_best_proxy_features(X_raw)
    return {
        "raw_only": X_raw.copy(),
        "raw_plus_best_proxy": pd.concat([X_raw, proxy], axis=1),
    }, proxy


def build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), X.columns.tolist()),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def make_pipeline(X: pd.DataFrame, model, *, scale_numeric: bool):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def evaluate(model, X, y):
    pred = model.predict(X)
    return evaluate_predictions(y, pred)


def compute_confusion_df(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in class_names],
        columns=[f"pred_{label}" for label in class_names],
    )


def format_metrics(metrics):
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )


def tune_random_forest(X_train, y_train, X_val, y_val):
    rows = []
    best_model = None
    best_params = None
    best_metrics = None

    for params in RF_TUNE_GRID:
        model = make_pipeline(
            X_train,
            RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            scale_numeric=False,
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        rows.append(
            {
                **params,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_params = params.copy()
            best_metrics = metrics

    return pd.DataFrame(rows), best_model, best_params, best_metrics


def tune_mlp(X_train, y_train, X_val, y_val):
    rows = []
    best_model = None
    best_params = None
    best_metrics = None

    for params in MLP_TUNE_GRID:
        model = make_pipeline(
            X_train,
            MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                learning_rate_init=params["learning_rate_init"],
                batch_size=128,
                max_iter=400,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        mlp_model = model.named_steps["model"]
        rows.append(
            {
                **params,
                "epochs_ran": int(getattr(mlp_model, "n_iter_", 0)),
                "loss_final": float(getattr(mlp_model, "loss_", np.nan)),
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_params = params.copy()
            best_metrics = metrics

    return pd.DataFrame(rows), best_model, best_params, best_metrics


def refit_best_model(model_name: str, best_params: dict, X_train_val, y_train_val):
    if model_name == "Random Forest":
        model = make_pipeline(
            X_train_val,
            RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                max_features=best_params["max_features"],
                min_samples_leaf=best_params["min_samples_leaf"],
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            scale_numeric=False,
        )
    elif model_name == "MLP":
        model = make_pipeline(
            X_train_val,
            MLPClassifier(
                hidden_layer_sizes=best_params["hidden_layer_sizes"],
                alpha=best_params["alpha"],
                learning_rate_init=best_params["learning_rate_init"],
                batch_size=128,
                max_iter=400,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train_val, y_train_val)
    return model


def plot_test_metric_comparison(results_df: pd.DataFrame, path: Path):
    metric_columns = ["test_accuracy", "test_balanced_accuracy", "test_f1"]
    pretty_labels = ["Accuracy", "Balanced acc", "Macro F1"]
    plot_df = results_df.copy()
    plot_df["label"] = plot_df["model"] + "\n" + plot_df["feature_set"]

    x_positions = np.arange(len(plot_df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for metric_index, metric_name in enumerate(metric_columns):
        ax.bar(
            x_positions + (metric_index - 1) * width,
            plot_df[metric_name],
            width=width,
            label=pretty_labels[metric_index],
        )

    ax.set_title("Random Forest vs MLP test metrics")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["label"])
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, path)


def plot_validation_vs_test(results_df: pd.DataFrame, path: Path):
    plot_df = results_df.copy()
    plot_df["label"] = plot_df["model"] + "\n" + plot_df["feature_set"]
    x_positions = np.arange(len(plot_df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar(x_positions - width / 2, plot_df["val_balanced_accuracy"], width=width, label="Validation")
    ax.bar(x_positions + width / 2, plot_df["test_balanced_accuracy"], width=width, label="Test")
    ax.set_title("Validation vs test balanced accuracy")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["label"])
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, path)


def plot_proxy_pca(proxy_df: pd.DataFrame, labels, path: Path):
    scaler = StandardScaler()
    proxy_scaled = scaler.fit_transform(proxy_df)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(proxy_scaled)

    plot_df = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
    plot_df["fault_pattern"] = labels
    patterns = sorted(plot_df["fault_pattern"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    colors = {pattern: cmap(index % 10) for index, pattern in enumerate(patterns)}

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for pattern in patterns:
        subset = plot_df[plot_df["fault_pattern"] == pattern]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=16,
            alpha=0.65,
            color=colors[pattern],
            label=pattern,
        )

    ax.set_title("PCA view of best proxy features")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Fault pattern", ncol=2, fontsize=9)
    save_figure(fig, path)


def plot_random_forest_importance(model, path: Path, top_n: int = 15):
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = [name.split("__", maxsplit=1)[-1] for name in preprocessor.get_feature_names_out()]
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(top_n)
    importance_df = importance_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title("Best Random Forest feature importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.25)
    save_figure(fig, path)

    return importance_df.iloc[::-1].reset_index(drop=True)


def plot_mlp_loss_curve(model, path: Path):
    mlp_model = model.named_steps["model"]
    loss_curve = getattr(mlp_model, "loss_curve_", None)
    if not loss_curve:
        return

    epochs = np.arange(1, len(loss_curve) + 1)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(epochs, loss_curve, marker="o")
    ax.set_title("Best MLP loss curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    save_figure(fig, path)


def build_markdown_summary(results_df: pd.DataFrame, summary: dict, rf_importance_df: pd.DataFrame):
    lines = [
        "# Random Forest vs MLP Proxy Comparison",
        "",
        f"- Rows: `{summary['dataset_rows']}`",
        f"- Primary validation metric: `{summary['primary_metric']}`",
        f"- Feature sets compared: `{', '.join(summary['feature_sets'])}`",
        "",
        "## Best Configuration",
        "",
        f"- Model: `{summary['best_configuration']['model']}`",
        f"- Feature set: `{summary['best_configuration']['feature_set']}`",
        f"- Test balanced accuracy: `{summary['best_configuration']['test_balanced_accuracy']:.4f}`",
        f"- Test F1: `{summary['best_configuration']['test_f1']:.4f}`",
        "",
        "## Results",
        "",
        "```text",
        results_df[
            [
                "model",
                "feature_set",
                "val_balanced_accuracy",
                "test_balanced_accuracy",
                "test_f1",
                "best_params",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Top Random Forest Features",
        "",
        "```text",
        rf_importance_df.to_string(index=False),
        "```",
        "",
        "## Proxy Features Used",
        "",
    ]

    for feature_name in summary["proxy_feature_names"]:
        lines.append(f"- `{feature_name}`")

    return "\n".join(lines)


def main():
    output_dir = ensure_output_dir(DEFAULT_OUTPUT_DIR)

    df, X_raw, y = load_data()
    feature_sets, proxy_df = build_feature_sets(X_raw)
    class_names = sorted(pd.Series(y).unique().tolist())

    X_train_val_idx, X_test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    y_train_val = y[X_train_val_idx]
    X_train_idx, X_val_idx = train_test_split(
        X_train_val_idx,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN_VAL,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    tuners = {
        "Random Forest": tune_random_forest,
        "MLP": tune_mlp,
    }

    results_rows = []
    tuned_models = {}

    print("Random Forest vs MLP proxy comparison")
    print("=====================================")
    print(f"Rows: {len(df)}")
    print(f"Feature sets: {', '.join(feature_sets.keys())}")
    print(f"Primary validation metric: {PRIMARY_METRIC}")

    for feature_set_name, X_features in feature_sets.items():
        X_train = X_features.iloc[X_train_idx]
        X_val = X_features.iloc[X_val_idx]
        X_test = X_features.iloc[X_test_idx]
        X_train_val = X_features.iloc[X_train_val_idx]
        y_train = y[X_train_idx]
        y_val = y[X_val_idx]
        y_test = y[X_test_idx]

        print(f"\nFeature set: {feature_set_name} ({X_features.shape[1]} columns)")

        for model_name, tuner in tuners.items():
            search_df, _, best_params, val_metrics = tuner(X_train, y_train, X_val, y_val)
            best_model = refit_best_model(model_name, best_params, X_train_val, y_train_val)
            test_predictions = best_model.predict(X_test)
            test_metrics = evaluate_predictions(y_test, test_predictions)
            confusion_df = compute_confusion_df(y_test, test_predictions, class_names)

            tuned_models[(model_name, feature_set_name)] = {
                "best_model": best_model,
                "search_df": search_df,
                "confusion_df": confusion_df,
            }

            row = {
                "model": model_name,
                "feature_set": feature_set_name,
                "feature_count": int(X_features.shape[1]),
                "best_params": json.dumps(best_params, default=str),
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
            results_rows.append(row)

            print(f"  {model_name}: val {format_metrics(val_metrics)} | test {format_metrics(test_metrics)}")

    results_df = pd.DataFrame(results_rows).sort_values(
        ["test_balanced_accuracy", "test_f1", "val_balanced_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_rf_key = None
    for _, row in results_df.iterrows():
        if row["model"] == "Random Forest":
            best_rf_key = (row["model"], row["feature_set"])
            break

    best_mlp_key = None
    for _, row in results_df.iterrows():
        if row["model"] == "MLP":
            best_mlp_key = (row["model"], row["feature_set"])
            break

    rf_importance_df = plot_random_forest_importance(
        tuned_models[best_rf_key]["best_model"],
        output_dir / "best_random_forest_feature_importance.png",
    )
    plot_mlp_loss_curve(
        tuned_models[best_mlp_key]["best_model"],
        output_dir / "best_mlp_loss_curve.png",
    )
    plot_test_metric_comparison(results_df, output_dir / "test_metric_comparison.png")
    plot_validation_vs_test(results_df, output_dir / "validation_vs_test_balanced_accuracy.png")
    plot_proxy_pca(proxy_df, y, output_dir / "proxy_feature_pca.png")

    save_dataframe(results_df, output_dir / "comparison_results.csv")
    for (model_name, feature_set_name), bundle in tuned_models.items():
        safe_name = f"{model_name.lower().replace(' ', '_')}_{feature_set_name}"
        save_dataframe(bundle["search_df"], output_dir / f"{safe_name}_search_results.csv")
        save_dataframe(bundle["confusion_df"].reset_index(), output_dir / f"{safe_name}_confusion_matrix.csv")

    summary = {
        "dataset_rows": int(len(df)),
        "primary_metric": PRIMARY_METRIC,
        "feature_sets": list(feature_sets.keys()),
        "proxy_feature_names": proxy_df.columns.tolist(),
        "class_names": class_names,
        "best_configuration": {
            "model": str(best_row["model"]),
            "feature_set": str(best_row["feature_set"]),
            "feature_count": int(best_row["feature_count"]),
            "best_params": best_row["best_params"],
            "test_balanced_accuracy": float(best_row["test_balanced_accuracy"]),
            "test_f1": float(best_row["test_f1"]),
        },
        "results": results_df.to_dict(orient="records"),
    }
    save_json(summary, output_dir / "comparison_summary.json")
    save_text(build_markdown_summary(results_df, summary, rf_importance_df), output_dir / "comparison_summary.md")

    print("\nTop configurations:")
    print(
        results_df[
            [
                "model",
                "feature_set",
                "val_balanced_accuracy",
                "test_balanced_accuracy",
                "test_f1",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
