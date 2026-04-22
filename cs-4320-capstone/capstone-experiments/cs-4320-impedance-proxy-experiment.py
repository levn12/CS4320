"""
CS 4320 - Impedance proxy feature experiment

Test whether physically motivated proxy features improve multiclass electrical
fault-pattern classification beyond the six raw measurements alone.

Feature sets compared:
1. raw_only: Ia, Ib, Ic, Va, Vb, Vc
2. proxy_only: impedance- and balance-inspired engineered features
3. raw_plus_proxy: raw measurements plus engineered proxy features

Models compared:
1. k-Nearest Neighbors
2. RBF-kernel SVM
3. Random Forest
"""

import json
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"
OUTPUT_DIR = BASE_DIR / "multiclass_fault_comparison_outputs"

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

KNN_K_VALUES = [1, 3, 5, 7, 11, 15]
KNN_WEIGHT_VALUES = ["uniform", "distance"]
KNN_P_VALUES = [1, 2]

SVM_C_VALUES = [1.0, 10.0, 100.0]
SVM_GAMMA_VALUES = [0.01, 0.1, "scale"]
SVM_CLASS_WEIGHT_VALUES = [None, "balanced"]

RF_TUNE_GRID = [
    {"n_estimators": 200, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": 20, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 1},
]


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data, path: Path):
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
    except PermissionError:
        print(f"Warning: could not save JSON output to {path}")


def save_dataframe(df: pd.DataFrame, path: Path):
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print(f"Warning: could not save tabular output to {path}")


def safe_ratio(numerator, denominator):
    return numerator / np.maximum(denominator, EPSILON)


def load_data(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path)
    y = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1).to_numpy()
    X_raw = df[RAW_FEATURE_COLS].copy()
    return df, X_raw, y


def build_proxy_features(X_raw: pd.DataFrame):
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
        proxy[f"{phase_name}_current_share"] = safe_ratio(abs_current, abs_current_sum)
        proxy[f"{phase_name}_voltage_share"] = safe_ratio(abs_voltage, abs_voltage_sum)

    current_matrix = X_raw[CURRENT_COLS].to_numpy()
    voltage_matrix = X_raw[VOLTAGE_COLS].to_numpy()
    current_norm = np.linalg.norm(current_matrix, axis=1)
    voltage_norm = np.linalg.norm(voltage_matrix, axis=1)

    proxy["current_vector_mag"] = current_norm
    proxy["voltage_vector_mag"] = voltage_norm
    proxy["z_proxy_3ph"] = safe_ratio(voltage_norm, current_norm)

    dot_product = np.sum(current_matrix * voltage_matrix, axis=1)
    proxy["v_i_alignment"] = safe_ratio(dot_product, current_norm * voltage_norm)
    proxy["instantaneous_power_proxy"] = dot_product
    proxy["abs_instantaneous_power_proxy"] = np.abs(dot_product)

    proxy["current_sum"] = X_raw[CURRENT_COLS].sum(axis=1)
    proxy["abs_current_sum"] = proxy["current_sum"].abs()

    proxy["current_abs_mean"] = abs_currents.mean(axis=1)
    proxy["voltage_abs_mean"] = abs_voltages.mean(axis=1)
    proxy["current_abs_std"] = abs_currents.std(axis=1, ddof=0)
    proxy["voltage_abs_std"] = abs_voltages.std(axis=1, ddof=0)
    proxy["current_unbalance"] = safe_ratio(proxy["current_abs_std"], proxy["current_abs_mean"])
    proxy["voltage_unbalance"] = safe_ratio(proxy["voltage_abs_std"], proxy["voltage_abs_mean"])

    proxy["current_abs_max"] = abs_currents.max(axis=1)
    proxy["current_abs_min"] = abs_currents.min(axis=1)
    proxy["voltage_abs_max"] = abs_voltages.max(axis=1)
    proxy["voltage_abs_min"] = abs_voltages.min(axis=1)
    proxy["current_range_ratio"] = safe_ratio(proxy["current_abs_max"], proxy["current_abs_min"] + EPSILON)
    proxy["voltage_range_ratio"] = safe_ratio(proxy["voltage_abs_max"], proxy["voltage_abs_min"] + EPSILON)

    phase_pairs = [("A", "B"), ("B", "C"), ("C", "A")]
    current_map = dict(zip(PHASE_NAMES, CURRENT_COLS))
    voltage_map = dict(zip(PHASE_NAMES, VOLTAGE_COLS))
    for left_phase, right_phase in phase_pairs:
        current_diff = X_raw[current_map[left_phase]] - X_raw[current_map[right_phase]]
        voltage_diff = X_raw[voltage_map[left_phase]] - X_raw[voltage_map[right_phase]]
        proxy[f"i_diff_{left_phase}{right_phase}"] = current_diff
        proxy[f"v_diff_{left_phase}{right_phase}"] = voltage_diff
        proxy[f"z_proxy_{left_phase}{right_phase}"] = safe_ratio(voltage_diff.abs(), current_diff.abs())

    return proxy


def build_feature_sets(X_raw: pd.DataFrame):
    proxy = build_proxy_features(X_raw)
    return {
        "raw_only": X_raw.copy(),
        "proxy_only": proxy,
        "raw_plus_proxy": pd.concat([X_raw, proxy], axis=1),
    }


def build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=numeric_steps),
                X.columns.tolist(),
            )
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


def tune_knn(X_train, y_train, X_val, y_val):
    best_model = None
    best_params = None
    best_metrics = None
    rows = []

    for n_neighbors, weights, p_value in product(KNN_K_VALUES, KNN_WEIGHT_VALUES, KNN_P_VALUES):
        model = make_pipeline(
            X_train,
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                p=p_value,
                metric="minkowski",
                n_jobs=1,
            ),
            scale_numeric=True,
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        rows.append(
            {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "p": p_value,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_params = {"n_neighbors": n_neighbors, "weights": weights, "p": p_value}
            best_metrics = metrics

    return pd.DataFrame(rows), best_model, best_params, best_metrics


def tune_rbf_svm(X_train, y_train, X_val, y_val):
    best_model = None
    best_params = None
    best_metrics = None
    rows = []

    for c_value, gamma_value, class_weight in product(
        SVM_C_VALUES,
        SVM_GAMMA_VALUES,
        SVM_CLASS_WEIGHT_VALUES,
    ):
        model = make_pipeline(
            X_train,
            SVC(
                kernel="rbf",
                C=c_value,
                gamma=gamma_value,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        rows.append(
            {
                "C": c_value,
                "gamma": gamma_value,
                "class_weight": str(class_weight),
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_params = {"C": c_value, "gamma": gamma_value, "class_weight": class_weight}
            best_metrics = metrics

    return pd.DataFrame(rows), best_model, best_params, best_metrics


def tune_random_forest(X_train, y_train, X_val, y_val):
    best_model = None
    best_params = None
    best_metrics = None
    rows = []

    for config in RF_TUNE_GRID:
        model = make_pipeline(
            X_train,
            RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                max_features=config["max_features"],
                min_samples_leaf=config["min_samples_leaf"],
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            scale_numeric=False,
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        rows.append(
            {
                **config,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_params = dict(config)
            best_metrics = metrics

    return pd.DataFrame(rows), best_model, best_params, best_metrics


def refit_and_evaluate(model_name: str, best_params: dict, X_train_val, y_train_val, X_test, y_test):
    if model_name == "kNN":
        model = make_pipeline(
            X_train_val,
            KNeighborsClassifier(
                n_neighbors=best_params["n_neighbors"],
                weights=best_params["weights"],
                p=best_params["p"],
                metric="minkowski",
                n_jobs=1,
            ),
            scale_numeric=True,
        )
    elif model_name == "RBF SVM":
        model = make_pipeline(
            X_train_val,
            SVC(
                kernel="rbf",
                C=best_params["C"],
                gamma=best_params["gamma"],
                class_weight=best_params["class_weight"],
                random_state=RANDOM_STATE,
            ),
            scale_numeric=True,
        )
    elif model_name == "Random Forest":
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(X_train_val, y_train_val)
    predictions = model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)
    return model, metrics, predictions


def save_confusion_plot(confusion_df: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(confusion_df.to_numpy(), cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(confusion_df.columns)))
    ax.set_xticklabels(confusion_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(confusion_df.index)))
    ax.set_yticklabels(confusion_df.index)

    max_value = int(confusion_df.to_numpy().max()) if len(confusion_df) else 0
    for row_index in range(confusion_df.shape[0]):
        for col_index in range(confusion_df.shape[1]):
            value = int(confusion_df.iat[row_index, col_index])
            text_color = "white" if value > max_value / 2 else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    try:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    except PermissionError:
        print(f"Warning: could not save plot output to {path}")
    plt.close(fig)


def collect_random_forest_importances(model):
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    cleaned_names = [name.split("__", maxsplit=1)[-1] for name in feature_names]
    importance_df = pd.DataFrame(
        {
            "feature": cleaned_names,
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importance_df


def main():
    ensure_output_dir()

    df, X_raw, y = load_data(DATA_PATH)
    class_names = sorted(pd.Series(y).unique().tolist())
    feature_sets = build_feature_sets(X_raw)

    train_val_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    y_train_val = y[train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN_VAL,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    results_rows = []
    search_tables = {}
    confusion_tables = {}
    best_models_for_importance = {}

    model_tuners = {
        "kNN": tune_knn,
        "RBF SVM": tune_rbf_svm,
        "Random Forest": tune_random_forest,
    }

    print("Impedance proxy feature experiment")
    print("=================================")
    print(f"Rows: {len(df)}")
    print(f"Feature sets: {', '.join(feature_sets.keys())}")
    print(f"Primary validation metric: {PRIMARY_METRIC}")

    for feature_set_name, X_features in feature_sets.items():
        X_train = X_features.iloc[train_indices]
        X_val = X_features.iloc[val_indices]
        X_test = X_features.iloc[test_indices]
        X_train_val = X_features.iloc[train_val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        print(f"\nFeature set: {feature_set_name} ({X_features.shape[1]} columns)")

        for model_name, tuner in model_tuners.items():
            search_df, _, best_params, best_val_metrics = tuner(X_train, y_train, X_val, y_val)
            search_tables[(feature_set_name, model_name)] = search_df

            final_model, test_metrics, predictions = refit_and_evaluate(
                model_name,
                best_params,
                X_train_val,
                y_train_val,
                X_test,
                y_test,
            )
            confusion_df = compute_confusion_df(y_test, predictions, class_names)
            confusion_tables[(feature_set_name, model_name)] = confusion_df

            if model_name == "Random Forest":
                best_models_for_importance[feature_set_name] = (final_model, X_features.columns.tolist())

            row = {
                "feature_set": feature_set_name,
                "model": model_name,
                "feature_count": int(X_features.shape[1]),
                "best_params": json.dumps(best_params, default=str),
                "val_accuracy": best_val_metrics["accuracy"],
                "val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
                "val_precision": best_val_metrics["precision"],
                "val_recall": best_val_metrics["recall"],
                "val_f1": best_val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
            results_rows.append(row)

            print(
                f"  {model_name}: "
                f"val {format_metrics(best_val_metrics)} | "
                f"test {format_metrics(test_metrics)}"
            )

    results_df = pd.DataFrame(results_rows).sort_values(
        ["test_balanced_accuracy", "test_f1", "val_balanced_accuracy"],
        ascending=False,
    )
    save_dataframe(results_df, OUTPUT_DIR / "feature_set_model_comparison.csv")

    for (feature_set_name, model_name), search_df in search_tables.items():
        safe_feature_name = feature_set_name.replace("+", "_plus_")
        safe_model_name = model_name.lower().replace(" ", "_")
        save_dataframe(search_df, OUTPUT_DIR / f"{safe_feature_name}_{safe_model_name}_search.csv")

    for (feature_set_name, model_name), confusion_df in confusion_tables.items():
        safe_feature_name = feature_set_name.replace("+", "_plus_")
        safe_model_name = model_name.lower().replace(" ", "_")
        save_dataframe(confusion_df.reset_index(), OUTPUT_DIR / f"{safe_feature_name}_{safe_model_name}_confusion.csv")
        save_confusion_plot(
            confusion_df,
            OUTPUT_DIR / f"{safe_feature_name}_{safe_model_name}_confusion.png",
            title=f"{feature_set_name} / {model_name} test confusion matrix",
        )

    for feature_set_name, (model, _) in best_models_for_importance.items():
        importance_df = collect_random_forest_importances(model)
        safe_feature_name = feature_set_name.replace("+", "_plus_")
        save_dataframe(importance_df, OUTPUT_DIR / f"{safe_feature_name}_random_forest_feature_importance.csv")

    best_row = results_df.iloc[0]
    summary = {
        "dataset_rows": int(len(df)),
        "primary_metric": PRIMARY_METRIC,
        "best_configuration": {
            "feature_set": str(best_row["feature_set"]),
            "model": str(best_row["model"]),
            "feature_count": int(best_row["feature_count"]),
            "best_params": best_row["best_params"],
            "test_balanced_accuracy": float(best_row["test_balanced_accuracy"]),
            "test_f1": float(best_row["test_f1"]),
        },
        "feature_sets": {
            feature_set_name: {"feature_count": int(feature_frame.shape[1])}
            for feature_set_name, feature_frame in feature_sets.items()
        },
    }
    save_json(summary, OUTPUT_DIR / "experiment_summary.json")

    print("\nTop configurations by test balanced accuracy:")
    print(
        results_df[
            [
                "feature_set",
                "model",
                "feature_count",
                "val_balanced_accuracy",
                "test_balanced_accuracy",
                "test_f1",
            ]
        ].to_string(index=False)
    )

    best_rf_row = results_df[results_df["model"] == "Random Forest"].iloc[0]
    best_rf_feature_set = str(best_rf_row["feature_set"])
    best_rf_importance_df = collect_random_forest_importances(best_models_for_importance[best_rf_feature_set][0])
    print(f"\nTop Random Forest proxy features for {best_rf_feature_set}:")
    print(best_rf_importance_df.head(12).to_string(index=False))

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
