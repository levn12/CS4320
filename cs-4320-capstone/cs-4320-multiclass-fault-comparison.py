"""
CS 4320 - Capstone model comparison

Compare four model families on multiclass electrical fault-pattern detection:
1. k-Nearest Neighbors
2. RBF-kernel SVM
3. Random Forest
4. Multilayer Perceptron

Workflow:
1. Load the electrical fault dataset.
2. Build one multiclass target from the four fault-indicator bits.
3. Create one reproducible train / validation / test split shared by all models.
4. Tune each model family on the same training/validation split.
5. Refit each best model on train+validation and evaluate once on test.
6. Print a concise comparison table and short note about the best model.
"""

import copy
import json
from itertools import product
from pathlib import Path
import random

import matplotlib
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
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


matplotlib.use("Agg")


FAULT_COLS = ["G", "C", "B", "A"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electrical_fault_data.csv"
OUTPUT_DIR = BASE_DIR / "multiclass_fault_comparison_outputs"

RANDOM_STATE = 4320
PRIMARY_METRIC = "balanced_accuracy"
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN_VAL = 0.25

KNN_K_VALUES = [1, 3, 5, 7, 11, 15, 21, 31]
KNN_WEIGHT_VALUES = ["uniform", "distance"]
KNN_P_VALUES = [1, 2]

SVM_C_VALUES = [0.1, 1.0, 10.0, 100.0]
SVM_GAMMA_VALUES = [0.01, 0.1, 1.0, "scale"]
SVM_CLASS_WEIGHT_VALUES = [None, "balanced"]

RF_TUNE_GRID = [
    {"n_estimators": 200, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": 20, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": 12, "max_features": "sqrt", "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": None, "max_features": 0.5, "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": 20, "max_features": 0.5, "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": None, "max_features": "sqrt", "min_samples_leaf": 2},
    {"n_estimators": 400, "max_depth": 20, "max_features": "sqrt", "min_samples_leaf": 2},
]

MLP_TUNE_GRID = [
    {"hidden_dims": (128, 64), "dropout_rates": (0.20, 0.10), "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"hidden_dims": (256, 128), "dropout_rates": (0.20, 0.10), "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"hidden_dims": (128, 64), "dropout_rates": (0.10, 0.00), "learning_rate": 5e-4, "weight_decay": 1e-4},
    {"hidden_dims": (256, 128), "dropout_rates": (0.25, 0.10), "learning_rate": 5e-4, "weight_decay": 5e-4},
]

MLP_BATCH_SIZE = 128
MLP_EPOCHS = 80
MLP_PATIENCE = 12


def set_seed(seed: int):
    # Keep the split and neural-network training reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_dataframe(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def load_data(data_path: Path = DATA_PATH):
    # Read the electrical fault dataset and build the multiclass fault-pattern target.
    df = pd.read_csv(data_path)
    y = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1).to_numpy()
    X = df.drop(columns=FAULT_COLS)
    return df, X, y


def build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool):
    # Use median imputation for every model, with scaling only for distance-based models.
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
    # Keep preprocessing and model fitting together so every family stays leakage-safe.
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true, pred):
    # Compute multiclass metrics from a set of predictions.
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def evaluate(model, X, y):
    # Compute multiclass metrics using macro averaging so every fault type matters.
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


def plot_confusion_matrix(confusion_df: pd.DataFrame, title: str, path: Path):
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
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_knn_results(results_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for (weights, p_value), group in results_df.groupby(["weights", "p"]):
        sorted_group = group.sort_values("n_neighbors")
        ax.plot(
            sorted_group["n_neighbors"],
            sorted_group["val_balanced_accuracy"],
            marker="o",
            label=f"weights={weights}, p={p_value}",
        )

    ax.set_title("kNN validation balanced accuracy")
    ax.set_xlabel("n_neighbors")
    ax.set_ylabel("Validation balanced accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_svm_results(results_df: pd.DataFrame, path: Path):
    class_weight_labels = [None, "balanced"]
    fig, axes = plt.subplots(1, len(class_weight_labels), figsize=(12, 5), constrained_layout=True)

    if len(class_weight_labels) == 1:
        axes = [axes]

    gamma_labels = [str(value) for value in SVM_GAMMA_VALUES]
    c_labels = [str(value) for value in SVM_C_VALUES]

    for axis, class_weight in zip(axes, class_weight_labels):
        subset = results_df[results_df["class_weight"].astype(str) == str(class_weight)].copy()
        subset["gamma_label"] = subset["gamma"].astype(str)
        heatmap = subset.pivot(index="C", columns="gamma_label", values="val_balanced_accuracy").reindex(
            index=SVM_C_VALUES,
            columns=gamma_labels,
        )

        image = axis.imshow(heatmap.to_numpy(), cmap="viridis", aspect="auto")
        axis.set_title(f"RBF SVM validation balanced accuracy\nclass_weight={class_weight}")
        axis.set_xticks(range(len(gamma_labels)))
        axis.set_xticklabels(gamma_labels)
        axis.set_yticks(range(len(c_labels)))
        axis.set_yticklabels(c_labels)
        axis.set_xlabel("gamma")
        axis.set_ylabel("C")

        for row_index in range(heatmap.shape[0]):
            for col_index in range(heatmap.shape[1]):
                value = heatmap.iat[row_index, col_index]
                axis.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rf_results(results_df: pd.DataFrame, path: Path):
    plot_df = results_df.copy()
    plot_df["config_label"] = [
        f"n={n}\nd={d}\nf={f}\nleaf={leaf}"
        for n, d, f, leaf in zip(
            plot_df["n_estimators"],
            plot_df["max_depth"],
            plot_df["max_features"],
            plot_df["min_samples_leaf"],
        )
    ]
    plot_df = plot_df.sort_values("val_balanced_accuracy", ascending=False)

    x_positions = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(x_positions - width / 2, plot_df["val_balanced_accuracy"], width=width, label="Balanced accuracy")
    ax.bar(x_positions + width / 2, plot_df["val_f1"], width=width, label="Macro F1")
    ax.set_title("Random Forest tuned validation results")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Validation score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["config_label"], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_history(history: dict, path: Path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["validation_loss"], marker="s", label="Validation loss")
    axes[0].set_title("MLP loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    axes[1].plot(epochs, history["validation_accuracy"], marker="s", label="Validation accuracy")
    axes[1].set_title("MLP accuracy curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(final_test_df: pd.DataFrame, path: Path):
    metric_columns = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]
    pretty_labels = ["Accuracy", "Balanced acc", "Precision", "Recall", "Macro F1"]

    x_positions = np.arange(len(final_test_df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for metric_index, metric_name in enumerate(metric_columns):
        offset = (metric_index - 2) * width
        ax.bar(
            x_positions + offset,
            final_test_df[metric_name],
            width=width,
            label=pretty_labels[metric_index],
        )

    ax.set_title("Final test comparison across model families")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(final_test_df["model"])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_split_info(y_train, y_val, y_test):
    # Print split sizes and fault-pattern counts for each partition.
    print("Split sizes:")
    print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print("Fault-type counts:")
    for split_name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        counts = pd.Series(split_y).value_counts().sort_index()
        formatted = ", ".join([f"{label}={count}" for label, count in counts.items()])
        print(f"  {split_name}: {formatted}")


def encode_labels(reference_labels, labels_to_encode):
    # Convert text fault-pattern labels into integer class IDs for PyTorch.
    class_names = sorted(pd.Series(reference_labels).unique().tolist())
    class_to_index = {label: index for index, label in enumerate(class_names)}
    encoded = np.array([class_to_index[label] for label in labels_to_encode], dtype=np.int64)
    return encoded, class_names, class_to_index


def decode_labels(encoded_labels: np.ndarray, class_names: list[str]):
    return np.array([class_names[index] for index in encoded_labels], dtype=object)


class FaultDataset(Dataset):
    # Small Dataset wrapper for the neural-network comparison.

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            torch.tensor(self.X[index], dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.long),
        )


class FaultMLP(nn.Module):
    # Compact MLP modeled after HW12, adapted for quick model-family comparison.

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, int], dropout_rates: tuple[float, float]):
        super().__init__()
        hidden_one, hidden_two = hidden_dims
        dropout_one, dropout_two = dropout_rates
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_one),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_one),
            nn.Dropout(dropout_one),
            nn.Linear(hidden_one, hidden_two),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_two),
            nn.Dropout(dropout_two),
            nn.Linear(hidden_two, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def run_epoch(model, loader, loss_function, optimizer=None):
    # Shared train/eval loop for the MLP models.
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_X, batch_y in loader:
        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(batch_X)
            loss = loss_function(logits, batch_y)
            if is_training:
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * batch_y.size(0)
        total_correct += int((predictions == batch_y).sum().item())
        total_examples += int(batch_y.size(0))

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return average_loss, accuracy


def predict_mlp(model, loader):
    # Collect multiclass predictions for MLP evaluation.
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch_X, _ in loader:
            logits = model(batch_X)
            batch_predictions = torch.argmax(logits, dim=1)
            all_predictions.append(batch_predictions.cpu().numpy())

    return np.concatenate(all_predictions)


def tune_knn(X_train, y_train, X_val, y_val):
    # Follow the kNN homework idea: scale data and compare a range of neighborhood settings.
    results = []
    best_model = None
    best_params = None
    best_metrics = None

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
        results.append(
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
            best_params = {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "p": p_value,
            }
            best_metrics = metrics

    return pd.DataFrame(results), best_model, best_params, best_metrics


def tune_rbf_svm(X_train, y_train, X_val, y_val):
    # Follow the SVM homework idea: compare nonlinear margin settings on the same split.
    results = []
    best_model = None
    best_params = None
    best_metrics = None

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
        results.append(
            {
                "C": c_value,
                "gamma": gamma_value,
                "class_weight": class_weight,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_params = {
                "C": c_value,
                "gamma": gamma_value,
                "class_weight": class_weight,
            }
            best_metrics = metrics

    return pd.DataFrame(results), best_model, best_params, best_metrics


def tune_random_forest(X_train, y_train, X_val, y_val):
    # Follow the Random Forest homework idea: compare a default baseline, then tune a small grid.
    default_model = make_pipeline(
        X_train,
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=1,
            oob_score=True,
        ),
        scale_numeric=False,
    )
    default_model.fit(X_train, y_train)
    default_metrics = evaluate(default_model, X_val, y_val)
    default_oob = float(default_model.named_steps["model"].oob_score_)

    results = []
    best_model = None
    best_params = None
    best_metrics = None
    best_oob = None

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
                oob_score=True,
            ),
            scale_numeric=False,
        )
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_val, y_val)
        oob_score = float(model.named_steps["model"].oob_score_)
        results.append(
            {
                "n_estimators": params["n_estimators"],
                "max_depth": str(params["max_depth"]),
                "max_features": params["max_features"],
                "min_samples_leaf": params["min_samples_leaf"],
                "oob_accuracy": oob_score,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_model = model
            best_params = params.copy()
            best_metrics = metrics
            best_oob = oob_score

    return (
        default_metrics,
        default_oob,
        pd.DataFrame(results),
        best_model,
        best_params,
        best_metrics,
        best_oob,
    )


def fit_mlp_candidate(X_train, y_train, X_val, y_val, params: dict):
    # Train one MLP candidate and keep the best validation checkpoint plus training history.
    preprocessor = build_preprocessor(X_train, scale_numeric=True)
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)

    y_train_encoded, class_names, class_to_index = encode_labels(y_train, y_train)
    y_val_encoded = np.array([class_to_index[label] for label in y_val], dtype=np.int64)

    train_loader = DataLoader(FaultDataset(X_train_processed, y_train_encoded), batch_size=MLP_BATCH_SIZE, shuffle=True)
    train_eval_loader = DataLoader(FaultDataset(X_train_processed, y_train_encoded), batch_size=MLP_BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(FaultDataset(X_val_processed, y_val_encoded), batch_size=MLP_BATCH_SIZE, shuffle=False)

    num_classes = len(class_names)
    set_seed(RANDOM_STATE)
    model = FaultMLP(
        input_dim=X_train_processed.shape[1],
        output_dim=num_classes,
        hidden_dims=params["hidden_dims"],
        dropout_rates=params["dropout_rates"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    loss_function = nn.CrossEntropyLoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    best_epoch = 1
    wait = 0

    for epoch in range(1, MLP_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_function, optimizer=optimizer)
        validation_loss, validation_accuracy = run_epoch(model, val_loader, loss_function, optimizer=None)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        if validation_accuracy > best_val_accuracy:
            best_val_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= MLP_PATIENCE:
            break

    model.load_state_dict(best_state)
    train_predictions_encoded = predict_mlp(model, train_eval_loader)
    val_predictions_encoded = predict_mlp(model, val_loader)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "history": history,
        "best_epoch": best_epoch,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "train_metrics": evaluate_predictions(y_train_encoded, train_predictions_encoded),
        "val_metrics": evaluate_predictions(y_val_encoded, val_predictions_encoded),
        "train_predictions": decode_labels(train_predictions_encoded, class_names),
        "val_predictions": decode_labels(val_predictions_encoded, class_names),
    }


def tune_mlp(X_train, y_train, X_val, y_val):
    # Follow the HW12 idea: scale features, train a compact MLP, and keep the best validation checkpoint.
    results = []
    best_params = None
    best_metrics = None
    best_run = None

    for params in MLP_TUNE_GRID:
        run = fit_mlp_candidate(X_train, y_train, X_val, y_val, params)
        metrics = run["val_metrics"]
        results.append(
            {
                "hidden_dims": str(params["hidden_dims"]),
                "dropout_rates": str(params["dropout_rates"]),
                "learning_rate": params["learning_rate"],
                "weight_decay": params["weight_decay"],
                "best_epoch": run["best_epoch"],
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_f1": metrics["f1"],
            }
        )

        if best_metrics is None or metrics[PRIMARY_METRIC] > best_metrics[PRIMARY_METRIC]:
            best_params = params.copy()
            best_params["best_epoch"] = run["best_epoch"]
            best_metrics = metrics
            best_run = run

    return pd.DataFrame(results), best_params, best_metrics, best_run


def refit_and_evaluate(X_train_val, y_train_val, X_test, y_test, *, model_name: str, best_params: dict, class_names: list[str]):
    if model_name == "kNN":
        final_model = make_pipeline(
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
        final_model.fit(X_train_val, y_train_val)
        test_predictions = final_model.predict(X_test)
    elif model_name == "RBF SVM":
        final_model = make_pipeline(
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
        final_model.fit(X_train_val, y_train_val)
        test_predictions = final_model.predict(X_test)
    elif model_name == "MLP":
        preprocessor = build_preprocessor(X_train_val, scale_numeric=True)
        X_train_val_processed = preprocessor.fit_transform(X_train_val).astype(np.float32)
        X_test_processed = preprocessor.transform(X_test).astype(np.float32)

        y_train_val_encoded, class_names, class_to_index = encode_labels(y_train_val, y_train_val)
        y_test_encoded = np.array([class_to_index[label] for label in y_test], dtype=np.int64)

        train_loader = DataLoader(
            FaultDataset(X_train_val_processed, y_train_val_encoded),
            batch_size=MLP_BATCH_SIZE,
            shuffle=True,
        )
        test_loader = DataLoader(
            FaultDataset(X_test_processed, y_test_encoded),
            batch_size=MLP_BATCH_SIZE,
            shuffle=False,
        )

        set_seed(RANDOM_STATE)
        model = FaultMLP(
            input_dim=X_train_val_processed.shape[1],
            output_dim=len(class_names),
            hidden_dims=best_params["hidden_dims"],
            dropout_rates=best_params["dropout_rates"],
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
        )
        loss_function = nn.CrossEntropyLoss()

        for _ in range(best_params["best_epoch"]):
            run_epoch(model, train_loader, loss_function, optimizer=optimizer)

        test_predictions = decode_labels(predict_mlp(model, test_loader), class_names)
    else:
        final_model = make_pipeline(
            X_train_val,
            RandomForestClassifier(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                max_features=best_params["max_features"],
                min_samples_leaf=best_params["min_samples_leaf"],
                random_state=RANDOM_STATE,
                n_jobs=1,
                oob_score=False,
            ),
            scale_numeric=False,
        )
        final_model.fit(X_train_val, y_train_val)
        test_predictions = final_model.predict(X_test)

    metrics = evaluate_predictions(y_test, test_predictions)
    confusion_df = compute_confusion_df(y_test, test_predictions, class_names)
    return metrics, confusion_df


def main():
    set_seed(RANDOM_STATE)
    ensure_output_dir()
    df, X, y = load_data(DATA_PATH)
    class_names = sorted(pd.Series(y).unique().tolist())

    # Use one shared stratified split so every model family sees the same rows.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN_VAL,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print("Capstone multiclass model comparison")
    print("====================================")
    print(f"Rows: {len(df)}")
    print(f"Primary validation metric: {PRIMARY_METRIC}")
    print_split_info(y_train, y_val, y_test)

    knn_results_df, best_knn_model, best_knn_params, best_knn_metrics = tune_knn(X_train, y_train, X_val, y_val)
    print("\nkNN validation results:")
    print(knn_results_df.to_string(index=False))
    print(
        "Best kNN by validation "
        f"{PRIMARY_METRIC}: n_neighbors={best_knn_params['n_neighbors']}, "
        f"weights={best_knn_params['weights']}, p={best_knn_params['p']}, "
        f"{format_metrics(best_knn_metrics)}"
    )

    svm_results_df, best_svm_model, best_svm_params, best_svm_metrics = tune_rbf_svm(X_train, y_train, X_val, y_val)
    print("\nRBF SVM validation results:")
    print(svm_results_df.to_string(index=False))
    print(
        "Best RBF SVM by validation "
        f"{PRIMARY_METRIC}: C={best_svm_params['C']}, gamma={best_svm_params['gamma']}, "
        f"class_weight={best_svm_params['class_weight']}, {format_metrics(best_svm_metrics)}"
    )

    (
        default_rf_metrics,
        default_rf_oob,
        rf_results_df,
        best_rf_model,
        best_rf_params,
        best_rf_metrics,
        best_rf_oob,
    ) = tune_random_forest(X_train, y_train, X_val, y_val)
    print("\nRandom Forest default validation result:")
    print(f"  {format_metrics(default_rf_metrics)}")
    print(f"  OOB accuracy={default_rf_oob:.4f}")
    print("\nRandom Forest tuned validation results:")
    print(rf_results_df.to_string(index=False))
    print(
        "Best Random Forest by validation "
        f"{PRIMARY_METRIC}: n_estimators={best_rf_params['n_estimators']}, "
        f"max_depth={best_rf_params['max_depth']}, max_features={best_rf_params['max_features']}, "
        f"min_samples_leaf={best_rf_params['min_samples_leaf']}, "
        f"OOB accuracy={best_rf_oob:.4f}, {format_metrics(best_rf_metrics)}"
    )

    mlp_results_df, best_mlp_params, best_mlp_metrics, best_mlp_run = tune_mlp(X_train, y_train, X_val, y_val)
    print("\nMLP validation results:")
    print(mlp_results_df.to_string(index=False))
    print(
        "Best MLP by validation "
        f"{PRIMARY_METRIC}: hidden_dims={best_mlp_params['hidden_dims']}, "
        f"dropout_rates={best_mlp_params['dropout_rates']}, learning_rate={best_mlp_params['learning_rate']}, "
        f"weight_decay={best_mlp_params['weight_decay']}, best_epoch={best_mlp_params['best_epoch']}, "
        f"{format_metrics(best_mlp_metrics)}"
    )

    plot_knn_results(knn_results_df, OUTPUT_DIR / "knn_validation_plot.png")
    plot_svm_results(svm_results_df, OUTPUT_DIR / "rbf_svm_validation_plot.png")
    plot_rf_results(rf_results_df, OUTPUT_DIR / "random_forest_validation_plot.png")
    plot_mlp_history(best_mlp_run["history"], OUTPUT_DIR / "mlp_training_history.png")

    train_val_metrics = {
        "kNN": {
            "train": evaluate(best_knn_model, X_train, y_train),
            "val": evaluate(best_knn_model, X_val, y_val),
        },
        "RBF SVM": {
            "train": evaluate(best_svm_model, X_train, y_train),
            "val": evaluate(best_svm_model, X_val, y_val),
        },
        "Random Forest": {
            "train": evaluate(best_rf_model, X_train, y_train),
            "val": evaluate(best_rf_model, X_val, y_val),
        },
        "MLP": {
            "train": best_mlp_run["train_metrics"],
            "val": best_mlp_run["val_metrics"],
        },
    }

    validation_comparison_df = pd.DataFrame(
        [
            {
                "model": "kNN",
                "key_hyperparameters": (
                    f"k={best_knn_params['n_neighbors']}, weights={best_knn_params['weights']}, p={best_knn_params['p']}"
                ),
                "val_accuracy": best_knn_metrics["accuracy"],
                "val_balanced_accuracy": best_knn_metrics["balanced_accuracy"],
                "val_f1": best_knn_metrics["f1"],
            },
            {
                "model": "RBF SVM",
                "key_hyperparameters": (
                    f"C={best_svm_params['C']}, gamma={best_svm_params['gamma']}, "
                    f"class_weight={best_svm_params['class_weight']}"
                ),
                "val_accuracy": best_svm_metrics["accuracy"],
                "val_balanced_accuracy": best_svm_metrics["balanced_accuracy"],
                "val_f1": best_svm_metrics["f1"],
            },
            {
                "model": "Random Forest",
                "key_hyperparameters": (
                    f"n_estimators={best_rf_params['n_estimators']}, max_depth={best_rf_params['max_depth']}, "
                    f"max_features={best_rf_params['max_features']}, min_samples_leaf={best_rf_params['min_samples_leaf']}"
                ),
                "val_accuracy": best_rf_metrics["accuracy"],
                "val_balanced_accuracy": best_rf_metrics["balanced_accuracy"],
                "val_f1": best_rf_metrics["f1"],
            },
            {
                "model": "MLP",
                "key_hyperparameters": (
                    f"hidden_dims={best_mlp_params['hidden_dims']}, dropout={best_mlp_params['dropout_rates']}, "
                    f"lr={best_mlp_params['learning_rate']}, wd={best_mlp_params['weight_decay']}"
                ),
                "val_accuracy": best_mlp_metrics["accuracy"],
                "val_balanced_accuracy": best_mlp_metrics["balanced_accuracy"],
                "val_f1": best_mlp_metrics["f1"],
            },
        ]
    )
    print("\nValidation comparison table:")
    print(validation_comparison_df.to_string(index=False))

    validation_winner_row = validation_comparison_df.loc[
        validation_comparison_df["val_balanced_accuracy"].idxmax()
    ]
    validation_winner_name = str(validation_winner_row["model"])

    final_test_runs = {
        "kNN": refit_and_evaluate(
            X_train_val, y_train_val, X_test, y_test,
            model_name="kNN", best_params=best_knn_params, class_names=class_names
        ),
        "RBF SVM": refit_and_evaluate(
            X_train_val, y_train_val, X_test, y_test,
            model_name="RBF SVM", best_params=best_svm_params, class_names=class_names
        ),
        "Random Forest": refit_and_evaluate(
            X_train_val, y_train_val, X_test, y_test,
            model_name="Random Forest", best_params=best_rf_params, class_names=class_names
        ),
        "MLP": refit_and_evaluate(
            X_train_val, y_train_val, X_test, y_test,
            model_name="MLP", best_params=best_mlp_params, class_names=class_names
        ),
    }

    final_test_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "test_accuracy": run[0]["accuracy"],
                "test_balanced_accuracy": run[0]["balanced_accuracy"],
                "test_precision": run[0]["precision"],
                "test_recall": run[0]["recall"],
                "test_f1": run[0]["f1"],
            }
            for model_name, run in final_test_runs.items()
        ]
    )
    print("\nFinal test comparison:")
    print(final_test_df.to_string(index=False))

    plot_final_comparison(final_test_df, OUTPUT_DIR / "final_test_comparison_plot.png")

    for model_name, (_, confusion_df) in final_test_runs.items():
        safe_name = model_name.lower().replace(" ", "_")
        save_dataframe(confusion_df.reset_index(), OUTPUT_DIR / f"{safe_name}_test_confusion_matrix.csv")
        plot_confusion_matrix(
            confusion_df,
            title=f"{model_name} test confusion matrix",
            path=OUTPUT_DIR / f"{safe_name}_test_confusion_matrix.png",
        )

    model_summary_rows = [
        {
            "model": "kNN",
            "key_hyperparameters": f"k={best_knn_params['n_neighbors']}, weights={best_knn_params['weights']}, p={best_knn_params['p']}",
            "train_accuracy": train_val_metrics["kNN"]["train"]["accuracy"],
            "train_balanced_accuracy": train_val_metrics["kNN"]["train"]["balanced_accuracy"],
            "train_precision": train_val_metrics["kNN"]["train"]["precision"],
            "train_recall": train_val_metrics["kNN"]["train"]["recall"],
            "train_f1": train_val_metrics["kNN"]["train"]["f1"],
            "val_accuracy": train_val_metrics["kNN"]["val"]["accuracy"],
            "val_balanced_accuracy": train_val_metrics["kNN"]["val"]["balanced_accuracy"],
            "val_precision": train_val_metrics["kNN"]["val"]["precision"],
            "val_recall": train_val_metrics["kNN"]["val"]["recall"],
            "val_f1": train_val_metrics["kNN"]["val"]["f1"],
            "test_accuracy": final_test_runs["kNN"][0]["accuracy"],
            "test_balanced_accuracy": final_test_runs["kNN"][0]["balanced_accuracy"],
            "test_precision": final_test_runs["kNN"][0]["precision"],
            "test_recall": final_test_runs["kNN"][0]["recall"],
            "test_f1": final_test_runs["kNN"][0]["f1"],
        },
        {
            "model": "RBF SVM",
            "key_hyperparameters": f"C={best_svm_params['C']}, gamma={best_svm_params['gamma']}, class_weight={best_svm_params['class_weight']}",
            "train_accuracy": train_val_metrics["RBF SVM"]["train"]["accuracy"],
            "train_balanced_accuracy": train_val_metrics["RBF SVM"]["train"]["balanced_accuracy"],
            "train_precision": train_val_metrics["RBF SVM"]["train"]["precision"],
            "train_recall": train_val_metrics["RBF SVM"]["train"]["recall"],
            "train_f1": train_val_metrics["RBF SVM"]["train"]["f1"],
            "val_accuracy": train_val_metrics["RBF SVM"]["val"]["accuracy"],
            "val_balanced_accuracy": train_val_metrics["RBF SVM"]["val"]["balanced_accuracy"],
            "val_precision": train_val_metrics["RBF SVM"]["val"]["precision"],
            "val_recall": train_val_metrics["RBF SVM"]["val"]["recall"],
            "val_f1": train_val_metrics["RBF SVM"]["val"]["f1"],
            "test_accuracy": final_test_runs["RBF SVM"][0]["accuracy"],
            "test_balanced_accuracy": final_test_runs["RBF SVM"][0]["balanced_accuracy"],
            "test_precision": final_test_runs["RBF SVM"][0]["precision"],
            "test_recall": final_test_runs["RBF SVM"][0]["recall"],
            "test_f1": final_test_runs["RBF SVM"][0]["f1"],
        },
        {
            "model": "Random Forest",
            "key_hyperparameters": (
                f"n_estimators={best_rf_params['n_estimators']}, max_depth={best_rf_params['max_depth']}, "
                f"max_features={best_rf_params['max_features']}, min_samples_leaf={best_rf_params['min_samples_leaf']}"
            ),
            "train_accuracy": train_val_metrics["Random Forest"]["train"]["accuracy"],
            "train_balanced_accuracy": train_val_metrics["Random Forest"]["train"]["balanced_accuracy"],
            "train_precision": train_val_metrics["Random Forest"]["train"]["precision"],
            "train_recall": train_val_metrics["Random Forest"]["train"]["recall"],
            "train_f1": train_val_metrics["Random Forest"]["train"]["f1"],
            "val_accuracy": train_val_metrics["Random Forest"]["val"]["accuracy"],
            "val_balanced_accuracy": train_val_metrics["Random Forest"]["val"]["balanced_accuracy"],
            "val_precision": train_val_metrics["Random Forest"]["val"]["precision"],
            "val_recall": train_val_metrics["Random Forest"]["val"]["recall"],
            "val_f1": train_val_metrics["Random Forest"]["val"]["f1"],
            "test_accuracy": final_test_runs["Random Forest"][0]["accuracy"],
            "test_balanced_accuracy": final_test_runs["Random Forest"][0]["balanced_accuracy"],
            "test_precision": final_test_runs["Random Forest"][0]["precision"],
            "test_recall": final_test_runs["Random Forest"][0]["recall"],
            "test_f1": final_test_runs["Random Forest"][0]["f1"],
            "oob_accuracy_default": default_rf_oob,
            "oob_accuracy_best_tuned": best_rf_oob,
        },
        {
            "model": "MLP",
            "key_hyperparameters": (
                f"hidden_dims={best_mlp_params['hidden_dims']}, dropout={best_mlp_params['dropout_rates']}, "
                f"lr={best_mlp_params['learning_rate']}, wd={best_mlp_params['weight_decay']}, epoch={best_mlp_params['best_epoch']}"
            ),
            "train_accuracy": train_val_metrics["MLP"]["train"]["accuracy"],
            "train_balanced_accuracy": train_val_metrics["MLP"]["train"]["balanced_accuracy"],
            "train_precision": train_val_metrics["MLP"]["train"]["precision"],
            "train_recall": train_val_metrics["MLP"]["train"]["recall"],
            "train_f1": train_val_metrics["MLP"]["train"]["f1"],
            "val_accuracy": train_val_metrics["MLP"]["val"]["accuracy"],
            "val_balanced_accuracy": train_val_metrics["MLP"]["val"]["balanced_accuracy"],
            "val_precision": train_val_metrics["MLP"]["val"]["precision"],
            "val_recall": train_val_metrics["MLP"]["val"]["recall"],
            "val_f1": train_val_metrics["MLP"]["val"]["f1"],
            "test_accuracy": final_test_runs["MLP"][0]["accuracy"],
            "test_balanced_accuracy": final_test_runs["MLP"][0]["balanced_accuracy"],
            "test_precision": final_test_runs["MLP"][0]["precision"],
            "test_recall": final_test_runs["MLP"][0]["recall"],
            "test_f1": final_test_runs["MLP"][0]["f1"],
        },
    ]
    model_summary_df = pd.DataFrame(model_summary_rows)

    save_dataframe(knn_results_df, OUTPUT_DIR / "knn_validation_search_results.csv")
    save_dataframe(svm_results_df, OUTPUT_DIR / "rbf_svm_validation_search_results.csv")
    save_dataframe(rf_results_df, OUTPUT_DIR / "random_forest_validation_search_results.csv")
    save_dataframe(mlp_results_df, OUTPUT_DIR / "mlp_validation_search_results.csv")
    save_dataframe(validation_comparison_df, OUTPUT_DIR / "validation_comparison_table.csv")
    save_dataframe(final_test_df, OUTPUT_DIR / "final_test_comparison_table.csv")
    save_dataframe(model_summary_df, OUTPUT_DIR / "model_summary_train_val_test_metrics.csv")

    summary_data = {
        "dataset_rows": int(len(df)),
        "random_state": RANDOM_STATE,
        "primary_validation_metric": PRIMARY_METRIC,
        "class_names": class_names,
        "split_sizes": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "split_class_counts": {
            "train": pd.Series(y_train).value_counts().sort_index().to_dict(),
            "validation": pd.Series(y_val).value_counts().sort_index().to_dict(),
            "test": pd.Series(y_test).value_counts().sort_index().to_dict(),
        },
        "default_random_forest": {
            "validation_metrics": default_rf_metrics,
            "oob_accuracy": default_rf_oob,
        },
        "best_model_by_validation_metric": validation_winner_name,
        "models": {
            "kNN": {
                "best_params": {
                    "n_neighbors": best_knn_params["n_neighbors"],
                    "weights": best_knn_params["weights"],
                    "p": best_knn_params["p"],
                },
                "train_metrics": train_val_metrics["kNN"]["train"],
                "validation_metrics": train_val_metrics["kNN"]["val"],
                "test_metrics": final_test_runs["kNN"][0],
            },
            "RBF SVM": {
                "best_params": {
                    "C": best_svm_params["C"],
                    "gamma": str(best_svm_params["gamma"]),
                    "class_weight": str(best_svm_params["class_weight"]),
                },
                "train_metrics": train_val_metrics["RBF SVM"]["train"],
                "validation_metrics": train_val_metrics["RBF SVM"]["val"],
                "test_metrics": final_test_runs["RBF SVM"][0],
            },
            "Random Forest": {
                "best_params": {
                    "n_estimators": best_rf_params["n_estimators"],
                    "max_depth": best_rf_params["max_depth"],
                    "max_features": str(best_rf_params["max_features"]),
                    "min_samples_leaf": best_rf_params["min_samples_leaf"],
                },
                "train_metrics": train_val_metrics["Random Forest"]["train"],
                "validation_metrics": train_val_metrics["Random Forest"]["val"],
                "test_metrics": final_test_runs["Random Forest"][0],
                "oob_accuracy_best_tuned": best_rf_oob,
            },
            "MLP": {
                "best_params": {
                    "hidden_dims": list(best_mlp_params["hidden_dims"]),
                    "dropout_rates": list(best_mlp_params["dropout_rates"]),
                    "learning_rate": best_mlp_params["learning_rate"],
                    "weight_decay": best_mlp_params["weight_decay"],
                    "best_epoch": best_mlp_params["best_epoch"],
                },
                "train_metrics": train_val_metrics["MLP"]["train"],
                "validation_metrics": train_val_metrics["MLP"]["val"],
                "test_metrics": final_test_runs["MLP"][0],
            },
        },
        "saved_files": {
            "validation_search_csvs": [
                "knn_validation_search_results.csv",
                "rbf_svm_validation_search_results.csv",
                "random_forest_validation_search_results.csv",
                "mlp_validation_search_results.csv",
            ],
            "summary_tables": [
                "validation_comparison_table.csv",
                "final_test_comparison_table.csv",
                "model_summary_train_val_test_metrics.csv",
            ],
            "plots": [
                "knn_validation_plot.png",
                "rbf_svm_validation_plot.png",
                "random_forest_validation_plot.png",
                "mlp_training_history.png",
                "final_test_comparison_plot.png",
            ],
        },
    }
    save_json(summary_data, OUTPUT_DIR / "comparison_summary.json")

    report_lines = [
        "# Multiclass Fault Comparison Summary",
        "",
        f"- Best model by validation {PRIMARY_METRIC}: **{validation_winner_name}**",
        f"- Rows: `{len(df)}`",
        f"- Split sizes: train `{len(y_train)}`, val `{len(y_val)}`, test `{len(y_test)}`",
        "",
        "## Final Test Snapshot",
        "",
    ]
    for row in model_summary_rows:
        report_lines.append(
            f"- **{row['model']}**: "
            f"accuracy={row['test_accuracy']:.4f}, "
            f"balanced_accuracy={row['test_balanced_accuracy']:.4f}, "
            f"precision={row['test_precision']:.4f}, "
            f"recall={row['test_recall']:.4f}, "
            f"f1={row['test_f1']:.4f}"
        )
    (OUTPUT_DIR / "comparison_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nBest-model note:")
    print(
        f"  Based on validation {PRIMARY_METRIC}, the best model was {validation_winner_name}. "
        "That model is the safest choice if you want to keep model selection tied to validation only."
    )
    print(f"  Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
