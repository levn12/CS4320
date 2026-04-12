from pathlib import Path
import copy
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "saas_customer_churn_mlp.csv"
PLOT_PATH = BASE_DIR / "hw11_part_a_training_dynamics.png"

RANDOM_STATE = 4320
TEST_SIZE = 0.20
VAL_SIZE = 0.25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BASELINE_EPOCHS = 120
MLP_EPOCHS = 80
PATIENCE = 12


def compute_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def predict_probabilities(model, X_array):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_array, dtype=torch.float32)).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy()


def train_model(X_train, y_train, X_val, y_val, hidden_dim=None, epochs=80):
    if hidden_dim is None:
        model = nn.Sequential(nn.Linear(X_train.shape[1], 1))
    else:
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1),
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.to_numpy(), dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_roc_auc": [], "val_roc_auc": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X).squeeze(1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_probs = predict_probabilities(model, X_train)
        val_probs = predict_probabilities(model, X_val)

        with torch.no_grad():
            train_loss = criterion(model(torch.tensor(X_train, dtype=torch.float32)).squeeze(1), y_train_tensor).item()
            val_loss = criterion(model(torch.tensor(X_val, dtype=torch.float32)).squeeze(1), y_val_tensor).item()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(float(val_loss))
        history["train_roc_auc"].append(compute_metrics(y_train, train_probs)["roc_auc"])
        history["val_roc_auc"].append(compute_metrics(y_val, val_probs)["roc_auc"])

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)

    train_probs = predict_probabilities(model, X_train)
    val_probs = predict_probabilities(model, X_val)

    return {
        "model": model,
        "history": history,
        "epochs_ran": len(history["epoch"]),
        "train_metrics": compute_metrics(y_train, train_probs),
        "val_metrics": compute_metrics(y_val, val_probs),
    }


def plot_training_curves(result_32, result_64, result_128, result_256):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(result_32["history"]["epoch"], result_32["history"]["train_loss"], label="MLP 32 train")
    axes[0].plot(result_32["history"]["epoch"], result_32["history"]["val_loss"], label="MLP 32 val")
    axes[0].plot(result_64["history"]["epoch"], result_64["history"]["train_loss"], label="MLP 64 train")
    axes[0].plot(result_64["history"]["epoch"], result_64["history"]["val_loss"], label="MLP 64 val")
    axes[0].plot(result_128["history"]["epoch"], result_128["history"]["train_loss"], "--", label="MLP 128 train")
    axes[0].plot(result_128["history"]["epoch"], result_128["history"]["val_loss"], "--", label="MLP 128 val")
    axes[0].plot(result_256["history"]["epoch"], result_256["history"]["train_loss"], ":", label="MLP 256 train")
    axes[0].plot(result_256["history"]["epoch"], result_256["history"]["val_loss"], ":", label="MLP 256 val")
    axes[0].set_title("Loss vs. Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy loss")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend()

    axes[1].plot(result_32["history"]["epoch"], result_32["history"]["train_roc_auc"], label="MLP 32 train")
    axes[1].plot(result_32["history"]["epoch"], result_32["history"]["val_roc_auc"], label="MLP 32 val")
    axes[1].plot(result_64["history"]["epoch"], result_64["history"]["train_roc_auc"], label="MLP 64 train")
    axes[1].plot(result_64["history"]["epoch"], result_64["history"]["val_roc_auc"], label="MLP 64 val")
    axes[1].plot(result_128["history"]["epoch"], result_128["history"]["train_roc_auc"], "--", label="MLP 128 train")
    axes[1].plot(result_128["history"]["epoch"], result_128["history"]["val_roc_auc"], "--", label="MLP 128 val")
    axes[1].plot(result_256["history"]["epoch"], result_256["history"]["train_roc_auc"], ":", label="MLP 256 train")
    axes[1].plot(result_256["history"]["epoch"], result_256["history"]["val_roc_auc"], ":", label="MLP 256 val")
    axes[1].set_title("ROC-AUC vs. Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn_risk"])
    y = df["churn_risk"].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X_train.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
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

    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    baseline_result = train_model(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        hidden_dim=None,
        epochs=BASELINE_EPOCHS,
    )
    mlp_32_result = train_model(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        hidden_dim=32,
        epochs=MLP_EPOCHS,
    )
    mlp_64_result = train_model(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        hidden_dim=64,
        epochs=MLP_EPOCHS,
    )
    mlp_128_result = train_model(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        hidden_dim=128,
        epochs=MLP_EPOCHS,
    )
    mlp_256_result = train_model(
        X_train_processed,
        y_train,
        X_val_processed,
        y_val,
        hidden_dim=256,
        epochs=MLP_EPOCHS,
    )

    plot_training_curves(mlp_32_result, mlp_64_result, mlp_128_result, mlp_256_result)

    comparison_df = pd.DataFrame(
        [
            {"model": "Baseline logistic regression", "split": "train", **baseline_result["train_metrics"]},
            {"model": "Baseline logistic regression", "split": "validation", **baseline_result["val_metrics"]},
            {"model": "MLP hidden_dim=32", "split": "train", **mlp_32_result["train_metrics"]},
            {"model": "MLP hidden_dim=32", "split": "validation", **mlp_32_result["val_metrics"]},
            {"model": "MLP hidden_dim=64", "split": "train", **mlp_64_result["train_metrics"]},
            {"model": "MLP hidden_dim=64", "split": "validation", **mlp_64_result["val_metrics"]},
            {"model": "MLP hidden_dim=128", "split": "train", **mlp_128_result["train_metrics"]},
            {"model": "MLP hidden_dim=128", "split": "validation", **mlp_128_result["val_metrics"]},
            {"model": "MLP hidden_dim=256", "split": "train", **mlp_256_result["train_metrics"]},
            {"model": "MLP hidden_dim=256", "split": "validation", **mlp_256_result["val_metrics"]},
        ]
    )[["model", "split", "accuracy", "precision", "recall", "f1", "roc_auc"]]

    best_name = "Baseline logistic regression"
    best_result = baseline_result
    best_val_auc = baseline_result["val_metrics"]["roc_auc"]

    if mlp_32_result["val_metrics"]["roc_auc"] > best_val_auc:
        best_name = "MLP hidden_dim=32"
        best_result = mlp_32_result
        best_val_auc = mlp_32_result["val_metrics"]["roc_auc"]

    if mlp_64_result["val_metrics"]["roc_auc"] > best_val_auc:
        best_name = "MLP hidden_dim=64"
        best_result = mlp_64_result
        best_val_auc = mlp_64_result["val_metrics"]["roc_auc"]

    if mlp_128_result["val_metrics"]["roc_auc"] > best_val_auc:
        best_name = "MLP hidden_dim=128"
        best_result = mlp_128_result
        best_val_auc = mlp_128_result["val_metrics"]["roc_auc"]

    if mlp_256_result["val_metrics"]["roc_auc"] > best_val_auc:
        best_name = "MLP hidden_dim=256"
        best_result = mlp_256_result

    test_probs = predict_probabilities(best_result["model"], X_test_processed)
    test_metrics = compute_metrics(y_test, test_probs)

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    print("Assignment 11 Part A - Neural Networks (MLP)")
    print("=============================================")
    print(f"Rows: {len(df)}")
    print(f"Train/validation/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print("Missing-value summary:")
    print(missing_counts.to_string() if not missing_counts.empty else "No missing values found.")

    print("\nBaseline and MLP comparison:")
    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
            },
        )
    )

    print("\nArchitecture variation:")
    print("Changed exactly one setting: hidden layer width in a 2-hidden-layer MLP (32, 64, 128, and 256 units).")
    print(f"MLP 32 epochs run before early stopping: {mlp_32_result['epochs_ran']}")
    print(f"MLP 64 epochs run before early stopping: {mlp_64_result['epochs_ran']}")
    print(f"MLP 128 epochs run before early stopping: {mlp_128_result['epochs_ran']}")
    print(f"MLP 256 epochs run before early stopping: {mlp_256_result['epochs_ran']}")
    print(f"Saved training-dynamics plot: {PLOT_PATH}")

    print("\nModel selected from validation evidence:")
    print(f"Chosen model: {best_name}")
    print(
        f"Validation ROC-AUC values: baseline={baseline_result['val_metrics']['roc_auc']:.4f}, "
        f"mlp32={mlp_32_result['val_metrics']['roc_auc']:.4f}, "
        f"mlp64={mlp_64_result['val_metrics']['roc_auc']:.4f}, "
        f"mlp128={mlp_128_result['val_metrics']['roc_auc']:.4f}, "
        f"mlp256={mlp_256_result['val_metrics']['roc_auc']:.4f}"
    )

    print("\nFinal one-time test evaluation for the chosen model:")
    print(
        pd.DataFrame([{"model": best_name, "split": "test", **test_metrics}]).to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
