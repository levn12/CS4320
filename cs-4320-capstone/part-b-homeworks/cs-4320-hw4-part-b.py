"""
CS 4320 - Assignment 4 (Part B)

Simple workflow:
1) Load electrical fault data
2) Build one binary target: fault vs no fault
3) Split into train/validation/test
4) Preprocess with sklearn (fit on train only)
5) Train logistic regression with manual gradient descent
6) Evaluate on test and save loss curve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Fault indicator columns from dataset.
FAULT_COLS = ["G", "C", "B", "A"]

# File/output paths.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"
PLOT_PATH = BASE_DIR / "hw4_part_b_binary_loss_curve.png"

# Split/training settings.
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 4320
LR = 0.05
EPOCHS = 1200
EPS = 1e-12


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add leading column of ones for intercept term."""
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack((ones, X))


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Binary cross-entropy loss."""
    p = sigmoid(Xb @ w)
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_log_grad(Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of binary cross-entropy wrt weights."""
    p = sigmoid(Xb @ w)
    return (Xb.T @ (p - y)) / Xb.shape[0]


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Binary target: 1 if any of G/C/B/A is faulted, else 0.
    fault_matrix = df[FAULT_COLS].to_numpy(dtype=np.float64)
    y = (fault_matrix.sum(axis=1) > 0).astype(np.float64)

    # Features are measured currents/voltages.
    X_df = df.drop(columns=FAULT_COLS)
    feature_cols = X_df.columns.tolist()

    # Split test first, then split remaining into train/val.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )

    # Sklearn preprocessing, fitted on train only.
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, feature_cols)],
        remainder="drop",
    )

    X_train_p = np.asarray(pre.fit_transform(X_train), dtype=np.float64)
    X_val_p = np.asarray(pre.transform(X_val), dtype=np.float64)
    X_test_p = np.asarray(pre.transform(X_test), dtype=np.float64)

    Xb_train = add_bias_column(X_train_p)
    Xb_val = add_bias_column(X_val_p)
    Xb_test = add_bias_column(X_test_p)

    # Train binary logistic regression with gradient descent.
    w = np.zeros(Xb_train.shape[1], dtype=np.float64)
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(EPOCHS):
        grad = binary_log_grad(Xb_train, y_train, w)
        w = w - LR * grad

        tr_loss = binary_log_loss(Xb_train, y_train, w)
        va_loss = binary_log_loss(Xb_val, y_val, w)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if (epoch + 1) % 100 == 0:
            print(f"epoch={epoch + 1:4d} train_loss={tr_loss:.5f} val_loss={va_loss:.5f}")

    # Evaluate on test set.
    p_test = sigmoid(Xb_test @ w)
    y_pred = (p_test >= 0.5).astype(np.float64)
    metrics = binary_metrics(y_test, y_pred)

    print("\nData summary")
    print(f"  rows: {len(df)}")
    print(f"  features: {feature_cols}")
    print(f"  train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"  no-fault rows: {int(np.sum(y == 0))}")
    print(f"  fault rows: {int(np.sum(y == 1))}")

    print("\nBinary fault detection metrics")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")

    # Save loss curve.
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("binary cross-entropy")
    plt.title("HW4 Part B - Binary Logistic Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"\nSaved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
