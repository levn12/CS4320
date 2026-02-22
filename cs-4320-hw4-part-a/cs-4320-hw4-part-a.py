"""CS 4320 HW4 Part A: Pipeline preprocessing + manual linear regression.

Workflow:
1) Load and split the concrete strength dataset (train/val/test).
2) Build a preprocessing pipeline (impute, scale, one-hot encode).
3) Train linear regression weights with vectorized batch gradient descent.
4) Plot train/validation MSE curves and report final test metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Start with some useful constants for later.

# Identify the column we are predicting
TARGET_COL = "compressive_strength_mpa"
# Save the path of the current file's directory for loading/saving data and plots.
BASE_DIR = Path(__file__).resolve().parent
# Locate the dataset CSV file (assumed to be in the same directory as this script).
DATA_PATH = BASE_DIR / "concrete_compressive_strength.csv"
# Path to save the loss curve plot (will be created in the same directory as this script).
LOSS_PLOT_PATH = BASE_DIR / "loss_curve.png"

# Split sizes and random seed for reproducibility.
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 4320

# Gradient descent hyperparameters (learning rate and number of epochs).
LR = 0.01
EPOCHS = 1500

# Here are some helper functions that will be used in the main workflow.

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Return X with a leading column of ones for the intercept term."""
    # Create a column of ones with the same number of rows as X.
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    # Add the rows of ones as the first column of X and return the new array.
    return np.hstack((ones, X))
    """ Math note: we add a bias column of ones to allow the model to learn an intercept term.
    It's ones so when we multiply by the weights, the first weight (w[0]) effectively becomes
    the intercept that can shift the predictions up or down."""


def mse_loss(Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Compute mean squared error using bias-augmented matrix Xb."""
    residuals = Xb @ w - y
    return float(np.mean(residuals ** 2))


def mse_grad(Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute vectorized MSE gradient: (2/n) * Xb^T (Xb w - y)."""
    n = Xb.shape[0]
    residuals = Xb @ w - y
    return (2.0 / n) * (Xb.T @ residuals)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_pred - y_true)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R^2)."""
    # Sum of squares of residuals
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    # Total sum of squares
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    # R^2 is 1 - (SS_res / SS_tot)
    return 1.0 - (ss_res / ss_tot)


def main() -> None:
    # Load dataset and split target/features.
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    X_df = df.drop(columns=[TARGET_COL])

    # Detect numeric vs categorical columns to route through separate pipelines.
    numeric_features = [c for c in X_df.columns if is_numeric_dtype(X_df[c])]
    categorical_features = [c for c in X_df.columns if c not in numeric_features]

    # Split test first, then take validation from the remaining train/val data.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # We want the test_size for validation relative to the remaining trainval data, not the original full dataset.
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
    )

    # Numeric preprocessing: fill missing values, then standardize scale.
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing: fill missing categories, then one-hot encode.
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine per-type transforms into one column-wise preprocessing object.
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )

    # Fit preprocessing ONLY on train, then transform val/test (prevents leakage).
    X_train_p = np.asarray(pre.fit_transform(X_train), dtype=np.float64)
    X_val_p = np.asarray(pre.transform(X_val), dtype=np.float64)
    X_test_p = np.asarray(pre.transform(X_test), dtype=np.float64)

    # Add intercept column once for each split.
    Xb_tr = add_bias_column(X_train_p)
    Xb_va = add_bias_column(X_val_p)
    Xb_te = add_bias_column(X_test_p)

    # Initialize weights to zeros.
    w = np.zeros(Xb_tr.shape[1], dtype=np.float64)

    # Create lists to track train/validation losses for plotting later.
    train_losses: list[float] = []
    val_losses: list[float] = []

    # Batch gradient descent loop.
    for epoch in range(EPOCHS):
        # 1) gradient on training data
        grad = mse_grad(Xb_tr, y_train, w)
        # 2) gradient step
        w = w - LR * grad

        # 3) log losses each epoch for train/validation
        train_loss = mse_loss(Xb_tr, y_train, w)
        val_loss = mse_loss(Xb_va, y_val, w)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 4) periodic progress print
        if (epoch + 1) % 100 == 0:
            print(
                f"epoch={epoch + 1:4d} "
                f"train_mse={train_loss:10.4f} "
                f"val_mse={val_loss:10.4f}"
            )

    # Save the learning-curve figure.
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Gradient Descent Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close()

    # Evaluate once on the held-out test split.
    y_test_pred = Xb_te @ w
    test_mse = float(np.mean((y_test_pred - y_test) ** 2))
    test_rmse = rmse(y_test, y_test_pred)
    test_mae = mae(y_test, y_test_pred)
    test_r2 = r2(y_test, y_test_pred)

    print("\nData summary")
    print(f"  rows: {len(df)}")
    print(f"  train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"  numeric features: {numeric_features}")
    print(f"  categorical features: {categorical_features}")

    print("\nTest metrics")
    print(f"  MSE : {test_mse:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE : {test_mae:.4f}")
    print(f"  R^2 : {test_r2:.4f}")
    print(f"\nSaved plot: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
