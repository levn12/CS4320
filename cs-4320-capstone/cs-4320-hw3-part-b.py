"""
CS 4320 — Assignment 3 (Part B)

Workflow:
1) Load data
2) Split into train/val/test with seed for reproducibility
3) Separate target y from features X
4) Fit preprocessing on TRAIN ONLY:
   - numeric mean
   - scaling mean/std
5) Apply those artifacts to val/test
"""

import numpy as np
import pandas as pd

CSV_PATH = r"C:\Users\levid\School_Programming\CS4320\cs-4320-capstone\electrical_fault_data.csv"
SEED = 4320  # Same seed as part A for convenience.

# Target: Detect electrical fault in wires G, C, B, A and which columns fault occurred in
TARGET_COLS = ["G", "C", "B", "A"]  # Columns representing fault detection in each wire


def split_indices(n: int, seed: int, train_frac: float = 0.70, val_frac: float = 0.15):
    """Deterministic split using a seeded permutation (same idea as lecture)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main():
    df = pd.read_csv(CSV_PATH)

    # Check data for missing values and print result to make sure nothing is missing.
    missing_counts = df.isnull().sum()
    print("Missing values per column:")
    print(missing_counts)

    # 1) Split - create indices for copying from original data
    train_idx, val_idx, test_idx = split_indices(len(df), SEED)

    # Create train/val/test splits by copying from original data using the indices.
    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 2) Separate targets
    y_train = train_df[TARGET_COLS].to_numpy(dtype=float)
    y_val   = val_df[TARGET_COLS].to_numpy(dtype=float)
    y_test  = test_df[TARGET_COLS].to_numpy(dtype=float)

    # 3) Choose feature columns (drop targets)
    X_train = train_df.drop(columns=TARGET_COLS)
    X_val   = val_df.drop(columns=TARGET_COLS)
    X_test  = test_df.drop(columns=TARGET_COLS)

    # 4) FIT scaling on TRAIN ONLY
    # Calculate the mean and std for numeric columns from X_train.
    X_num_means = X_train.mean()
    X_num_stds = X_train.std()

    # Then apply to X_train / X_val / X_test using the formula: (X - mean) / std
    X_train = (X_train - X_num_means) / (X_num_stds)
    X_val = (X_val - X_num_means) / (X_num_stds)
    X_test = (X_test - X_num_means) / (X_num_stds)

    # No imputation is needed since there are no missing values.

    # Print shapes and other info to double check before moving on to modeling.
    print("X shapes:", X_train.shape, X_val.shape, X_test.shape)
    print("y shapes:", y_train.shape, y_val.shape, y_test.shape)

    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    print("Numeric cols:", X_train.columns.tolist())


if __name__ == "__main__":
    main()
