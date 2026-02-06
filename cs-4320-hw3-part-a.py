#!/usr/bin/env python3
"""
CS 4320 — Assignment 3 (Part A) HINTS
Ames Housing (curated) — starter + hints script (NO scikit-learn)

This file is designed to help you get unstuck without giving away the full solution.
You are still responsible for implementing the required steps and writing up what you did.

Rules reminder:
- You MAY use numpy/pandas for array/data operations.
- You may NOT use scikit-learn to do splitting, imputation, scaling, or encoding.

Recommended workflow (leakage-safe):
1) Load data
2) Split into train/val/test with the required seed
3) Separate target y from features X
4) Fit preprocessing on TRAIN ONLY:
   - numeric median
   - categorical mode
   - scaling mean/std
   - one-hot categories
5) Apply those artifacts to val/test
"""

import numpy as np
import pandas as pd

CSV_PATH = "ames_curated.csv"
SEED = 4320  # required seed (so everyone gets the same split)

TARGET_COL = "saleprice"

# You should decide which columns are safe/appropriate to use as model inputs.
# (Hint: identifiers are usually not appropriate.)

# Define columns to be excluded in my preproccessing.
POSSIBLE_EXCLUDES = ["pid", TARGET_COL]


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

    # 1) Split
    train_idx, val_idx, test_idx = split_indices(len(df), SEED)

    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # 2) Separate target -- use y to remind us it's an output
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_val   = val_df[TARGET_COL].to_numpy(dtype=float)
    y_test  = test_df[TARGET_COL].to_numpy(dtype=float)

    # 3) Choose feature columns (drop target + other columns you believe should be excluded)
    # Use x to remind us it's an input
    X_train = train_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in train_df.columns])
    X_val   = val_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in val_df.columns])
    X_test  = test_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in test_df.columns])

    # 4) Identify numeric vs categorical
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    # 5) FIT imputation on TRAIN ONLY

    # Calculates the median for numeric colums and the mode for categorical columns from X_train.
    # When there's a tie for the mode, pandas returns multiple values, so we take the first one with .iloc[0].
    X_num_medians = X_train[numeric_cols].median()
    X_cat_modes = X_train[cat_cols].mode().iloc[0]

    # Then apply to X_train / X_val / X_test using fillna()

    # Using the medians/modes from X_train, we fill in the missing values in X_train, X_val, and X_test.
    # For numeric columns, we use the medians, and for categorical columns, we use the modes.
    X_train = X_train.fillna(value=X_num_medians)
    X_train[cat_cols] = X_train[cat_cols].fillna(value=X_cat_modes)
    X_val = X_val.fillna(value=X_num_medians)
    X_val[cat_cols] = X_val[cat_cols].fillna(value=X_cat_modes)
    X_test = X_test.fillna(value=X_num_medians)
    X_test[cat_cols] = X_test[cat_cols].fillna(value=X_cat_modes)

    # 6) FIT scaling on TRAIN ONLY (numeric only)

    # Calculate the mean and std for numeric columns from X_train.
    X_num_means = X_train[numeric_cols].mean()
    X_num_stds = X_train[numeric_cols].std()

    # Then apply to X_train / X_val / X_test using the formula: (X - mean) / std
    X_train[numeric_cols] = (X_train[numeric_cols] - X_num_means) / (X_num_stds)
    X_val[numeric_cols] = (X_val[numeric_cols] - X_num_means) / (X_num_stds)
    X_test[numeric_cols] = (X_test[numeric_cols] - X_num_means) / (X_num_stds)

    # 7) FIT one-hot categories on TRAIN ONLY
    # Build a list of categories from X_train.
    cat_list = []

    # For each column (which we made a list of above), add the unique categories from X_train to cat_list.
    # For each categorical column, create new one-hot columns for each category.
    # We ensure that the order of the categories is deterministic by using the unique values from X_train.
    # IMPORTANT: if val/test contains unseen categories, they should map to all-zeros
    for col in cat_cols:
        categories = X_train[col].unique()
        cat_list.append(categories)
        for category in categories:
            X_train[f"{col}_{category}"] = (X_train[col] == category).astype(int)
            X_val[f"{col}_{category}"] = (X_val[col] == category).astype(int)
            X_test[f"{col}_{category}"] = (X_test[col] == category).astype(int)

    # Drop original categorical columns now that we've expanded them to one-hot.
    # This keeps the final feature matrices numeric-only and avoids duplicate info.
    X_train.drop(columns=cat_cols, inplace=True)
    X_val.drop(columns=cat_cols, inplace=True)
    X_test.drop(columns=cat_cols, inplace=True)

    # Final: produce numpy arrays
    # Concatenate scaled numeric + one-hot categorical is already done in X_train/X_val/X_test.
    
    X_train_np = X_train.to_numpy(dtype=float)
    X_val_np = X_val.to_numpy(dtype=float)
    X_test_np = X_test.to_numpy(dtype=float)

    # Print shapes and other info to double check before moving on to modeling.

    print("X shapes:", X_train_np.shape, X_val_np.shape, X_test_np.shape)
    print("y shapes:", y_train.shape, y_val.shape, y_test.shape)

    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", cat_cols)
    print(X_train.head())

    # print(X_train_np.shape, y_train.shape)


if __name__ == "__main__":
    main()
