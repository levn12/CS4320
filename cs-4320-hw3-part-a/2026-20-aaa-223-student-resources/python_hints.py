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

    # 2) Separate target
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_val   = val_df[TARGET_COL].to_numpy(dtype=float)
    y_test  = test_df[TARGET_COL].to_numpy(dtype=float)

    # 3) Choose feature columns (drop target + other columns you believe should be excluded)
    X_train = train_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in train_df.columns])
    X_val   = val_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in val_df.columns])
    X_test  = test_df.drop(columns=[c for c in POSSIBLE_EXCLUDES if c in test_df.columns])

    # 4) Identify numeric vs categorical
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in numeric_cols]

    # 5) FIT imputation on TRAIN ONLY
    # TODO: compute numeric medians from X_train
    # TODO: compute categorical modes from X_train
    # Then apply to X_train / X_val / X_test using fillna()

    # 6) FIT scaling on TRAIN ONLY (numeric only)
    # TODO: compute mean/std from *imputed* X_train
    # TODO: apply (x - mean) / std to X_train / X_val / X_test

    # 7) FIT one-hot categories on TRAIN ONLY
    # TODO: build a list of categories per categorical column from X_train
    # TODO: create one-hot columns in a deterministic order
    # IMPORTANT: if val/test contains unseen categories, they should map to all-zeros

    # Final: produce numpy arrays
    # TODO: concatenate scaled numeric + one-hot categorical into final matrices
    # X_train_np = ...
    # X_val_np = ...
    # X_test_np = ...

    print("Train/Val/Test sizes:", len(train_df), len(val_df), len(test_df))
    print("Numeric cols:", numeric_cols)
    print("Categorical cols:", cat_cols)

    # print(X_train_np.shape, y_train.shape)


if __name__ == "__main__":
    main()
