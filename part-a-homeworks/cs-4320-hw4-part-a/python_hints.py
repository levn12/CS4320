#!/usr/bin/env python3
"""
CS 4320 — Assignment 4 (Part A) — Python Hints (Student-Facing)

This file is deliberately *incomplete*. It shows:
  - Where to put code
  - What functions / objects to look up
  - Small "shape checks" and sanity tips
But it does NOT provide a working end-to-end implementation.

Goal: help you write your own code in your assignment file / notebook.

How to use:
  - Read a section
  - Copy the *pattern* (not the whole block)
  - Replace TODOs with your own code
"""

from __future__ import annotations

# You will likely use these imports (add more if you need them):
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn pieces commonly used in this assignment
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ============================================================================
# 0) Quick glossary (what is a "pipeline" in this assignment?)
# ============================================================================
"""
Pipeline idea:
  raw DataFrame  -->  ColumnTransformer (num + cat preprocessing)  -->  NumPy array
                                                        |
                                                        v
                                            (your NumPy gradient descent)
"""

# ============================================================================
# 1) Loading data (pattern only)
# ============================================================================
"""
df = pd.read_csv("your_file.csv")

# TODO: choose your target column name
y = df["TODO_TARGET_COL"].to_numpy(dtype=np.float64)

# Everything else is features
X_df = df.drop(columns=["TODO_TARGET_COL"])
"""

# Tip: Print a quick summary
"""
print(df.head())
print(df.dtypes)
"""

# ============================================================================
# 2) Train / Validation / Test split (pattern, not full code)
# ============================================================================
"""
Common pattern:
  1) split off TEST first
  2) split remaining into TRAIN and VAL

# TODO: pick fractions
test_size = 0.15
val_size  = 0.15
random_state = 4320

# 1) split test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_df, y,
    test_size=test_size,
    random_state=random_state,
)

# 2) compute val fraction relative to trainval
val_fraction = val_size / (1.0 - test_size)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_fraction,
    random_state=random_state,
)

print("sizes:", len(X_train), len(X_val), len(X_test))
"""

# ============================================================================
# 3) Detect numeric vs categorical features (hint)
# ============================================================================
"""
DO NOT assume only dtype == 'object' is categorical.
Use pandas' numeric dtype checks.

from pandas.api.types import is_numeric_dtype

numeric_features = [c for c in X_df.columns if is_numeric_dtype(X_df[c])]
categorical_features = [c for c in X_df.columns if c not in numeric_features]

print("numeric:", numeric_features)
print("categorical:", categorical_features)
"""

# ============================================================================
# 4) Build preprocessing (Pipeline + ColumnTransformer)
# ============================================================================
"""
Numeric pipeline often includes:
  - SimpleImputer(strategy="median")
  - StandardScaler()

Categorical pipeline often includes:
  - SimpleImputer(strategy="most_frequent")
  - OneHotEncoder(handle_unknown="ignore", ...)

NOTE: OneHotEncoder changed argument names across sklearn versions.
If you get an error about sparse/sparse_output, check your sklearn version.

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ],
    remainder="drop",
)

# IMPORTANT: fit ONLY on training data
X_train_p = pre.fit_transform(X_train)
X_val_p   = pre.transform(X_val)
X_test_p  = pre.transform(X_test)

# convert to NumPy float arrays
X_train_p = np.asarray(X_train_p, dtype=np.float64)
X_val_p   = np.asarray(X_val_p, dtype=np.float64)
X_test_p  = np.asarray(X_test_p, dtype=np.float64)

print("X_train_p shape:", X_train_p.shape)
"""

# ============================================================================
# 5) Add a bias term (intercept)
# ============================================================================
"""
If X_train_p is shape (n, d), you can create Xb shape (n, d+1)
with a first column of ones.

# TODO: implement add_bias_column(X)
# - input: X with shape (n, d)
# - output: Xb with shape (n, d+1), first col all ones

# Shape sanity:
#   Xb.shape[0] == X.shape[0]
#   Xb.shape[1] == X.shape[1] + 1
"""

# ============================================================================
# 6) Prediction function (linear regression)
# ============================================================================
"""
With bias-augmented Xb and weights w:

y_hat = Xb @ w

Where:
  Xb is shape (n, d+1)
  w  is shape (d+1,)

# TODO: implement predict(X, w) that:
#   - adds bias column internally
#   - returns y_hat (n,)
"""

# ============================================================================
# 7) MSE loss + gradient (equations)
# ============================================================================
"""
Mean Squared Error (MSE):

  L(w) = (1/n) * sum_i (y_hat_i - y_i)^2
       = (1/n) * ||Xb w - y||^2

Gradient of MSE:

  ∇L(w) = (2/n) * Xb^T (Xb w - y)

Hints:
- Use vectorized NumPy (no loops over rows).
- Keep y as shape (n,) not (n,1) unless you are consistent.

# TODO: implement:
#   - mse_loss(Xb, y, w) -> float
#   - mse_grad(Xb, y, w) -> np.ndarray shape (d+1,)
"""

# ============================================================================
# 8) Batch gradient descent loop (structure only)
# ============================================================================
"""
Typical loop:

# Prepare bias-augmented matrices once
Xb_tr = add_bias_column(X_train_p)
Xb_va = add_bias_column(X_val_p)

# Initialize weights (small random values or zeros)
w = np.zeros(Xb_tr.shape[1], dtype=np.float64)
# OR: rng = np.random.default_rng(4320); w = rng.normal(0, 0.01, size=Xb_tr.shape[1])

train_losses = []
val_losses = []

for epoch in range(epochs):
    # 1) compute gradient on training
    grad = mse_grad(Xb_tr, y_train, w)

    # 2) update
    w = w - lr * grad

    # 3) track losses (OPTIONAL but recommended)
    train_losses.append(mse_loss(Xb_tr, y_train, w))
    val_losses.append(mse_loss(Xb_va, y_val, w))

    # 4) occasionally print progress
    # if (epoch+1) % 100 == 0: print(epoch+1, train_losses[-1], val_losses[-1])

Common debugging tips:
- If loss explodes (goes to inf/very large): reduce lr by 10x.
- If loss barely changes: increase lr slightly or run more epochs.
- If train and val are identical: check for leakage (fit_transform on val/test is wrong).
"""

# ============================================================================
# 9) Plotting a loss curve (pattern only)
# ============================================================================
"""
plt.figure()
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)
plt.close()
"""

# ============================================================================
# 10) Optional: evaluation metrics (what to compute)
# ============================================================================
"""
Common regression metrics:
  - MSE: mean((y_hat - y)^2)
  - RMSE: sqrt(MSE)
  - MAE: mean(abs(y_hat - y))
  - R^2: 1 - SS_res / SS_tot

You may use sklearn.metrics OR implement with NumPy.

If you implement R^2:
  SS_res = sum((y - y_hat)^2)
  SS_tot = sum((y - mean(y))^2)
"""

# ============================================================================
# 11) Checklist before you submit
# ============================================================================
"""
[ ] I did NOT fit preprocessing on validation or test data.
[ ] I confirmed my transformed X arrays are numeric and have expected shapes.
[ ] My MSE decreases over training for a reasonable lr.
[ ] I saved a loss curve plot.
[ ] I evaluated on the TEST set only at the end.
"""
