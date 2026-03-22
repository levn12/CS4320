from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse

# --------------------------------------------------
# 1. Set dataset directory
# --------------------------------------------------

DATA_DIR = Path(".")  # adjust if needed

# --------------------------------------------------
# 2. Load raw messages (for error analysis)
# --------------------------------------------------

messages = pd.read_csv(DATA_DIR / "messages.csv")

print("Messages shape:", messages.shape)
print(messages.head())

# --------------------------------------------------
# 3. Load feature matrices (sparse!)
# --------------------------------------------------

X_counts = sparse.load_npz(DATA_DIR / "X_counts.npz")
X_tfidf  = sparse.load_npz(DATA_DIR / "X_tfidf.npz")

print("Counts matrix shape:", X_counts.shape)
print("TF-IDF matrix shape:", X_tfidf.shape)

# --------------------------------------------------
# 4. Load labels
# --------------------------------------------------

y = messages["label"].to_numpy(dtype=int)

print("Label distribution:", np.bincount(y))

# --------------------------------------------------
# 5. Load fixed splits
# --------------------------------------------------

with open(DATA_DIR / "split.json", "r") as f:
    split = json.load(f)

train_ids = np.array(split["train_ids"], dtype=int)
val_ids   = np.array(split["val_ids"], dtype=int)
test_ids  = np.array(split["test_ids"], dtype=int)

print("Train/Val/Test sizes:",
      len(train_ids), len(val_ids), len(test_ids))

# --------------------------------------------------
# 6. Create split datasets
# --------------------------------------------------

# Choose which representation to use:
X = X_counts       # for MultinomialNB
# X = X_tfidf      # for kNN (recommended)

X_train = X[train_ids]
X_val   = X[val_ids]
X_test  = X[test_ids]

y_train = y[train_ids]
y_val   = y[val_ids]
y_test  = y[test_ids]

print("Training feature matrix shape:", X_train.shape)
