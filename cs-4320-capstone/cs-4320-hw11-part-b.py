"""
CS 4320 - Assignment 11 Part B

Capstone adaptation of the neural-network workflow:
1. Load the electrical fault dataset.
2. Reuse one train/validation/test split for the whole assignment.
3. Preprocess the six measurement columns with median imputation + scaling.
4. Train a simple MLP for binary fault detection (fault vs no fault).
5. Train a simple MLP for multiclass fault-pattern detection.
6. Report held-out test metrics for both tasks.

The script stays intentionally compact because this is a homework assignment,
but it still follows the same overall structure as Part A.
"""

from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# The electrical dataset stores fault-indicator bits and six measured signals.
FAULT_COLS = ["G", "C", "B", "A"]
MEASUREMENT_COLS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# File locations for this capstone homework folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electrical_fault_data.csv"

# Reproducible split/training settings.
RANDOM_STATE = 4320
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 32
BINARY_EPOCHS = 80
MULTICLASS_EPOCHS = 100
PATIENCE = 10


def compute_binary_metrics(y_true, probabilities, threshold=0.5):
    """Convert probabilities into class predictions and compute standard binary metrics."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions),
    }


def compute_multiclass_metrics(y_true, predictions):
    """Compute simple multiclass metrics using macro averaging for balance across classes."""
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions),
    }


def train_network(X_train, y_train, X_val, y_val, output_dim, epochs):
    """Train one small MLP with early stopping on validation loss."""
    # Use one hidden layer to keep the assignment simple and readable.
    if output_dim == 1:
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        criterion = nn.BCEWithLogitsLoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    else:
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, output_dim),
        )
        criterion = nn.CrossEntropyLoss()
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Mini-batches help the optimizer converge smoothly without adding much code.
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)

            if output_dim == 1:
                loss = criterion(logits.squeeze(1), batch_y)
            else:
                loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        # Evaluate the full train and validation sets at the end of each epoch.
        model.eval()
        with torch.no_grad():
            train_logits = model(torch.tensor(X_train, dtype=torch.float32))
            val_logits = model(torch.tensor(X_val, dtype=torch.float32))

            if output_dim == 1:
                train_loss = criterion(train_logits.squeeze(1), y_train_tensor).item()
                val_loss = criterion(val_logits.squeeze(1), y_val_tensor).item()
            else:
                train_loss = criterion(train_logits, y_train_tensor).item()
                val_loss = criterion(val_logits, y_val_tensor).item()

        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(float(val_loss))

        # Keep the best validation-loss checkpoint and stop early once improvement stalls.
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, history


def main():
    # Seed every library used here so results stay reproducible.
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # Load the capstone electrical-fault dataset.
    df = pd.read_csv(DATA_PATH)

    # Use only the six measured electrical signals as input features.
    X = df[MEASUREMENT_COLS]

    # Build the full multiclass target by combining the four fault bits into one string label.
    fault_pattern = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1)
    class_names = sorted(fault_pattern.unique().tolist())
    class_to_index = {label: index for index, label in enumerate(class_names)}
    y_multiclass = fault_pattern.map(class_to_index).to_numpy()

    # Build the simpler binary target: any fault at all versus no fault.
    y_binary = (fault_pattern != "0000").astype(int).to_numpy()

    # Make one shared row split, stratified by multiclass labels so every task gets balanced classes.
    row_indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_multiclass,
    )

    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_multiclass[train_val_idx],
    )

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]

    # Fit preprocessing on train only, then reuse it for both tasks.
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
                MEASUREMENT_COLS,
            )
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    # Slice both targets using the same shared train/validation/test indices.
    y_binary_train = y_binary[train_idx]
    y_binary_val = y_binary[val_idx]
    y_binary_test = y_binary[test_idx]

    y_multi_train = y_multiclass[train_idx]
    y_multi_val = y_multiclass[val_idx]
    y_multi_test = y_multiclass[test_idx]

    # Train the simple binary MLP first.
    binary_model, binary_history = train_network(
        X_train_processed,
        y_binary_train,
        X_val_processed,
        y_binary_val,
        output_dim=1,
        epochs=BINARY_EPOCHS,
    )

    binary_model.eval()
    with torch.no_grad():
        binary_test_logits = binary_model(torch.tensor(X_test_processed, dtype=torch.float32)).squeeze(1)
        binary_test_probabilities = torch.sigmoid(binary_test_logits).cpu().numpy()

    binary_metrics = compute_binary_metrics(y_binary_test, binary_test_probabilities)

    # Then train the multiclass MLP on the same processed features.
    multiclass_model, multiclass_history = train_network(
        X_train_processed,
        y_multi_train,
        X_val_processed,
        y_multi_val,
        output_dim=len(class_names),
        epochs=MULTICLASS_EPOCHS,
    )

    multiclass_model.eval()
    with torch.no_grad():
        multiclass_test_logits = multiclass_model(torch.tensor(X_test_processed, dtype=torch.float32))
        multiclass_test_predictions = torch.argmax(multiclass_test_logits, dim=1).cpu().numpy()

    multiclass_metrics = compute_multiclass_metrics(y_multi_test, multiclass_test_predictions)

    # Build a readable confusion matrix for the multiclass task.
    multiclass_confusion_df = pd.DataFrame(
        multiclass_metrics["confusion_matrix"],
        index=[f"true_{label}" for label in class_names],
        columns=[f"pred_{label}" for label in class_names],
    )

    # Also show how many examples belong to each multiclass label.
    class_counts = fault_pattern.value_counts().sort_index()

    print("Assignment 11 Part B - Capstone Neural Networks")
    print("===============================================")
    print(f"Rows: {len(df)}")
    print(f"Features: {MEASUREMENT_COLS}")
    print(f"Fault patterns: {class_names}")
    print(f"Train/validation/test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print("\nFault-pattern counts:")
    print(class_counts.to_string())

    print("\nSimple fault detection (binary: any fault vs no fault)")
    print("------------------------------------------------------")
    print(f"Binary epochs run: {len(binary_history['train_loss'])}")
    print(f"Test accuracy : {binary_metrics['accuracy']:.4f}")
    print(f"Test precision: {binary_metrics['precision']:.4f}")
    print(f"Test recall   : {binary_metrics['recall']:.4f}")
    print(f"Test f1       : {binary_metrics['f1']:.4f}")
    print(f"Test roc_auc  : {binary_metrics['roc_auc']:.4f}")
    print("Confusion matrix [[tn, fp], [fn, tp]]:")
    print(binary_metrics["confusion_matrix"])

    print("\nMulti-class fault detection (six fault patterns)")
    print("------------------------------------------------")
    print(f"Multiclass epochs run: {len(multiclass_history['train_loss'])}")
    print(f"Test accuracy       : {multiclass_metrics['accuracy']:.4f}")
    print(f"Macro precision     : {multiclass_metrics['precision_macro']:.4f}")
    print(f"Macro recall        : {multiclass_metrics['recall_macro']:.4f}")
    print(f"Macro f1            : {multiclass_metrics['f1_macro']:.4f}")
    print("\nMulticlass confusion matrix:")
    print(multiclass_confusion_df.to_string())


if __name__ == "__main__":
    main()
