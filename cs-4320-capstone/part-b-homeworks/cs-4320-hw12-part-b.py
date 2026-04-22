"""
CS 4320 - Assignment 12 Part B

Capstone adaptation of this week's deep-learning workflow:
1. Load the electrical fault dataset.
2. Build one multiclass target from the four fault-indicator columns.
3. Use only the six measured electrical signals as model inputs.
4. Make one reproducible train / validation / test split.
5. Train a small neural network for multiclass fault prediction.
6. Keep the best validation checkpoint in memory.
7. Evaluate the best model one time on the test set.
"""

from pathlib import Path
import copy
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"
OUTPUT_DIR = BASE_DIR / "hw12_part_b_outputs"
PLOT_PATH = OUTPUT_DIR / "hw12_part_b_training_curve.png"

FAULT_BIT_COLS = ["G", "C", "B", "A"]
FEATURE_COLS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

RANDOM_STATE = 4320
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
PATIENCE = 12


def set_seed(seed: int) -> None:
    # Fix random seeds so the split and training run are reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_multiclass_target(df: pd.DataFrame):
    # Combine the four fault-indicator bits into one text label like "0111".
    fault_pattern = df[FAULT_BIT_COLS].astype(int).astype(str).agg("".join, axis=1)

    # Turn text labels into integer class IDs that PyTorch can train on.
    class_names = sorted(fault_pattern.unique().tolist())
    class_to_index = {label: index for index, label in enumerate(class_names)}
    y = fault_pattern.map(class_to_index).to_numpy(dtype=np.int64)

    return fault_pattern, class_names, class_to_index, y


def stratified_split_indices(
    y: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
):
    # Make train / validation / test splits while preserving class balance.
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    test_indices = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        shuffled_indices = rng.permutation(class_indices)

        class_count = len(class_indices)
        train_count = int(round(class_count * train_fraction))
        validation_count = int(round(class_count * validation_fraction))

        train_indices.extend(shuffled_indices[:train_count].tolist())
        validation_indices.extend(shuffled_indices[train_count : train_count + validation_count].tolist())
        test_indices.extend(shuffled_indices[train_count + validation_count :].tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    validation_indices = np.array(validation_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)

    return train_indices, validation_indices, test_indices


class FaultDataset(Dataset):
    # Small Dataset class that returns one feature row and one class label.

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        features = torch.tensor(self.X[index], dtype=torch.float32)
        label = torch.tensor(self.y[index], dtype=torch.long)
        return features, label


class FaultMLP(nn.Module):
    # A small MLP is a reasonable neural model for this tabular problem.

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.10),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def run_epoch(model, loader, loss_function, optimizer=None):
    # Use one shared function for training and evaluation.
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_data_time = 0.0
    total_compute_time = 0.0

    last_step_end = time.perf_counter()

    for batch_X, batch_y in loader:
        batch_ready = time.perf_counter()
        total_data_time += batch_ready - last_step_end

        compute_start = time.perf_counter()

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

        last_step_end = time.perf_counter()
        total_compute_time += last_step_end - compute_start

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy, total_data_time, total_compute_time


def predict_classes(model, loader):
    # Collect predictions for confusion matrix and per-class analysis.
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            logits = model(batch_X)
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.append(batch_predictions.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    return np.concatenate(targets), np.concatenate(predictions)


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    # Build a confusion matrix without needing extra packages.
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[true_value, pred_value] += 1
    return matrix


def macro_metrics_from_confusion_matrix(confusion: np.ndarray):
    # Compute macro precision, recall, and f1 from the confusion matrix.
    precisions = []
    recalls = []
    f1_scores = []

    for class_index in range(confusion.shape[0]):
        true_positive = confusion[class_index, class_index]
        false_positive = confusion[:, class_index].sum() - true_positive
        false_negative = confusion[class_index, :].sum() - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1_scores)),
    }


def save_training_plot(history):
    # Save one simple plot of training vs validation loss.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
    ax.plot(history["epoch"], history["validation_loss"], marker="s", label="Validation loss")
    ax.set_title("HW12 Part B - MLP Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(RANDOM_STATE)

    # Load the electrical fault dataset.
    df = pd.read_csv(DATA_PATH)

    # Build the multiclass target from the four fault-bit columns.
    fault_pattern, class_names, class_to_index, y = build_multiclass_target(df)

    # Use only measured electrical signals as inputs.
    # We do not use G/C/B/A as features because those columns define the target labels.
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)

    # Create one reproducible train / validation / test split.
    train_idx, validation_idx, test_idx = stratified_split_indices(
        y=y,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_STATE,
    )

    X_train = X[train_idx]
    X_validation = X[validation_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_validation = y[validation_idx]
    y_test = y[test_idx]

    # Normalize features using train-only statistics.
    train_mean = X_train.mean(axis=0, keepdims=True)
    train_std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - train_mean) / train_std
    X_validation = (X_validation - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    # Build PyTorch datasets and dataloaders.
    train_dataset = FaultDataset(X_train, y_train)
    validation_dataset = FaultDataset(X_validation, y_validation)
    test_dataset = FaultDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Show one batch so the data pipeline is explicit.
    example_features, example_labels = next(iter(train_loader))
    batch_info = {
        "feature_batch_shape": list(example_features.shape),
        "label_batch_shape": list(example_labels.shape),
        "feature_dtype": str(example_features.dtype),
        "label_dtype": str(example_labels.dtype),
    }

    # Create and train the multiclass MLP.
    model = FaultMLP(input_dim=len(FEATURE_COLS), output_dim=len(class_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.CrossEntropyLoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "train_data_time": [],
        "train_compute_time": [],
    }

    best_validation_accuracy = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy, train_data_time, train_compute_time = run_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy, _, _ = run_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            optimizer=None,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)
        history["train_data_time"].append(train_data_time)
        history["train_compute_time"].append(train_compute_time)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.4f}"
        )

        # Keep the best validation checkpoint in memory.
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    # Reload the best validation checkpoint before final test evaluation.
    model.load_state_dict(best_state)

    test_loss, test_accuracy, test_data_time, test_compute_time = run_epoch(
        model=model,
        loader=test_loader,
        loss_function=loss_function,
        optimizer=None,
    )

    y_test_true, y_test_pred = predict_classes(model, test_loader)
    confusion = confusion_matrix_numpy(y_test_true, y_test_pred, num_classes=len(class_names))
    macro_metrics = macro_metrics_from_confusion_matrix(confusion)

    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )

    class_counts = fault_pattern.value_counts().sort_index()
    per_class_accuracy = {}
    for class_index, class_name in enumerate(class_names):
        row_total = confusion[class_index].sum()
        per_class_accuracy[class_name] = float(confusion[class_index, class_index] / row_total) if row_total > 0 else 0.0

    save_training_plot(history)

    print("\nAssignment 12 Part B - Capstone Deep Learning")
    print("=============================================")
    print(f"Rows: {len(df)}")
    print(f"Feature columns: {FEATURE_COLS}")
    print(f"Class names: {class_names}")
    print(f"Class mapping: {class_to_index}")
    print(f"Train / validation / test sizes: {len(train_dataset)} / {len(validation_dataset)} / {len(test_dataset)}")
    print(f"Batch inspection: {batch_info}")
    print("\nFault-pattern counts:")
    print(class_counts.to_string())

    print("\nBest validation checkpoint:")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")

    print("\nFinal test evaluation:")
    print(f"Test loss          : {test_loss:.4f}")
    print(f"Test accuracy      : {test_accuracy:.4f}")
    print(f"Macro precision    : {macro_metrics['precision_macro']:.4f}")
    print(f"Macro recall       : {macro_metrics['recall_macro']:.4f}")
    print(f"Macro f1           : {macro_metrics['f1_macro']:.4f}")

    print("\nPer-class test accuracy:")
    for class_name, accuracy in per_class_accuracy.items():
        print(f"  {class_name}: {accuracy:.4f}")

    print("\nConfusion matrix:")
    print(confusion_df.to_string())

    print("\nResource notes:")
    print(f"Total training data-loading time : {sum(history['train_data_time']):.2f} seconds")
    print(f"Total training model-compute time: {sum(history['train_compute_time']):.2f} seconds")
    print(f"Test data-loading time           : {test_data_time:.2f} seconds")
    print(f"Test model-compute time          : {test_compute_time:.2f} seconds")
    print(f"Saved training curve             : {PLOT_PATH}")


if __name__ == "__main__":
    main()
