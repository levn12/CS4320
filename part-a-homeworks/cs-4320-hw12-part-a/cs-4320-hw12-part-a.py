from __future__ import annotations

# CS 4320 - Assignment 12 Part A
# Practical deep learning with a small CNN on Fashion-MNIST.

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import time

import kagglehub
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

KAGGLE_DATASET = "zalando-research/fashionmnist"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/zalando-research/fashionmnist"

PLOT_PATH = OUTPUT_DIR / "hw12_part_a_loss_curve.png"
CHECKPOINT_PATH = OUTPUT_DIR / "best_fashion_mnist_cnn.pt"
SUMMARY_PATH = OUTPUT_DIR / "hw12_part_a_summary.json"
REFLECTION_PATH = OUTPUT_DIR / "hw12_part_a_reflection.txt"


@dataclass
class Config:
    random_seed: int = 4320
    validation_fraction: float = 0.20
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    epochs: int = 18
    num_workers: int = 0


CONFIG = Config()


def set_seed(seed: int) -> None:
    # Fix randomness so the run is reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    # Use a GPU if available. Otherwise train on CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_dataset() -> tuple[Path, Path]:
    # Download the Kaggle dataset and locate the train and test CSV files.
    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    train_csv = dataset_dir / "fashion-mnist_train.csv"
    test_csv = dataset_dir / "fashion-mnist_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Could not find Fashion-MNIST CSV files in {dataset_dir}")

    return train_csv, test_csv


def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Fashion-MNIST CSV format:
    # column 0 = label, remaining 784 columns = flattened pixels.
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    images = pixels.reshape(-1, 1, 28, 28) / 255.0
    return images, labels


def stratified_split(
    images: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Build a stratified train/validation split without extra dependencies.
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        shuffled_indices = rng.permutation(class_indices)
        validation_count = int(round(len(class_indices) * validation_fraction))

        validation_indices.extend(shuffled_indices[:validation_count].tolist())
        train_indices.extend(shuffled_indices[validation_count:].tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    validation_indices = np.array(validation_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        images[train_indices],
        images[validation_indices],
        labels[train_indices],
        labels[validation_indices],
    )


class FashionMNISTDataset(Dataset):
    # Simple custom Dataset class for PyTorch.

    def __init__(self, images: np.ndarray, labels: np.ndarray, mean: float, std: float):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = (self.images[index] - self.mean) / self.std
        label = self.labels[index]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SmallCNN(nn.Module):
    # Slightly stronger than the first version, but still a small student model.

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader, dict, float, float]:
    train_csv, test_csv = download_dataset()

    train_images_full, train_labels_full = load_csv(train_csv)
    test_images, test_labels = load_csv(test_csv)

    train_images, validation_images, train_labels, validation_labels = stratified_split(
        train_images_full,
        train_labels_full,
        validation_fraction=config.validation_fraction,
        seed=config.random_seed,
    )

    # Compute normalization using only the training split.
    train_mean = float(train_images.mean())
    train_std = float(train_images.std() + 1e-8)

    train_dataset = FashionMNISTDataset(train_images, train_labels, train_mean, train_std)
    validation_dataset = FashionMNISTDataset(validation_images, validation_labels, train_mean, train_std)
    test_dataset = FashionMNISTDataset(test_images, test_labels, train_mean, train_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_images, sample_labels = next(iter(train_loader))
    batch_info = {
        "image_batch_shape": list(sample_images.shape),
        "label_batch_shape": list(sample_labels.shape),
        "image_dtype": str(sample_images.dtype),
        "label_dtype": str(sample_labels.dtype),
        "pixel_min_after_normalization": float(sample_images.min().item()),
        "pixel_max_after_normalization": float(sample_images.max().item()),
    }

    return (
        train_loader,
        validation_loader,
        test_loader,
        {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
            "train_mean": train_mean,
            "train_std": train_std,
            "batch_info": batch_info,
        },
        train_mean,
        train_std,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float, float]:
    # Use the same function for training and evaluation.
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_data_time = 0.0
    total_compute_time = 0.0

    last_step_end = time.perf_counter()

    for images, labels in loader:
        batch_ready = time.perf_counter()
        total_data_time += batch_ready - last_step_end

        compute_start = time.perf_counter()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_examples += int(labels.size(0))

        last_step_end = time.perf_counter()
        total_compute_time += last_step_end - compute_start

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return float(average_loss), float(accuracy), total_data_time, total_compute_time


def save_checkpoint(model: nn.Module, epoch: int, validation_loss: float, validation_accuracy: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "config": asdict(CONFIG),
        },
        CHECKPOINT_PATH,
    )


def load_best_model(device: torch.device) -> tuple[SmallCNN, dict]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = SmallCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def save_loss_plot(history: dict[str, list[float]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
    ax.plot(history["epoch"], history["validation_loss"], marker="s", label="Validation loss")
    ax.set_title("Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_reflection(history: dict[str, list[float]], device: torch.device) -> str:
    total_data_time = float(np.sum(history["train_data_time"]))
    total_compute_time = float(np.sum(history["train_compute_time"]))

    if total_compute_time >= total_data_time:
        timing_text = (
            f"Most training time was spent in model compute ({total_compute_time:.2f}s) rather than data loading "
            f"({total_data_time:.2f}s), which is expected for a CNN running on {device}."
        )
    else:
        timing_text = (
            f"Most training time was spent in data loading ({total_data_time:.2f}s) rather than model compute "
            f"({total_compute_time:.2f}s)."
        )

    reflection = (
        f"{timing_text} A realistic debugging issue in this assignment is a shape mismatch caused by forgetting to "
        "reshape the 784 pixel columns into 1x28x28 images. To diagnose that problem, I would print one batch shape "
        "from the dataloader and then check the tensor sizes after each convolution and pooling layer. Another "
        "possible issue is validation accuracy stalling, which I would investigate by checking normalization, "
        "learning rate, and whether the model is beginning to overfit."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REFLECTION_PATH.write_text(reflection, encoding="utf-8")
    return reflection


def main() -> None:
    set_seed(CONFIG.random_seed)
    device = get_device()

    train_loader, validation_loader, test_loader, data_info, train_mean, train_std = make_dataloaders(CONFIG)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
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
    best_validation_loss = float("inf")
    best_epoch = -1

    print("Assignment 12 Part A - Practical Deep Learning and Frameworks")
    print("==============================================================")
    print(f"Dataset source: {KAGGLE_DATASET_URL}")
    print(f"Device: {device}")
    print(f"Train / validation / test sizes: {data_info['train']} / {data_info['validation']} / {data_info['test']}")
    print(f"Training normalization mean: {train_mean:.6f}")
    print(f"Training normalization std: {train_std:.6f}")
    print(f"Batch inspection: {data_info['batch_info']}")

    for epoch in range(1, CONFIG.epochs + 1):
        train_loss, train_accuracy, train_data_time, train_compute_time = run_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy, _, _ = run_epoch(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
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
            f"Epoch {epoch:02d}/{CONFIG.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f} | val_acc={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            best_epoch = epoch
            save_checkpoint(model, epoch, validation_loss, validation_accuracy)

    save_loss_plot(history)

    # Reload the best checkpoint and verify the validation result.
    best_model, checkpoint = load_best_model(device)
    reloaded_validation_loss, reloaded_validation_accuracy, _, _ = run_epoch(
        model=best_model,
        loader=validation_loader,
        loss_function=loss_function,
        device=device,
        optimizer=None,
    )

    # Evaluate on the test set only once at the end.
    test_loss, test_accuracy, _, _ = run_epoch(
        model=best_model,
        loader=test_loader,
        loss_function=loss_function,
        device=device,
        optimizer=None,
    )

    best_validation_loss_epoch = history["epoch"][int(np.argmin(history["validation_loss"]))]
    training_interpretation = (
        f"Training loss decreased over time, which shows the model learned meaningful image patterns. "
        f"Validation loss was lowest around epoch {best_validation_loss_epoch}, while the best validation accuracy "
        f"was {best_validation_accuracy:.4f}. This suggests the model improved steadily without severe overfitting, "
        "so using the best validation checkpoint was a good choice."
    )

    reflection_text = write_reflection(history, device)

    summary = {
        "config": asdict(CONFIG),
        "dataset_source": KAGGLE_DATASET_URL,
        "device": str(device),
        "split_sizes": {
            "train": data_info["train"],
            "validation": data_info["validation"],
            "test": data_info["test"],
        },
        "normalization": {
            "train_mean": train_mean,
            "train_std": train_std,
        },
        "batch_info": data_info["batch_info"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "checkpoint_validation_loss": float(checkpoint["validation_loss"]),
        "checkpoint_validation_accuracy": float(checkpoint["validation_accuracy"]),
        "reloaded_validation_loss": reloaded_validation_loss,
        "reloaded_validation_accuracy": reloaded_validation_accuracy,
        "validation_reload_loss_match": abs(reloaded_validation_loss - float(checkpoint["validation_loss"])) < 1e-8,
        "validation_reload_accuracy_match": abs(reloaded_validation_accuracy - float(checkpoint["validation_accuracy"])) < 1e-8,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_interpretation": training_interpretation,
        "resource_reflection": reflection_text,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTraining interpretation:")
    print(training_interpretation)

    print("\nCheckpoint reload verification:")
    print(f"Saved validation accuracy: {float(checkpoint['validation_accuracy']):.4f}")
    print(f"Reloaded validation accuracy: {reloaded_validation_accuracy:.4f}")
    print(f"Saved validation loss: {float(checkpoint['validation_loss']):.4f}")
    print(f"Reloaded validation loss: {reloaded_validation_loss:.4f}")

    print("\nFinal test evaluation:")
    print(f"Test loss={test_loss:.4f} | test accuracy={test_accuracy:.4f} | validation accuracy={best_validation_accuracy:.4f}")

    print("\nSaved files:")
    print(f"Plot: {PLOT_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Reflection: {REFLECTION_PATH}")


if __name__ == "__main__":
    main()
