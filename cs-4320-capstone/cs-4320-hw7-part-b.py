# CS 4320 - Assignment 7 (Part B)
# Comparing model families on electrical fault detection: GaussianNB vs. kNN.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FAULT_COLS = ["G", "C", "B", "A"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electrical_fault_data.csv"
RANDOM_STATE = 4320
K_VALUES = [1, 3, 5, 7, 11, 100]


# Load the capstone dataset and build the binary target used in earlier assignments.
def load_data(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path)
    X = df.drop(columns=FAULT_COLS)
    y = (df[FAULT_COLS].sum(axis=1) > 0).astype(int).to_numpy()
    row_ids = df.index.to_numpy()
    return df, X, y, row_ids


# Compute the classification metrics used to compare the model families.
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# Format a metric dictionary into a single printable line.
def format_metrics(metrics):
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )


# Print split sizes and positive-class rates for the reproducible train/val/test partition.
def print_split_info(y_train, y_val, y_test):
    print("Split sizes:")
    print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print("Class balance (positive = any fault):")
    print(f"  train: positives={int(y_train.sum())}, rate={y_train.mean():.3f}")
    print(f"  val:   positives={int(y_val.sum())}, rate={y_val.mean():.3f}")
    print(f"  test:  positives={int(y_test.sum())}, rate={y_test.mean():.3f}")


# Show a few representative false negatives and false positives for qualitative discussion.
def print_error_analysis(model_name, model, X, y, row_ids):
    y_pred = model.predict(X)
    wrong_positions = np.where(y_pred != y)[0]
    if len(wrong_positions) == 0:
        print(f"No mistakes for {model_name} on this split.")
        return

    wrong_ids = row_ids[wrong_positions]
    wrong_df = X.iloc[wrong_positions].copy()
    wrong_df.insert(0, "row_id", wrong_ids)
    wrong_df["y_true"] = y[wrong_positions]
    wrong_df["y_pred"] = y_pred[wrong_positions]

    print(f"\n{model_name} error analysis: {len(wrong_positions)} misclassified rows")

    fn = wrong_df[(wrong_df["y_true"] == 1) & (wrong_df["y_pred"] == 0)].head(3)
    fp = wrong_df[(wrong_df["y_true"] == 0) & (wrong_df["y_pred"] == 1)].head(3)

    if len(fn):
        print("  False negatives (fault predicted no fault):")
        for row in fn.itertuples(index=False):
            print(
                "    "
                f"row={row.row_id}, Ia={row.Ia:.3f}, Ib={row.Ib:.3f}, Ic={row.Ic:.3f}, "
                f"Va={row.Va:.3f}, Vb={row.Vb:.3f}, Vc={row.Vc:.3f}"
            )
    if len(fp):
        print("  False positives (no fault predicted fault):")
        for row in fp.itertuples(index=False):
            print(
                "    "
                f"row={row.row_id}, Ia={row.Ia:.3f}, Ib={row.Ib:.3f}, Ic={row.Ic:.3f}, "
                f"Va={row.Va:.3f}, Vb={row.Vb:.3f}, Vc={row.Vc:.3f}"
            )


def main():
    # Load the capstone data and keep the same binary target definition as HW5/HW6 Part B.
    df, X, y, row_ids = load_data(DATA_PATH)

    # Create a reproducible 60/20/20 stratified split because this dataset does not ship with fixed ids.
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X,
        y,
        row_ids,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_train_val,
        y_train_val,
        ids_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print("HW7 Part B: Comparing GaussianNB and kNN on Electrical Fault Data")
    print("=================================================================")
    print(f"Rows: {len(df)}")
    print("Naive Bayes note: MultinomialNB is not appropriate here because these features are")
    print("continuous sensor readings that include negative values, so GaussianNB is the right NB variant.")
    print_split_info(y_train, y_val, y_test)

    # Train GaussianNB on standardized numeric features.
    gnb_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("gnb", GaussianNB()),
        ]
    )
    gnb_model.fit(X_train, y_train)

    y_val_pred_gnb = gnb_model.predict(X_val)
    gnb_val_metrics = evaluate(y_val, y_val_pred_gnb)
    print("\nGaussianNB validation metrics:")
    print(format_metrics(gnb_val_metrics))
    print("Why GaussianNB? Each feature is continuous, so the model assumes class-conditional Gaussian distributions.")

    # Try several neighborhood sizes for kNN on the same standardized features.
    knn_results = []

    for k in K_VALUES:
        knn_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=k,
                        metric="euclidean",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        knn_model.fit(X_train, y_train)
        y_val_pred_knn = knn_model.predict(X_val)
        metrics_knn = evaluate(y_val, y_val_pred_knn)
        knn_results.append({"k": k, **metrics_knn})

    knn_df = pd.DataFrame(knn_results)
    print("\nValidation results for kNN:")
    print(knn_df.to_string(index=False))

    # Select the best k using validation F1 so model selection stays off the test split.
    best_row = knn_df.loc[knn_df["f1"].idxmax()]
    best_k = int(best_row["k"])
    best_knn_val_metrics = {
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }
    print(f"Best k selected by validation F1: k={best_k}")

    # Refit the best kNN model on the original training split for the final comparison.
    best_knn = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=best_k,
                    metric="euclidean",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    best_knn.fit(X_train, y_train)

    # Print the validation comparison before looking at qualitative errors.
    print("\nValidation comparison:")
    print("model, key_hyperparams, accuracy, precision, recall, f1")
    print("GaussianNB, standardized numeric features, " + format_metrics(gnb_val_metrics))
    print(f"kNN (k={best_k}, euclidean), standardized numeric features, " + format_metrics(best_knn_val_metrics))

    print_error_analysis("GaussianNB", gnb_model, X_val, y_val, ids_val)
    print_error_analysis("kNN", best_knn, X_val, y_val, ids_val)

    # Choose the final model from validation performance only, then evaluate once on test.
    winner_name = "GaussianNB"
    winner_model = gnb_model
    winner_reason = (
        "GaussianNB had the higher validation F1, suggesting the class-conditional distribution assumption "
        "fit the sensor measurements better than local-neighbor voting."
    )

    if best_knn_val_metrics["f1"] > gnb_val_metrics["f1"]:
        winner_name = f"kNN (k={best_k})"
        winner_model = best_knn
        winner_reason = (
            "kNN had the higher validation F1, suggesting nearby sensor patterns were more informative "
            "than the Gaussian independence assumptions."
        )

    print(f"\nWinner selected by validation F1: {winner_name}")
    print(f"Selection note: {winner_reason}")
    y_test_pred = winner_model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred)
    print("Final test metrics for winner:")
    print(format_metrics(test_metrics))


if __name__ == "__main__":
    main()
