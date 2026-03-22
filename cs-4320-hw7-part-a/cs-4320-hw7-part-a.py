# CS 4320 - Assignment 7 Part A
# Comparing model families: Naive Bayes vs. kNN on SMS spam.

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


BASE_DIR = Path(__file__).resolve().parent


# Load the saved message table, sparse feature matrices, labels, and fixed split ids.
def load_data(data_dir: Path = BASE_DIR):
    messages = pd.read_csv(data_dir / "messages.csv")
    X_counts = sparse.load_npz(data_dir / "X_counts.npz")
    X_tfidf = sparse.load_npz(data_dir / "X_tfidf.npz")
    y = messages["label"].to_numpy(dtype=int)

    with open(data_dir / "split.json", "r") as f:
        split = json.load(f)
    train_ids = np.array(split["train_ids"], dtype=int)
    val_ids = np.array(split["val_ids"], dtype=int)
    test_ids = np.array(split["test_ids"], dtype=int)

    return {
        "messages": messages,
        "X_counts": X_counts,
        "X_tfidf": X_tfidf,
        "y": y,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


# Compute the assignment's classification metrics for a set of predictions.
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# Format metric values into one short printable line.
def format_metrics(metrics):
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )


# Print the number of examples in each instructor-provided split.
def print_split_info(train_ids, val_ids, test_ids):
    print("Split sizes:")
    print(f"  train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")


# Print ham/spam counts so we can confirm the splits are reasonably balanced.
def print_class_balance(y, train_ids, val_ids, test_ids):
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        counts = np.bincount(y[ids], minlength=2)
        print(f"  {name}: ham={counts[0]}, spam={counts[1]}, spam_rate={counts[1]/len(ids):.3f}")


# Show a few representative mistakes so the report can discuss model behavior qualitatively.
def print_error_analysis(model_name, model, X, y, ids, messages):
    y_pred = model.predict(X)
    wrong_positions = np.where(y_pred != y)[0]
    if len(wrong_positions) == 0:
        print(f"No mistakes for {model_name} on this split.")
        return

    wrong_ids = ids[wrong_positions]
    y_true = y[wrong_positions]
    y_pred_sub = y_pred[wrong_positions]
    text = messages.loc[wrong_ids, "text"].to_numpy()

    print(f"\n{model_name} error analysis: {len(wrong_positions)} misclassified examples")
    # Separate false negatives from false positives so the error types are easy to compare.
    df = pd.DataFrame({"idx": wrong_ids, "text": text, "y_true": y_true, "y_pred": y_pred_sub})
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].head(3)
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].head(3)

    if len(fn):
        print("  False negatives (spam predicted ham):")
        for row in fn.itertuples(index=False):
            print(f"    idx={row.idx}, text='{row.text[:80]}...', y_true={row.y_true}, y_pred={row.y_pred}")
    if len(fp):
        print("  False positives (ham predicted spam):")
        for row in fp.itertuples(index=False):
            print(f"    idx={row.idx}, text='{row.text[:80]}...', y_true={row.y_true}, y_pred={row.y_pred}")


def main():
    # Load the fixed features, labels, and splits from the dataset directory.
    data = load_data(BASE_DIR)
    messages = data["messages"]
    X_counts = data["X_counts"]
    X_tfidf = data["X_tfidf"]
    y = data["y"]
    train_ids = data["train_ids"]
    val_ids = data["val_ids"]
    test_ids = data["test_ids"]

    print("HW7 Part A: Comparing Naive Bayes and kNN")
    print("===========================================")
    print_split_info(train_ids, val_ids, test_ids)
    print_class_balance(y, train_ids, val_ids, test_ids)

    # Slice each feature representation into the same train/validation/test partitions.
    X_train_counts = X_counts[train_ids]
    X_val_counts = X_counts[val_ids]
    X_test_counts = X_counts[test_ids]

    X_train_tfidf = X_tfidf[train_ids]
    X_val_tfidf = X_tfidf[val_ids]
    X_test_tfidf = X_tfidf[test_ids]

    y_train = y[train_ids]
    y_val = y[val_ids]
    y_test = y[test_ids]

    # Train Multinomial Naive Bayes on raw count features because the model expects non-negative counts.
    nb_model = MultinomialNB()
    nb_model.fit(X_train_counts, y_train)

    y_val_pred_nb = nb_model.predict(X_val_counts)
    nb_val_metrics = evaluate(y_val, y_val_pred_nb)
    print("\nNaive Bayes (MultinomialNB) validation metrics:")
    print(format_metrics(nb_val_metrics))
    print("Why MultinomialNB? The data are non-negative token counts from CountVectorizer.")

    # Check Naive Bayes assumptions qualitatively on spam classification.
    print("Naive Bayes assumption note: conditional independence is not strictly true for text,")
    print("but MultinomialNB often works well on bag-of-words counts in spam detection.")

    # Try several k values for kNN on TF-IDF features using MaxAbsScaler to preserve sparsity.
    k_values = [1, 3, 5, 7]
    knn_results = []

    for k in k_values:
        # Build a fresh pipeline for each k so scaling and distance-based classification stay linked together.
        knn_pipe = Pipeline(
            [
                ("scale", MaxAbsScaler()),
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
        knn_pipe.fit(X_train_tfidf, y_train)
        y_val_pred_knn = knn_pipe.predict(X_val_tfidf)
        metrics_knn = evaluate(y_val, y_val_pred_knn)
        knn_results.append({"k": k, **metrics_knn})

    knn_df = pd.DataFrame(knn_results)
    print("\nValidation results for kNN (TF-IDF + MaxAbs scaling):")
    print(knn_df.to_string(index=False))

    # Select the best k using validation F1 because that is the assignment's comparison metric.
    best_row = knn_df.loc[knn_df["f1"].idxmax()]
    best_k = int(best_row["k"])
    best_knn_val_metrics = {
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }
    print(f"Best k selected by validation F1: k={best_k}")

    # Refit the chosen kNN model on the original training split for the later comparison steps.
    best_knn = Pipeline(
        [
            ("scale", MaxAbsScaler()),
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
    best_knn.fit(X_train_tfidf, y_train)

    # Print the validation comparison table before looking at specific mistakes.
    print("\nValidation comparison:")
    print("model, key_hyperparams, accuracy, precision, recall, f1")
    print(
        "Naive Bayes, MultinomialNB (counts), "
        + format_metrics(nb_val_metrics)
    )
    print(
        f"kNN (k={best_k}, euclidean, TF-IDF), "
        + format_metrics(best_knn_val_metrics)
    )

    print_error_analysis("Naive Bayes", nb_model, X_val_counts, y_val, val_ids, messages)
    print_error_analysis("kNN", best_knn, X_val_tfidf, y_val, val_ids, messages)

    # Choose the final model strictly from validation performance before touching the test split.
    winner_name = "Naive Bayes"
    winner_model = nb_model
    winner_X_test = X_test_counts
    winner_reason = "Naive Bayes had the higher validation F1, which matches its strength on sparse count data."

    if best_knn_val_metrics["f1"] > nb_val_metrics["f1"]:
        winner_name = f"kNN (k={best_k})"
        winner_model = best_knn
        # kNN used TF-IDF features, so the test set must use the same representation.
        winner_X_test = X_test_tfidf
        winner_reason = "kNN had the higher validation F1, suggesting neighborhood similarity in TF-IDF space worked better."

    print(f"\nWinner selected by validation F1: {winner_name}")
    print(f"Selection note: {winner_reason}")
    y_test_pred = winner_model.predict(winner_X_test)
    test_metrics = evaluate(y_test, y_test_pred)
    print("Final test metrics for winner:")
    print(format_metrics(test_metrics))


if __name__ == "__main__":
    main()
