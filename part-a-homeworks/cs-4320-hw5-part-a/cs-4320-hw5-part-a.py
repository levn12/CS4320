import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def evaluate_at_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def print_metric_block(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for name, value in metrics.items():
        print(f"{name:10s}: {value:.4f}")


def confusion_stats(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tpr
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    # Load data
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "telco_churn.csv"

    df = pd.read_csv(data_path)
    # Parse known numeric-like column where blanks can occur. Coerced NaNs will be imputed later.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Define features and target
    target_col = "Churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 60/20/20 split via two stratified steps:
    # 1) 80% train_val, 20% test
    # 2) split train_val into 75% train and 25% val -> overall 60/20/20
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=4320,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        stratify=y_train_val,
        random_state=4320,
    )

    # Identify numeric and categorical features for preprocessing pipelines.
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    # Define preprocessing pipelines for numeric and categorical features.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine pipelines into a single ColumnTransformer that applies the appropriate transformations to each column type.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # Define the full pipeline with preprocessing and logistic regression model.
    model = LogisticRegression(max_iter=2000, random_state=4320)
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Fit only on training data to keep validation/test isolated.
    clf.fit(X_train, y_train)

    # Evaluate on validation set at a default threshold of 0.5 to get initial metrics and confusion matrix.
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds_05 = (val_probs >= 0.50).astype(int)

    cm = confusion_matrix(y_val, val_preds_05)
    tn, fp, fn, tp = cm.ravel()

    # Set threshold at 0.5 for initial evaluation and reporting.
    val_metrics_05 = evaluate_at_threshold(y_val, val_probs, 0.50)
    val_roc_auc = roc_auc_score(y_val, val_probs)
    val_precision_curve, val_recall_curve, val_thresholds_curve = precision_recall_curve(
        y_val, val_probs
    )
    val_pr_auc = auc(val_recall_curve, val_precision_curve)

    # Pick threshold using validation set: maximize F1.
    f1_by_threshold = []
    for t in val_thresholds_curve:
        f1_by_threshold.append(f1_score(y_val, (val_probs >= t).astype(int), zero_division=0))
    best_idx = int(np.argmax(f1_by_threshold))
    chosen_threshold = float(val_thresholds_curve[best_idx])
    if abs(chosen_threshold - 0.50) < 0.02:
        # Force explicit comparison with a non-0.5 threshold for assignment requirements.
        print(f"Optimal threshold based on validation F1 is {chosen_threshold:.2f}, which is close to 0.50.\n Changing to 0.35 for assignment purposes.")
        chosen_threshold = 0.35

    val_metrics_chosen = evaluate_at_threshold(y_val, val_probs, chosen_threshold)
    val_preds_chosen = (val_probs >= chosen_threshold).astype(int)
    cm_chosen = confusion_matrix(y_val, val_preds_chosen)
    tn_c, fp_c, fn_c, tp_c = cm_chosen.ravel()

    # Reporting and interpretation
    print("Assignment 5 Part A: Classification Workflow")
    print("===========================================")
    print(f"Rows: {len(df)}")
    print("Split strategy: stratified random split (preserve class balance).")
    print(
        f"Split proportions: train={len(X_train)/len(df):.2%}, "
        f"val={len(X_val)/len(df):.2%}, test={len(X_test)/len(df):.2%}"
    )
    print(
        f"Class balance (positive=1): train={y_train.mean():.2%}, "
        f"val={y_val.mean():.2%}, test={y_test.mean():.2%}"
    )

    print_metric_block("Validation metrics @ threshold=0.50", val_metrics_05)
    print(f"Validation ROC AUC : {val_roc_auc:.4f}")
    print(f"Validation PR AUC  : {val_pr_auc:.4f}")

    print("\nValidation confusion matrix @ threshold=0.50")
    print("------------------------------------------------")
    print(cm)
    print(
        f"Interpretation: TN={tn}, FP={fp}, FN={fn}, TP={tp}. "
    )

    print_metric_block(
        f"Validation metrics @ chosen threshold={chosen_threshold:.2f}",
        val_metrics_chosen,
    )
    print(f"\nValidation confusion matrix @ threshold={chosen_threshold:.2f}")
    print("-" * 48)
    print(cm_chosen)
    print(
        f"Interpretation: TN={tn_c}, FP={fp_c}, FN={fn_c}, TP={tp_c}. "
        "Compared with 0.50, this reflects the precision/recall shift caused by moving the threshold."
    )
    print(
        "Threshold tradeoff summary: lowering the threshold usually increases recall "
        "(catch more churners) but can reduce precision (more false alarms)."
    )

    # Curves on validation set with operating points for threshold=0.50 and chosen threshold.
    fpr, tpr, _ = roc_curve(y_val, val_probs)
    threshold_values = [0.50, float(chosen_threshold)]
    saved_plots = []
    for threshold in threshold_values:
        stats = confusion_stats(y_val, val_probs, threshold)
        thr_tag = f"{threshold:.2f}".replace(".", "")

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC AUC = {val_roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.scatter(
            stats["fpr"],
            stats["tpr"],
            color="red",
            label=f"Operating point @ threshold={threshold:.2f}",
            zorder=5,
        )
        plt.annotate(
            f"thr={threshold:.2f}",
            (stats["fpr"], stats["tpr"]),
            textcoords="offset points",
            xytext=(8, -12),
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Validation ROC Curve (threshold={threshold:.2f})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_out = base_dir / f"hw5_part_a_validation_roc_thr_{thr_tag}.png"
        plt.savefig(roc_out, dpi=140)
        plt.close()
        saved_plots.append(roc_out)

        plt.figure(figsize=(6, 5))
        plt.plot(val_recall_curve, val_precision_curve, label=f"PR AUC = {val_pr_auc:.3f}")
        plt.scatter(
            stats["recall"],
            stats["precision"],
            color="red",
            label=f"Operating point @ threshold={threshold:.2f}",
            zorder=5,
        )
        plt.annotate(
            f"thr={threshold:.2f}",
            (stats["recall"], stats["precision"]),
            textcoords="offset points",
            xytext=(8, -12),
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Validation Precision-Recall Curve (threshold={threshold:.2f})")
        plt.legend(loc="lower left")
        plt.tight_layout()
        pr_out = base_dir / f"hw5_part_a_validation_pr_thr_{thr_tag}.png"
        plt.savefig(pr_out, dpi=140)
        plt.close()
        saved_plots.append(pr_out)

    # Final single test evaluation with locked threshold.
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_probs, chosen_threshold)
    test_roc_auc = roc_auc_score(y_test, test_probs)
    test_precision_curve, test_recall_curve, _ = precision_recall_curve(y_test, test_probs)
    test_pr_auc = auc(test_recall_curve, test_precision_curve)

    print_metric_block(
        f"Final TEST metrics @ threshold={chosen_threshold:.2f}",
        test_metrics,
    )
    print(f"Test ROC AUC      : {test_roc_auc:.4f}")
    print(f"Test PR AUC       : {test_pr_auc:.4f}")
    print(
        "Validation vs test comparison: if the values are close, behavior generalizes. "
        "If they differ, likely causes include finite sample effects, class mix variation, "
        "or mild overfitting to validation choices."
    )
    print("\nSaved plots:")
    for plot_path in saved_plots:
        print(f"- {plot_path}")


if __name__ == "__main__":
    main()
