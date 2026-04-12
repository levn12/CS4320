# CS 4320 - Assignment 10 Part A
# Unsupervised learning with PCA and k-means clustering.

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "retail_customer_behavior_unsupervised.csv"
PLOT_PATH = BASE_DIR / "hw10_part_a_pca_clusters.png"

# Keep all unsupervised results reproducible across runs.
RANDOM_STATE = 4320
# Try several cluster counts so the final choice is evidence-based instead of guessed.
K_VALUES = [2, 3, 4, 5, 6]
# Report the first several PCA variance ratios before focusing on a 2D view.
N_VARIANCE_COMPONENTS_TO_REPORT = 8


def load_data(data_path: Path = DATA_PATH):
    # Read the provided unsupervised dataset from disk.
    df = pd.read_csv(data_path)

    # Exclude the identifier because it is not a behavioral feature.
    excluded_columns = ["customer_id"]
    X = df.drop(columns=excluded_columns)

    return df, X, excluded_columns


def build_preprocessor(X: pd.DataFrame):
    # Split columns by type so numeric and categorical data can be handled appropriately.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    # Keep imputation, scaling, and one-hot encoding inside one transformer so the workflow is explicit.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        # Median imputation is a simple robust choice for numeric features.
                        ("imputer", SimpleImputer(strategy="median")),
                        # Standardize numeric columns because PCA and k-means are scale-sensitive.
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        # Use the most frequent category so missing categorical values are still usable.
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        # One-hot encoding lets categorical features participate in PCA and k-means.
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor, numeric_cols, categorical_cols


def fit_preprocessor(X: pd.DataFrame):
    # Fit the preprocessing pipeline once and return the transformed matrix for unsupervised analysis.
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return preprocessor, X_processed, feature_names, numeric_cols, categorical_cols


def run_pca(X_processed: np.ndarray, feature_names):
    # Fit PCA across all components so we can inspect explained variance broadly.
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full_scores = pca_full.fit_transform(X_processed)

    # Fit a simple 2D PCA view for the required scatter plot.
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_2d_scores = pca_2d.fit_transform(X_processed)

    # Record the strongest loading magnitudes to support cautious interpretation of PC1 and PC2.
    loading_summaries = []
    for component_index, component in enumerate(pca_2d.components_, start=1):
        loadings = pd.Series(component, index=feature_names)
        top_loadings = (
            pd.DataFrame(
                {
                    "feature": loadings.abs().sort_values(ascending=False).head(6).index,
                    "loading": loadings.loc[
                        loadings.abs().sort_values(ascending=False).head(6).index
                    ].to_numpy(),
                }
            )
            .reset_index(drop=True)
        )
        top_loadings.insert(0, "component", f"PC{component_index}")
        loading_summaries.append(top_loadings)

    loading_df = pd.concat(loading_summaries, ignore_index=True)

    return pca_full, pca_full_scores, pca_2d, pca_2d_scores, loading_df


def evaluate_kmeans_options(X_processed: np.ndarray):
    # Train one k-means model for each candidate k and collect internal quality metrics.
    rows = []
    fitted_models = {}

    for k_value in K_VALUES:
        model = KMeans(
            n_clusters=k_value,
            n_init=20,
            random_state=RANDOM_STATE,
        )
        cluster_labels = model.fit_predict(X_processed)
        rows.append(
            {
                "k": k_value,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(X_processed, cluster_labels)),
            }
        )
        fitted_models[k_value] = (model, cluster_labels)

    results_df = pd.DataFrame(rows)
    return results_df, fitted_models


def choose_k(results_df: pd.DataFrame):
    # Choose the k with the highest silhouette score, then use lower inertia as a tie-breaker.
    ordered = results_df.sort_values(
        by=["silhouette_score", "inertia"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return int(ordered.loc[0, "k"])


def summarize_pca_variance(pca_full: PCA):
    # Build a compact explained-variance table for the first several principal components.
    variance_df = pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(pca_full.explained_variance_ratio_) + 1)],
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca_full.explained_variance_ratio_),
        }
    )
    return variance_df


def summarize_clusters(df: pd.DataFrame, cluster_labels, numeric_cols, categorical_cols):
    # Attach the chosen cluster labels back to the original dataframe for interpretation.
    summary_df = df.copy()
    summary_df["cluster"] = cluster_labels

    # Report cluster size because a very tiny cluster can indicate instability or a niche subgroup.
    cluster_sizes = (
        summary_df["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="count")
    )
    cluster_sizes["share"] = cluster_sizes["count"] / len(summary_df)

    # Use raw-scale feature means so the cluster descriptions are easier to write in normal language.
    numeric_summary = (
        summary_df.groupby("cluster")[numeric_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    # Also report medians to reduce sensitivity to outliers in the interpretation section.
    numeric_medians = (
        summary_df.groupby("cluster")[numeric_cols]
        .median()
        .round(2)
        .reset_index()
    )

    # For categorical features, report the most common value and its share within each cluster.
    categorical_rows = []
    for cluster_id, cluster_frame in summary_df.groupby("cluster"):
        for column in categorical_cols:
            counts = cluster_frame[column].fillna("missing").value_counts(normalize=True)
            top_category = counts.index[0]
            top_share = counts.iloc[0]
            categorical_rows.append(
                {
                    "cluster": int(cluster_id),
                    "feature": column,
                    "top_category": top_category,
                    "top_share": float(top_share),
                }
            )
    categorical_summary = pd.DataFrame(categorical_rows)

    return summary_df, cluster_sizes, numeric_summary, numeric_medians, categorical_summary


def build_cluster_profile_table(numeric_summary: pd.DataFrame, cluster_sizes: pd.DataFrame):
    # Pick a small set of high-signal numeric features for a compact printable cluster profile.
    profile_features = [
        "annual_income_k",
        "monthly_orders",
        "avg_basket_usd",
        "discount_share",
        "app_sessions_per_month",
        "website_minutes_per_month",
        "support_tickets_6m",
        "days_since_last_order",
        "satisfaction_score",
        "account_balance_points",
    ]
    available_features = [feature for feature in profile_features if feature in numeric_summary.columns]
    profile_df = numeric_summary[["cluster", *available_features]].merge(cluster_sizes, on="cluster", how="left")
    return profile_df


def print_feature_selection_info(numeric_cols, categorical_cols, excluded_columns):
    # Make the included and excluded feature choices explicit for the assignment prompt.
    print("Feature selection for unsupervised analysis:")
    print(f"  included numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"  included categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(
        f"  excluded features ({len(excluded_columns)}): {excluded_columns} "
        "(identifier columns are not meaningful behavioral signals for distance-based methods)"
    )


def print_missing_value_info(df: pd.DataFrame):
    # Show where missingness exists before explaining the imputation choices.
    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    print("\nMissing-value summary:")
    if missing_counts.empty:
        print("  No missing values were found.")
    else:
        print(missing_counts.to_string())
        print("  Numeric columns will use median imputation.")
        print("  Categorical columns will use most-frequent-value imputation.")


def print_pca_results(variance_df: pd.DataFrame, loading_df: pd.DataFrame):
    # Report the first several explained-variance values plus a short loading summary.
    print("\nPCA explained variance:")
    print(
        variance_df.head(N_VARIANCE_COMPONENTS_TO_REPORT).to_string(
            index=False,
            formatters={
                "explained_variance_ratio": "{:.4f}".format,
                "cumulative_explained_variance": "{:.4f}".format,
            },
        )
    )

    print("\nTop PCA loading magnitudes for the 2D view:")
    print(loading_df.to_string(index=False, formatters={"loading": "{:.4f}".format}))

    print("\nPCA interpretation notes:")
    print("  Look for broad grouping, gradients, overlap, and outliers in the reduced view.")
    print("  Do not treat PCA axes as literal real-world categories or as proof of true segments.")
    print("  A 2D projection compresses information, so overlap in the plot may hide higher-dimensional structure.")


def print_kmeans_results(results_df: pd.DataFrame, chosen_k: int):
    # Report the k sweep and explain why the final k was selected.
    print("\nk-means model comparison:")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "inertia": "{:.2f}".format,
                "silhouette_score": "{:.4f}".format,
            },
        )
    )
    print(
        f"\nChosen k: {chosen_k} "
        "(selected by the strongest silhouette score, with inertia included as supporting evidence)."
    )
    print(
        "Caution: silhouette values near zero would suggest heavy overlap; here the values are positive "
        "but still modest, so the clusters should be treated as approximate structure rather than cleanly separated groups."
    )


def print_cluster_interpretation(cluster_profile_df: pd.DataFrame, categorical_summary: pd.DataFrame):
    # Print one compact summary table and then a categorical snapshot for each cluster.
    print("\nCluster profile summary (means on original scale):")
    print(
        cluster_profile_df.to_string(
            index=False,
            formatters={
                "share": "{:.3f}".format,
                "discount_share": "{:.3f}".format,
            },
        )
    )

    print("\nMost common categorical values within each cluster:")
    display_df = categorical_summary.copy()
    print(
        display_df.to_string(
            index=False,
            formatters={"top_share": "{:.3f}".format},
        )
    )

    print("\nCluster interpretation notes:")
    print("  Use relative differences across clusters to describe patterns, not absolute labels like 'good' or 'bad' customers.")
    print("  If two clusters differ only a little on many variables, describe that ambiguity directly.")
    print("  Because this dataset does not include a target label, external agreement metrics such as adjusted Rand index are not applicable here.")


def save_pca_cluster_plot(pca_scores_2d: np.ndarray, cluster_labels, output_path: Path, chosen_k: int):
    # Save one scatter plot of the 2D PCA space colored by the chosen cluster assignments.
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(
        pca_scores_2d[:, 0],
        pca_scores_2d[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=20,
        alpha=0.65,
        edgecolors="none",
    )
    ax.set_title(f"PCA 2D projection colored by k-means clusters (k={chosen_k})")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, alpha=0.18)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", frameon=True)
    ax.add_artist(legend)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # Load the raw dataset and identify which columns will be excluded from analysis.
    df, X, excluded_columns = load_data(DATA_PATH)

    # Fit the preprocessing workflow once so PCA and k-means operate on the same transformed data.
    preprocessor, X_processed, feature_names, numeric_cols, categorical_cols = fit_preprocessor(X)

    # Run PCA for explained variance reporting and the required 2D visualization.
    pca_full, pca_full_scores, pca_2d, pca_scores_2d, loading_df = run_pca(X_processed, feature_names)
    variance_df = summarize_pca_variance(pca_full)

    # Evaluate several k-means options before selecting a final cluster count.
    kmeans_results_df, fitted_models = evaluate_kmeans_options(X_processed)
    chosen_k = choose_k(kmeans_results_df)
    chosen_model, chosen_cluster_labels = fitted_models[chosen_k]

    # Summarize the chosen clusters back on the original raw-scale features for interpretation.
    (
        clustered_df,
        cluster_sizes_df,
        numeric_summary_df,
        numeric_medians_df,
        categorical_summary_df,
    ) = summarize_clusters(df, chosen_cluster_labels, numeric_cols, categorical_cols)
    cluster_profile_df = build_cluster_profile_table(numeric_summary_df, cluster_sizes_df)

    # Save the required reduced-space scatter plot.
    save_pca_cluster_plot(pca_scores_2d, chosen_cluster_labels, PLOT_PATH, chosen_k)

    print("Assignment 10 Part A - Unsupervised Learning")
    print("============================================")
    print(f"Rows: {len(df)}")
    print(f"Original feature count used for clustering/PCA: {X.shape[1]}")
    print(f"Transformed feature count after encoding: {X_processed.shape[1]}")
    print(f"Saved PCA cluster plot: {PLOT_PATH}")

    print_feature_selection_info(numeric_cols, categorical_cols, excluded_columns)
    print_missing_value_info(df)
    print_pca_results(variance_df, loading_df)
    print_kmeans_results(kmeans_results_df, chosen_k)
    print_cluster_interpretation(cluster_profile_df, categorical_summary_df)

    # Print a few extra numeric medians to make the write-up easier without turning the whole output into a report.
    print("\nCluster medians (selected for write-up support):")
    median_display_cols = [
        "cluster",
        "annual_income_k",
        "monthly_orders",
        "avg_basket_usd",
        "discount_share",
        "app_sessions_per_month",
        "support_tickets_6m",
        "days_since_last_order",
    ]
    available_median_cols = [col for col in median_display_cols if col in numeric_medians_df.columns]
    print(numeric_medians_df[available_median_cols].to_string(index=False))

    print("\nReflection notes:")
    print("  PCA is useful here for seeing broad structure and overlap in a low-dimensional view.")
    print("  k-means is useful for summarizing approximate groups that can guide later feature engineering or segmentation ideas.")
    print("  In a supervised workflow, these patterns could inform feature creation, outlier review, stratified sampling checks, or subgroup error analysis.")
    print("  Even so, unsupervised clusters do not prove natural categories, causation, or the best downstream prediction strategy.")


if __name__ == "__main__":
    main()
