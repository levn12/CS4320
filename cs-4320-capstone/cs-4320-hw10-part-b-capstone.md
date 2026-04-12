## 1. Project Context (Brief)

* **Project Title:** Electrical Grid Fault Detection
* **Data Modality:** Tabular
* **Task Type:** Unsupervised learning
* **One-Sentence Goal:** Use electrical current and voltage measurements to explore latent structure in the data with PCA and k-means clustering, then compare that structure to the known fault patterns only after clustering.

---

## 2. This Week's Technique and Its Assumptions

* **Technique / Model Family Covered This Week:** Principal component analysis for dimensionality reduction and k-means clustering for unsupervised grouping.
* **Key Assumptions of This Technique:**
  * PCA is most informative when variance structure in the scaled features reflects meaningful underlying patterns.
  * k-means works best when clusters are roughly compact and distance-based in the transformed feature space.
  * Cluster assignments are not labels, causes, or proof of natural categories.

**Fit Assessment (required):**

> I expect this technique to be a **good** fit for my project because:

My electrical fault dataset has only six continuous measurement features, so it is a good size for PCA and easy to visualize after dimensionality reduction. It also includes known fault-pattern labels, which makes it possible to do the assignment correctly by clustering without labels first and then comparing the clusters to the labels afterward. That combination makes this week useful because it shows both what the geometry of the measurements suggests on its own and where that geometry does not fully line up with the known fault categories.

---

## 3. Representation or Proxy Used

* **Representation or Proxy Chosen:** The six raw electrical measurement features `Ia`, `Ib`, `Ic`, `Va`, `Vb`, and `Vc` were used as the unsupervised feature space. The four fault-indicator columns `G`, `C`, `B`, and `A` were excluded from clustering and PCA, then combined afterward into post hoc fault-pattern labels such as `0000`, `0110`, `0111`, `1001`, `1011`, and `1111`.

* **Why this representation was reasonable for this week:**

This representation stays close to the physical sensor data instead of clustering on already-labeled columns. That is important because using the fault indicators directly would leak the answer into the unsupervised step and make the clustering exercise meaningless. Using only the measured currents and voltages gives PCA and k-means a fair chance to discover structure based on the behavior of the electrical system itself.

---

## 4. What Was Attempted

I implemented a capstone version of Assignment 10 using the electrical fault dataset. The workflow followed the same general structure as Part A, but I adapted it to the capstone project and used the known fault patterns only for post hoc evaluation rather than during clustering. The main steps were:

1. Load the CSV and separate the six electrical measurements from the four fault-indicator columns.
2. Build post hoc fault-pattern labels by concatenating the four indicator bits into strings.
3. Confirm that the clustering features have no missing values, while still keeping a median-imputation step in the preprocessing pipeline for consistency.
4. Standardize all six measurement features because both PCA and k-means are sensitive to scale.
5. Fit PCA on the scaled measurements and report explained variance ratios for all principal components.
6. Create a 2D PCA projection and save a scatter plot colored by the selected k-means cluster assignments.
7. Run k-means for multiple values of `k` from `2` through `8`.
8. Report inertia and silhouette score for each `k`.
9. Compute adjusted Rand index after clustering as a post hoc comparison to the known fault-pattern labels.
10. Choose a final `k`, summarize the clusters with feature means and medians, and inspect how each cluster mixes the known fault labels.

What I intentionally did not attempt:

* I did not use the fault-indicator columns as clustering features, because that would defeat the purpose of the unsupervised analysis.
* I did not force the number of clusters to equal the number of known fault patterns.
* I did not train a supervised classifier this week, because the assignment specifically focused on structure discovery rather than prediction.

Constraints encountered:

* The fault-pattern labels exist, which is useful for evaluation, but they create a temptation to judge clustering only by label agreement even though that is not the goal of unsupervised learning.
* k-means kept improving in silhouette score as `k` increased through the tested range, so choosing `k` still required judgment instead of a perfectly obvious elbow.

---

## 5. Results or Observations

The main result this week is that the electrical measurements have strong low-dimensional structure, but the clusters do not map neatly onto the known fault patterns. PCA captured most of the variance very quickly, with the first two components explaining about `56.44%` of the total variance and the first four explaining about `97.21%`. That suggests there is real structure in the measurements, and it also helps explain why the PCA visualization is meaningful instead of looking like random scatter.

Results snapshot from the run:

* Dataset rows: `7,861`
* Features used for unsupervised learning: `Ia`, `Ib`, `Ic`, `Va`, `Vb`, `Vc`
* Excluded from clustering and PCA: `G`, `C`, `B`, `A`
* Known post hoc fault patterns: `0000`, `0110`, `0111`, `1001`, `1011`, `1111`
* PCA explained variance ratios: `PC1=0.3074`, `PC2=0.2570`, `PC3=0.2253`, `PC4=0.1823`, `PC5=0.0279`, `PC6=0.0000`
* Cumulative explained variance after two components: `0.5644`
* Cumulative explained variance after four components: `0.9721`
* Best tested cluster count by internal metric: `k=8`
* Inertia at `k=8`: `11503.44`
* Silhouette score at `k=8`: `0.4006`
* Post hoc adjusted Rand index at `k=8`: `0.1688`

k-means comparison summary:

* `k=2`: inertia `35531.83`, silhouette `0.2495`, ARI `0.0026`
* `k=3`: inertia `29488.16`, silhouette `0.2584`, ARI `0.0725`
* `k=4`: inertia `24245.29`, silhouette `0.2909`, ARI `0.0762`
* `k=5`: inertia `20654.95`, silhouette `0.3173`, ARI `0.1116`
* `k=6`: inertia `17182.73`, silhouette `0.3534`, ARI `0.1534`
* `k=7`: inertia `14137.73`, silhouette `0.3799`, ARI `0.1662`
* `k=8`: inertia `11503.44`, silhouette `0.4006`, ARI `0.1688`

Additional observations:

* The PCA loading patterns suggest that the leading components are driven by relationships among phase currents and voltages rather than by one single measurement.
* The internal clustering metrics improved steadily as `k` increased through the tested range, which suggests multiple meaningful substructures in the electrical measurements.
* Even the best adjusted Rand index stayed fairly low, which means the unsupervised geometric groupings only partially aligned with the known fault categories.

---

## 6. Interpretation and Judgment

I think the biggest lesson from this week is that the measurement space has real structure, but that structure is not identical to the human-defined fault labels. The PCA results were strong enough to show that most of the variation can be summarized with only a few dimensions, which makes me more confident that the sensor readings contain consistent patterns rather than just noise. The 2D PCA plot is therefore useful for visualization, but it still cannot prove that the visible groups are true physical classes on its own. Additionally, the plot shapes indicate real signal, but it certainly doesn't look like clusters, more like loops or ellipses in feature-space.

The k-means results were also informative, especially because the silhouette score improved steadily and reached `0.4006` at `k=8`, which is much more convincing than the weaker retail example from Part A. Still, the post hoc adjusted Rand index of only `0.1688` shows that the clusters do not match the known fault patterns very closely. The unsupervised methods revealed meaningful structure in the electrical data, but they also showed an important limitation: useful geometric clusters are not automatically the same thing as the target categories I would use in a predictive model. There can also be structure present in the data aside from clusters. Perhaps there is a better unsupervised technique outhere that matches the structure of my data better.

---

## 7. Forward-Looking Adjustment

The next change I would make is to keep the same electrical measurements but push the analysis in one of these directions:

1. Compare PCA or cluster summaries within each known fault pattern to see what patterns are present interally.
2. Engineer physically motivated features such as current or voltage differences, ratios, or per-phase contrasts before rerunning PCA and clustering.
3. Find a different technique that can manage the structure of my data better than clustering.

---

## 8. Mismatch Acknowledgment (Complete Only If Applicable)

There was some mismatch this week. The clustering metrics suggested increasingly better geometric separation as `k` grew, but the post hoc agreement with the known fault patterns stayed fairly modest. That mismatch does not mean the unsupervised methods failed. Instead, it suggests the geometry of the electrical measurements may have different substructures aside from clusters, which this particular technique is geared towards..

---

## Submission Notes

* Written submission format: **Markdown or PDF**
* Code file included below: `cs-4320-hw10-part-b.py`


```python
# CS 4320 - Assignment 10 Part B
# Unsupervised learning on the electrical fault dataset with PCA and k-means clustering.

import os
from pathlib import Path

# Silence a sandbox-specific loky core-count warning so the assignment output stays readable.
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")


FAULT_COLS = ["G", "C", "B", "A"]
MEASUREMENT_COLS = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "electrical_fault_data.csv"
PLOT_PATH = BASE_DIR / "hw10_part_b_pca_clusters.png"

# Keep all unsupervised results reproducible across runs.
RANDOM_STATE = 4320
# Evaluate several candidate cluster counts before choosing one.
K_VALUES = [2, 3, 4, 5, 6, 7, 8]


def load_data(data_path: Path = DATA_PATH):
    # Read the electrical fault dataset from disk.
    df = pd.read_csv(data_path)

    # Build the post hoc reference labels from the fault-indicator bits.
    label_series = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1)

    # Use only the measurement columns for unsupervised learning.
    X = df[MEASUREMENT_COLS]
    excluded_columns = FAULT_COLS.copy()

    return df, X, label_series, excluded_columns


def build_preprocessor(X: pd.DataFrame):
    # All clustering features are numeric, so median imputation and scaling are enough here.
    numeric_cols = X.columns.tolist()
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
                numeric_cols,
            )
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor, numeric_cols


def fit_preprocessor(X: pd.DataFrame):
    # Fit the preprocessing workflow once so PCA and k-means use the same scaled feature space.
    preprocessor, numeric_cols = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return preprocessor, X_processed, feature_names, numeric_cols


def run_pca(X_processed: np.ndarray, feature_names):
    # Fit PCA over all components for explained-variance reporting.
    pca_full = PCA(random_state=4320)
    pca_full.fit(X_processed)

    # Fit a 2D PCA view for the required scatter plot.
    pca_2d = PCA(n_components=2, random_state=4320)
    pca_2d_scores = pca_2d.fit_transform(X_processed)

    # Summarize the strongest loadings so the principal axes can be interpreted carefully.
    loading_rows = []
    for component_index, component in enumerate(pca_2d.components_, start=1):
        loading_series = pd.Series(component, index=feature_names)
        ordered = loading_series.abs().sort_values(ascending=False).index
        for feature_name in ordered[:6]:
            loading_rows.append(
                {
                    "component": f"PC{component_index}",
                    "feature": feature_name,
                    "loading": float(loading_series[feature_name]),
                }
            )

    loading_df = pd.DataFrame(loading_rows)
    return pca_full, pca_2d, pca_2d_scores, loading_df


def summarize_pca_variance(pca_full: PCA):
    # Build a small explained-variance table for printing.
    return pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, len(pca_full.explained_variance_ratio_) + 1)],
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca_full.explained_variance_ratio_),
        }
    )


def evaluate_kmeans_options(X_processed: np.ndarray, true_labels: pd.Series):
    # Train one k-means model for each candidate k and collect internal and post hoc metrics.
    rows = []
    fitted_models = {}

    for k_value in K_VALUES:
        model = KMeans(
            n_clusters=k_value,
            n_init=20,
            random_state=4320,
        )
        cluster_labels = model.fit_predict(X_processed)
        rows.append(
            {
                "k": k_value,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(X_processed, cluster_labels)),
                "adjusted_rand_index": float(adjusted_rand_score(true_labels, cluster_labels)),
            }
        )
        fitted_models[k_value] = (model, cluster_labels)

    return pd.DataFrame(rows), fitted_models


def choose_k(results_df: pd.DataFrame):
    # Choose the k with the strongest silhouette score, then use lower inertia as a tie-breaker.
    ordered = results_df.sort_values(
        by=["silhouette_score", "inertia"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return int(ordered.loc[0, "k"])


def summarize_clusters(df: pd.DataFrame, cluster_labels, true_labels: pd.Series):
    # Attach clusters and true labels to the raw dataframe for interpretation tables.
    summary_df = df.copy()
    summary_df["cluster"] = cluster_labels
    summary_df["fault_pattern"] = true_labels.to_numpy()

    cluster_sizes = (
        summary_df["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster")
        .reset_index(name="count")
    )
    cluster_sizes["share"] = cluster_sizes["count"] / len(summary_df)

    measurement_means = (
        summary_df.groupby("cluster")[MEASUREMENT_COLS]
        .mean()
        .round(4)
        .reset_index()
    )
    measurement_medians = (
        summary_df.groupby("cluster")[MEASUREMENT_COLS]
        .median()
        .round(4)
        .reset_index()
    )

    contingency = pd.crosstab(summary_df["cluster"], summary_df["fault_pattern"], normalize="index")
    contingency = (100 * contingency).round(1).reset_index()

    dominant_label_rows = []
    for cluster_id, cluster_frame in summary_df.groupby("cluster"):
        label_share = cluster_frame["fault_pattern"].value_counts(normalize=True)
        dominant_label_rows.append(
            {
                "cluster": int(cluster_id),
                "dominant_fault_pattern": label_share.index[0],
                "dominant_share": float(label_share.iloc[0]),
            }
        )
    dominant_labels_df = pd.DataFrame(dominant_label_rows)

    return summary_df, cluster_sizes, measurement_means, measurement_medians, contingency, dominant_labels_df


def print_feature_selection_info(numeric_cols, excluded_columns):
    # Make the included and excluded feature choices explicit for the assignment prompt.
    print("Feature selection for unsupervised analysis:")
    print(f"  included measurement features ({len(numeric_cols)}): {numeric_cols}")
    print(
        f"  excluded columns ({len(excluded_columns)}): {excluded_columns} "
        "(these are label/indicator columns and were held out from clustering, then used only for post hoc evaluation)"
    )


def print_missing_value_info(df: pd.DataFrame):
    # Report missing values before noting the imputation strategy.
    missing_counts = df[MEASUREMENT_COLS].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    print("\nMissing-value summary for clustering features:")
    if missing_counts.empty:
        print("  No missing values were found in the measurement features.")
    else:
        print(missing_counts.to_string())
        print("  Numeric columns will use median imputation before scaling.")


def print_label_info(true_labels: pd.Series):
    # Show the available fault-pattern labels since they will be used post hoc only.
    print("\nAvailable post hoc fault-pattern labels:")
    counts = true_labels.value_counts().sort_index()
    formatted = ", ".join([f"{label}={count}" for label, count in counts.items()])
    print(f"  {formatted}")


def print_pca_results(variance_df: pd.DataFrame, loading_df: pd.DataFrame):
    # Report explained variance and the strongest PCA loadings.
    print("\nPCA explained variance:")
    print(
        variance_df.to_string(
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
    print("  The first two components capture broad measurement structure, not guaranteed physical fault categories.")
    print("  Visible separation in 2D supports structure in the data, but overlap still limits what can be concluded from the plot alone.")
    print("  PCA is a projection, so some relationships may become clearer or blurrier than they are in the full six-dimensional space.")


def print_kmeans_results(results_df: pd.DataFrame, chosen_k: int):
    # Report the k sweep and explain the selected cluster count.
    print("\nk-means model comparison:")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "inertia": "{:.2f}".format,
                "silhouette_score": "{:.4f}".format,
                "adjusted_rand_index": "{:.4f}".format,
            },
        )
    )
    print(
        f"\nChosen k: {chosen_k} "
        "(selected using internal evidence, with silhouette score as the main criterion and inertia as supporting evidence)."
    )
    print(
        "The adjusted Rand index is shown only as a post hoc comparison to the known fault-pattern labels. "
        "It does not mean the clusters are a supervised classifier or that k-means recovered the true classes exactly."
    )


def print_cluster_interpretation(cluster_profile_df: pd.DataFrame, dominant_labels_df: pd.DataFrame, contingency_df: pd.DataFrame):
    # Print a compact cluster summary plus the post hoc label mix inside each cluster.
    print("\nCluster profile summary (means on original scale):")
    print(
        cluster_profile_df.to_string(
            index=False,
            formatters={"share": "{:.3f}".format},
        )
    )

    print("\nDominant post hoc fault pattern inside each cluster:")
    print(
        dominant_labels_df.to_string(
            index=False,
            formatters={"dominant_share": "{:.3f}".format},
        )
    )

    print("\nPost hoc fault-pattern mix within each cluster (row percentages):")
    print(contingency_df.to_string(index=False))

    print("\nCluster interpretation notes:")
    print("  Some clusters align more with current/voltage magnitude and phase relationships than with a single clean fault label.")
    print("  Mixed label distributions inside a cluster mean the unsupervised groups are only partially aligned with the known fault patterns.")
    print("  This ambiguity is expected because k-means searches for geometric compactness, not the assignment's label definitions.")


def save_pca_cluster_plot(pca_scores_2d: np.ndarray, cluster_labels, output_path: Path, chosen_k: int):
    # Save one scatter plot of the 2D PCA space colored by the chosen cluster assignments.
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(
        pca_scores_2d[:, 0],
        pca_scores_2d[:, 1],
        c=cluster_labels,
        cmap="tab10",
        s=16,
        alpha=0.60,
        edgecolors="none",
    )
    ax.set_title(f"Electrical measurements in PCA space colored by k-means clusters (k={chosen_k})")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True, alpha=0.18)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", frameon=True)
    ax.add_artist(legend)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # Load the electrical fault dataset and keep the known labels separate from unsupervised modeling.
    df, X, true_labels, excluded_columns = load_data(DATA_PATH)

    # Fit preprocessing once so PCA and k-means both use the same scaled feature representation.
    preprocessor, X_processed, feature_names, numeric_cols = fit_preprocessor(X)

    # Run PCA for explained variance reporting and a 2D visualization.
    pca_full, pca_2d, pca_scores_2d, loading_df = run_pca(X_processed, feature_names)
    variance_df = summarize_pca_variance(pca_full)

    # Compare several cluster counts before choosing a final k.
    kmeans_results_df, fitted_models = evaluate_kmeans_options(X_processed, true_labels)
    chosen_k = choose_k(kmeans_results_df)
    chosen_model, chosen_cluster_labels = fitted_models[chosen_k]

    # Summarize clusters back on the original measurement scale and compare them to the known labels post hoc.
    (
        clustered_df,
        cluster_sizes_df,
        measurement_means_df,
        measurement_medians_df,
        contingency_df,
        dominant_labels_df,
    ) = summarize_clusters(df, chosen_cluster_labels, true_labels)
    cluster_profile_df = measurement_means_df.merge(cluster_sizes_df, on="cluster", how="left")

    # Save the required reduced-space scatter plot.
    save_pca_cluster_plot(pca_scores_2d, chosen_cluster_labels, PLOT_PATH, chosen_k)

    print("Assignment 10 Part B - Unsupervised Learning on Electrical Fault Data")
    print("======================================================================")
    print(f"Rows: {len(df)}")
    print(f"Original feature count used for clustering/PCA: {X.shape[1]}")
    print(f"Transformed feature count after preprocessing: {X_processed.shape[1]}")
    print(f"Saved PCA cluster plot: {PLOT_PATH}")

    print_feature_selection_info(numeric_cols, excluded_columns)
    print_missing_value_info(df)
    print_label_info(true_labels)
    print_pca_results(variance_df, loading_df)
    print_kmeans_results(kmeans_results_df, chosen_k)
    print_cluster_interpretation(cluster_profile_df, dominant_labels_df, contingency_df)

    print("\nCluster medians (selected for write-up support):")
    print(measurement_medians_df.to_string(index=False))

    chosen_row = kmeans_results_df.loc[kmeans_results_df["k"] == chosen_k].iloc[0]
    print("\nReflection notes:")
    print("  PCA shows that a small number of dimensions already captures most of the electrical variation.")
    print("  k-means finds meaningful geometric groupings in the measurement space, but those groups only partially match the true fault-pattern labels.")
    print(
        f"  For the selected k={chosen_k}, the silhouette score was {chosen_row['silhouette_score']:.4f} "
        f"and the post hoc adjusted Rand index was {chosen_row['adjusted_rand_index']:.4f}."
    )
    print("  In a supervised workflow, these insights could guide feature engineering, error analysis, label auditing, or class-structure exploration before fitting a classifier.")


if __name__ == "__main__":
    main()
```
