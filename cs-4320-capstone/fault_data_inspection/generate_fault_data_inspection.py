"""
Generate a compact inspection package for the electrical fault dataset.

Outputs are saved alongside this script:
- data_readme.md
- several exploratory plots
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "electrical_fault_data.csv"
README_PATH = BASE_DIR / "data_readme.md"

FAULT_COLS = ["G", "C", "B", "A"]
CURRENT_COLS = ["Ia", "Ib", "Ic"]
VOLTAGE_COLS = ["Va", "Vb", "Vc"]
FEATURE_COLS = CURRENT_COLS + VOLTAGE_COLS
RANDOM_STATE = 4320
MAX_SCATTER_POINTS = 3000


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["fault_pattern"] = df[FAULT_COLS].astype(int).astype(str).agg("".join, axis=1)
    df["fault_any"] = (df["fault_pattern"] != "0000").astype(int)
    return df


def sample_for_scatter(df: pd.DataFrame, max_points: int = MAX_SCATTER_POINTS):
    if len(df) <= max_points:
        return df.copy()
    return df.sample(n=max_points, random_state=RANDOM_STATE).copy()


def build_color_mapping(patterns):
    cmap = plt.get_cmap("tab10")
    return {pattern: cmap(index % 10) for index, pattern in enumerate(patterns)}


def save_class_balance_plot(counts: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(counts.index, counts.values)
    ax.set_title("Fault pattern counts")
    ax.set_xlabel("Fault pattern")
    ax.set_ylabel("Rows")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pca_projection(df: pd.DataFrame, path: Path):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
    plot_df["fault_pattern"] = df["fault_pattern"].to_numpy()
    plot_df = sample_for_scatter(plot_df)

    patterns = sorted(plot_df["fault_pattern"].unique().tolist())
    colors = build_color_mapping(patterns)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for pattern in patterns:
        subset = plot_df[plot_df["fault_pattern"] == pattern]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            s=18,
            alpha=0.65,
            color=colors[pattern],
            label=pattern,
        )

    ax.set_title("PCA projection of electrical measurements")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Fault pattern", ncol=2, fontsize=9)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pca.explained_variance_ratio_.tolist()


def save_3d_scatter(df: pd.DataFrame, columns: list[str], title: str, path: Path):
    plot_df = sample_for_scatter(df)
    patterns = sorted(plot_df["fault_pattern"].unique().tolist())
    colors = build_color_mapping(patterns)

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for pattern in patterns:
        subset = plot_df[plot_df["fault_pattern"] == pattern]
        ax.scatter(
            subset[columns[0]],
            subset[columns[1]],
            subset[columns[2]],
            s=12,
            alpha=0.60,
            color=colors[pattern],
            label=pattern,
        )

    ax.set_title(title)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_zlabel(columns[2])
    ax.legend(title="Fault pattern", ncol=2, fontsize=8)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_phase_a_voltage_current_plot(df: pd.DataFrame, path: Path):
    plot_df = sample_for_scatter(df)
    patterns = sorted(plot_df["fault_pattern"].unique().tolist())
    colors = build_color_mapping(patterns)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for pattern in patterns:
        subset = plot_df[plot_df["fault_pattern"] == pattern]
        ax.scatter(
            subset["Ia"],
            subset["Va"],
            s=18,
            alpha=0.65,
            color=colors[pattern],
            label=pattern,
        )

    ax.set_title("Phase A voltage vs current")
    ax.set_xlabel("Ia")
    ax.set_ylabel("Va")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Fault pattern", ncol=2, fontsize=9)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_phase_voltage_current_grid(df: pd.DataFrame, path: Path):
    plot_df = sample_for_scatter(df)
    patterns = sorted(plot_df["fault_pattern"].unique().tolist())
    colors = build_color_mapping(patterns)
    phase_pairs = [("Ia", "Va"), ("Ib", "Vb"), ("Ic", "Vc")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (current_col, voltage_col) in zip(axes, phase_pairs):
        for pattern in patterns:
            subset = plot_df[plot_df["fault_pattern"] == pattern]
            ax.scatter(
                subset[current_col],
                subset[voltage_col],
                s=14,
                alpha=0.55,
                color=colors[pattern],
            )

        ax.set_title(f"{current_col} vs {voltage_col}")
        ax.set_xlabel(current_col)
        ax.set_ylabel(voltage_col)
        ax.grid(True, alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[pattern], label=pattern)
        for pattern in patterns
    ]
    fig.legend(handles=handles, title="Fault pattern", loc="upper center", ncol=6)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, path: Path):
    correlation = df[FEATURE_COLS].corr()

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Feature correlation heatmap")
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_yticklabels(FEATURE_COLS)

    for row_index in range(correlation.shape[0]):
        for col_index in range(correlation.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{correlation.iat[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_feature_histograms(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, column in zip(axes, FEATURE_COLS):
        ax.hist(df[column], bins=40, color="tab:blue", alpha=0.75, edgecolor="black", linewidth=0.4)
        ax.set_title(column)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Feature distributions", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fault_pattern_mean_profile(df: pd.DataFrame, path: Path):
    grouped = df.groupby("fault_pattern")[FEATURE_COLS].mean().sort_index()
    standardized = (grouped - grouped.mean(axis=0)) / grouped.std(axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for pattern in standardized.index:
        ax.plot(FEATURE_COLS, standardized.loc[pattern], marker="o", label=pattern)

    ax.set_title("Standardized mean feature profile by fault pattern")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Standardized mean")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Fault pattern", ncol=3, fontsize=9)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_pca_variance_plot(explained_variance: list[float], path: Path):
    cumulative = np.cumsum(explained_variance)
    component_indices = np.arange(1, len(explained_variance) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    ax.bar(component_indices, explained_variance, alpha=0.75, label="Individual")
    ax.plot(component_indices, cumulative, marker="o", color="black", label="Cumulative")
    ax.set_title("PCA explained variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(component_indices)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_readme(df: pd.DataFrame, explained_variance: list[float], generated_files: list[str]):
    feature_summary = df[FEATURE_COLS].agg(["mean", "std", "min", "max"]).round(4)
    class_counts = df["fault_pattern"].value_counts().sort_index()
    correlation = df[FEATURE_COLS].corr()

    strongest_pairs = []
    for row_index, row_name in enumerate(FEATURE_COLS):
        for col_index in range(row_index + 1, len(FEATURE_COLS)):
            col_name = FEATURE_COLS[col_index]
            strongest_pairs.append((row_name, col_name, correlation.loc[row_name, col_name]))
    strongest_pairs = sorted(strongest_pairs, key=lambda item: abs(item[2]), reverse=True)[:6]

    lines = [
        "# Electrical Fault Data Inspection",
        "",
        "## Dataset Summary",
        "",
        f"- Source file: `{DATA_PATH.name}`",
        f"- Rows: `{len(df)}`",
        f"- Measurement features: `{', '.join(FEATURE_COLS)}`",
        f"- Fault indicator columns: `{', '.join(FAULT_COLS)}`",
        f"- Unique fault patterns: `{', '.join(sorted(class_counts.index.tolist()))}`",
        f"- Rows with any fault: `{int(df['fault_any'].sum())}`",
        f"- Rows with no fault: `{int((df['fault_any'] == 0).sum())}`",
        "",
        "## Fault Pattern Counts",
        "",
    ]

    for pattern, count in class_counts.items():
        lines.append(f"- `{pattern}`: `{int(count)}`")

    lines.extend(
        [
            "",
            "## Feature Summary",
            "",
            "```text",
            feature_summary.to_string(),
            "```",
            "",
            "## PCA Notes",
            "",
            f"- PC1 explained variance: `{explained_variance[0] * 100:.2f}%`",
            f"- PC2 explained variance: `{explained_variance[1] * 100:.2f}%`",
            f"- First two PCs combined: `{sum(explained_variance[:2]) * 100:.2f}%`",
            "",
            "## Strong Feature Correlations",
            "",
        ]
    )

    for first, second, value in strongest_pairs:
        lines.append(f"- `{first}` vs `{second}`: `{value:.3f}`")

    lines.extend(
        [
            "",
            "## Quick Interpretation",
            "",
            "- The current channels and voltage channels both show strong structure rather than random scatter.",
            "- The PCA projection is useful for seeing that the fault patterns occupy related but only partially separated regions.",
            "- The per-phase voltage-vs-current plots help show that the data behaves more like structured electrical states than isolated independent measurements.",
            "- The standardized mean-profile plot helps show how each fault pattern shifts the six measurements in a consistent way.",
            "",
            "## Generated Files",
            "",
        ]
    )

    for file_name in generated_files:
        lines.append(f"- `{file_name}`")

    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    df = load_data()

    generated_files = [
        "class_balance.png",
        "pca_projection_2d.png",
        "pca_explained_variance.png",
        "currents_3d.png",
        "voltages_3d.png",
        "phase_a_voltage_vs_current.png",
        "phase_voltage_current_grid.png",
        "feature_correlation_heatmap.png",
        "feature_distributions.png",
        "fault_pattern_mean_profile.png",
        "data_readme.md",
    ]

    save_class_balance_plot(df["fault_pattern"].value_counts().sort_index(), BASE_DIR / "class_balance.png")
    explained_variance = save_pca_projection(df, BASE_DIR / "pca_projection_2d.png")
    save_pca_variance_plot(explained_variance, BASE_DIR / "pca_explained_variance.png")
    save_3d_scatter(df, CURRENT_COLS, "3D current space", BASE_DIR / "currents_3d.png")
    save_3d_scatter(df, VOLTAGE_COLS, "3D voltage space", BASE_DIR / "voltages_3d.png")
    save_phase_a_voltage_current_plot(df, BASE_DIR / "phase_a_voltage_vs_current.png")
    save_phase_voltage_current_grid(df, BASE_DIR / "phase_voltage_current_grid.png")
    save_correlation_heatmap(df, BASE_DIR / "feature_correlation_heatmap.png")
    save_feature_histograms(df, BASE_DIR / "feature_distributions.png")
    save_fault_pattern_mean_profile(df, BASE_DIR / "fault_pattern_mean_profile.png")
    build_readme(df, explained_variance, generated_files)

    print(f"Saved data inspection outputs to: {BASE_DIR}")


if __name__ == "__main__":
    main()
