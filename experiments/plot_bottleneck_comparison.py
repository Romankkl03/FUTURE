from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.agregate_results import read_summary


DEFAULT_SUMMARY_PATH = Path("results/fusion_over_raw/summary.csv")
DEFAULT_OUTPUT_DIR = Path("results/fusion_over_raw/plots")

MODALITIES = "raw+stats+gaf+stft"

METHOD_ORDER = [
    "ordinary_bottleneck",
    "raw_residual_bottleneck",
    "context_only_residual_bottleneck",
    "raw_conditioned_context_bottleneck",
]

METHOD_LABELS = {
    "ordinary_bottleneck": "ordinary",
    "raw_residual_bottleneck": "raw_residual",
    "context_only_residual_bottleneck": "context_only_residual",
    "raw_conditioned_context_bottleneck": "raw_conditioned_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bottleneck variants on raw+stats+gaf+stft across datasets.",
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def method_from_architecture(architecture: str) -> str:
    return architecture.split("__", 1)[0]


def filter_bottleneck_runs(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[
        (df["modalities"] == MODALITIES)
        & df["architecture"].map(method_from_architecture).isin(METHOD_ORDER)
    ].copy()
    if subset.empty:
        raise ValueError(f"No bottleneck runs found for modalities={MODALITIES}")
    subset["method"] = subset["architecture"].map(method_from_architecture)
    return subset


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    subset = filter_bottleneck_runs(df)
    ordinary_by_seed = (
        subset.loc[subset["method"] == "ordinary_bottleneck", ["dataset", "seed", "macro_f1"]]
        .rename(columns={"macro_f1": "ordinary_macro_f1"})
    )

    rows: list[dict[str, object]] = []
    for dataset in sorted(subset["dataset"].unique()):
        dataset_runs = subset[subset["dataset"] == dataset]
        ordinary_median = float(
            dataset_runs.loc[dataset_runs["method"] == "ordinary_bottleneck", "macro_f1"].median()
        )
        for method in METHOD_ORDER:
            method_runs = dataset_runs[dataset_runs["method"] == method]
            paired = method_runs.merge(ordinary_by_seed, on=["dataset", "seed"], how="inner")
            median_f1 = float(method_runs["macro_f1"].median())
            share_above_ordinary = (
                np.nan
                if method == "ordinary_bottleneck"
                else float((paired["macro_f1"] > paired["ordinary_macro_f1"]).mean())
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "median_f1": median_f1,
                    "delta_vs_ordinary": median_f1 - ordinary_median,
                    "share_above_ordinary": share_above_ordinary,
                }
            )

    table = pd.DataFrame(rows)
    table["method"] = pd.Categorical(table["method"], categories=METHOD_ORDER, ordered=True)
    return table.sort_values(["dataset", "method"])


def build_median_matrix(table: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    matrix = table.pivot(index="dataset", columns="method", values="median_f1")
    return matrix.reindex(index=datasets, columns=METHOD_ORDER)


def build_delta_vs_best_matrix(median_matrix: pd.DataFrame) -> pd.DataFrame:
    row_best = median_matrix.max(axis=1)
    return median_matrix.sub(row_best, axis=0)


def best_method_per_row(matrix: pd.DataFrame) -> pd.Series:
    best: dict[str, str | None] = {}
    for dataset in matrix.index:
        row = matrix.loc[dataset]
        valid = row.dropna()
        if valid.empty:
            best[dataset] = None
            continue
        best[dataset] = valid.idxmax()
    return pd.Series(best, name="best_method")


def draw_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    column_labels: list[str],
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
    highlight_best: bool,
    cbar_label: str,
    show_cbar: bool,
) -> None:
    display = matrix.copy()
    display.columns = column_labels
    mask = display.isna()

    sns.heatmap(
        display,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        cbar=show_cbar,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 10},
    )

    if highlight_best:
        best_methods = best_method_per_row(matrix)
        for row_idx, dataset in enumerate(matrix.index):
            best_method = best_methods[dataset]
            if best_method is None:
                continue
            col_idx = METHOD_ORDER.index(best_method)
            ax.add_patch(
                patches.Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.5,
                )
            )

    ax.set_title(title)
    ax.set_xlabel("Bottleneck variant")
    ax.set_ylabel("Dataset")
    ax.tick_params(axis="x", rotation=25, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    if show_cbar and ax.collections:
        ax.collections[-1].colorbar.set_label(cbar_label)


def plot_bottleneck_comparison(
    table: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path, Path, Path]:
    datasets = sorted(table["dataset"].unique())
    column_labels = [METHOD_LABELS[method] for method in METHOD_ORDER]

    median_matrix = build_median_matrix(table, datasets)
    delta_matrix = build_delta_vs_best_matrix(median_matrix)

    median_values = median_matrix.to_numpy().ravel()
    median_values = median_values[~np.isnan(median_values)]
    median_vmin = float(median_values.min())
    median_vmax = float(median_values.max())

    delta_values = delta_matrix.to_numpy().ravel()
    delta_values = delta_values[~np.isnan(delta_values)]
    delta_vmin = float(delta_values.min())
    delta_vmax = 0.0

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, max(4.8, 0.55 * len(datasets) + 2.2)),
        constrained_layout=True,
    )

    draw_heatmap(
        axes[0],
        median_matrix,
        title="Median macro_f1 over seeds",
        column_labels=column_labels,
        cmap="YlGnBu",
        vmin=median_vmin,
        vmax=median_vmax,
        fmt=".3f",
        highlight_best=True,
        cbar_label="Median macro_f1",
        show_cbar=True,
    )
    draw_heatmap(
        axes[1],
        delta_matrix,
        title="Delta vs best bottleneck in row",
        column_labels=column_labels,
        cmap="RdPu",
        vmin=delta_vmin,
        vmax=delta_vmax,
        fmt=".3f",
        highlight_best=True,
        cbar_label="Delta vs row best",
        show_cbar=True,
    )

    fig.suptitle(
        f"Bottleneck variants on {MODALITIES.replace('gaf', 'GAF').replace('stft', 'STFT')}",
        y=1.03,
        fontsize=16,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "bottleneck_variants_heatmaps.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    table_path = output_dir / "bottleneck_variants.csv"
    table.to_csv(table_path, index=False)

    median_only_path = output_dir / "bottleneck_variants_median_heatmap.png"
    fig_m, ax_m = plt.subplots(
        figsize=(8.5, max(4.8, 0.55 * len(datasets) + 2.2)),
        constrained_layout=True,
    )
    draw_heatmap(
        ax_m,
        median_matrix,
        title="Median macro_f1 over seeds",
        column_labels=column_labels,
        cmap="YlGnBu",
        vmin=median_vmin,
        vmax=median_vmax,
        fmt=".3f",
        highlight_best=True,
        cbar_label="Median macro_f1",
        show_cbar=True,
    )
    ax_m.set_title("")
    fig_m.suptitle(
        f"Bottleneck variants on {MODALITIES.replace('gaf', 'GAF').replace('stft', 'STFT')}",
        y=1.02,
        fontsize=15,
    )
    fig_m.savefig(median_only_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_m)

    delta_only_path = output_dir / "bottleneck_variants_delta_heatmap.png"
    fig_d, ax_d = plt.subplots(
        figsize=(8.5, max(4.8, 0.55 * len(datasets) + 2.2)),
        constrained_layout=True,
    )
    draw_heatmap(
        ax_d,
        delta_matrix,
        title="Delta vs best bottleneck in row",
        column_labels=column_labels,
        cmap="RdPu",
        vmin=delta_vmin,
        vmax=delta_vmax,
        fmt=".3f",
        highlight_best=True,
        cbar_label="Delta vs row best",
        show_cbar=True,
    )
    ax_d.set_title("")
    fig_d.suptitle(
        f"Bottleneck delta vs row best ({MODALITIES.replace('gaf', 'GAF').replace('stft', 'STFT')})",
        y=1.02,
        fontsize=15,
    )
    fig_d.savefig(delta_only_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_d)

    return table_path, figure_path, median_only_path, delta_only_path


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    df = read_summary(Path(args.summary))
    table = build_comparison_table(df)
    table_path, combined_path, median_path, delta_path = plot_bottleneck_comparison(
        table, output_dir=Path(args.output_dir), dpi=args.dpi
    )
    return table_path, median_path, delta_path, combined_path


if __name__ == "__main__":
    args = parse_args()
    table_path, median_path, delta_path, combined_path = run(args)
    print(f"- {table_path}")
    print(f"- {median_path}")
    print(f"- {delta_path}")
    print(f"- {combined_path}")
