from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.agregate_results import BOTTLENECK_METHODS, fusion_method, read_summary


DEFAULT_SUMMARY_PATH = Path("results/fusion_over_raw/summary.csv")
DEFAULT_OUTPUT_DIR = Path("results/fusion_over_raw/plots")

FAMILY_ORDER = [
    "single",
    "concat",
    "gated",
    "raw_centered_residual",
    "film",
    "bottleneck",
]

SINGLE_ARCHITECTURES = {
    "raw_only",
    "raw_larger",
    "stats_only",
    "gaf_only",
    "stft_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip/swarm plots of median macro_f1 by model family (one plot per dataset).",
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--plot-type", choices=("strip", "swarm"), default="strip")
    parser.add_argument("--overlay", choices=("violin", "box", "none"), default="violin")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def assign_family(architecture: str) -> str | None:
    method = fusion_method(architecture)
    if method == "minirocket_ridge":
        return None
    if method in SINGLE_ARCHITECTURES:
        return "single"
    if method == "concat":
        return "concat"
    if method == "gated":
        return "gated"
    if method == "raw_centered_residual":
        return "raw_centered_residual"
    if method == "film":
        return "film"
    if method in BOTTLENECK_METHODS:
        return "bottleneck"
    return None


def feature_config_label(architecture: str, modalities: str) -> str:
    method = fusion_method(architecture)
    if method in SINGLE_ARCHITECTURES:
        if method == "raw_only":
            return "raw"
        if method == "raw_larger":
            return "raw_larger"
        return method.replace("_only", "")
    if method in BOTTLENECK_METHODS:
        return method
    return str(modalities)


def build_plot_points(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["family"] = working["architecture"].map(assign_family)
    working = working[working["family"].notna()].copy()
    working["feature_config"] = working.apply(
        lambda row: feature_config_label(row["architecture"], row["modalities"]),
        axis=1,
    )

    points = (
        working.groupby(["dataset", "family", "architecture", "modalities", "feature_config"], dropna=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            median_macro_f1=("macro_f1", "median"),
            std_macro_f1=("macro_f1", "std"),
            n_seeds=("macro_f1", "count"),
        )
        .reset_index()
    )
    family_stats = (
        points.groupby(["dataset", "family"], observed=True)["mean_macro_f1"]
        .agg(
            q25_macro_f1_by_family=lambda s: s.quantile(0.25),
            q75_macro_f1_by_family=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    family_stats["iqr_macro_f1_by_family"] = (
        family_stats["q75_macro_f1_by_family"] - family_stats["q25_macro_f1_by_family"]
    )
    points = points.merge(family_stats, on=["dataset", "family"], how="left")
    points["family"] = pd.Categorical(points["family"], categories=FAMILY_ORDER, ordered=True)
    return points.sort_values(["dataset", "family", "mean_macro_f1"], ascending=[True, True, False])


def build_family_vs_best_single_table(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["family"] = working["architecture"].map(assign_family)
    working = working[working["family"].notna()].copy()

    single_baselines = (
        working.loc[working["family"] == "single"]
        .groupby("dataset", dropna=False)["macro_f1"]
        .agg(
            single_median_f1="median",
            single_q75_f1=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    working = working.merge(single_baselines, on="dataset", how="left")
    working["above_single_median"] = working["macro_f1"] > working["single_median_f1"]

    table = (
        working.groupby(["dataset", "family"], observed=True)
        .agg(
            median_f1=("macro_f1", "median"),
            q75_f1=("macro_f1", lambda s: s.quantile(0.75)),
            single_median_f1=("single_median_f1", "first"),
            single_q75_f1=("single_q75_f1", "first"),
            share_above_single_median=("above_single_median", "mean"),
        )
        .reset_index()
    )
    table["delta_vs_single_median"] = table["median_f1"] - table["single_median_f1"]
    table["delta_vs_single_q75"] = table["q75_f1"] - table["single_q75_f1"]
    table["family"] = pd.Categorical(table["family"], categories=FAMILY_ORDER, ordered=True)
    return table.sort_values(["dataset", "family"]).loc[
        :,
        [
            "dataset",
            "family",
            "median_f1",
            "single_median_f1",
            "delta_vs_single_median",
            "q75_f1",
            "single_q75_f1",
            "delta_vs_single_q75",
            "share_above_single_median",
        ],
    ]


def build_top3_by_dataset_table(family_vs_best_single: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        family_vs_best_single.sort_values(["dataset", "median_f1"], ascending=[True, False])
        .groupby("dataset", observed=True)
        .head(3)
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.groupby("dataset", observed=True).cumcount() + 1
    return ranked.loc[
        :,
        ["dataset", "rank", "family", "median_f1", "delta_vs_single_median", "share_above_single_median"],
    ]


def build_family_median_matrix(family_vs_best_single: pd.DataFrame) -> pd.DataFrame:
    datasets = sorted(family_vs_best_single["dataset"].unique())
    matrix = family_vs_best_single.pivot(index="dataset", columns="family", values="median_f1")
    return matrix.reindex(index=datasets, columns=FAMILY_ORDER)


def plot_all_datasets_heatmap(
    family_vs_best_single: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> Path:
    matrix = build_family_median_matrix(family_vs_best_single)
    values = matrix.to_numpy().ravel()
    values = values[~np.isnan(values)]
    vmin = float(values.min())
    vmax = float(values.max())

    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(
        figsize=(max(9, len(FAMILY_ORDER) * 1.3), max(4.8, 0.55 * len(matrix.index) + 2)),
        constrained_layout=True,
    )

    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 10},
        cbar_kws={"label": "Median macro_f1"},
    )

    for row_idx, dataset in enumerate(matrix.index):
        row = matrix.loc[dataset]
        col_idx = row.idxmax()
        family_idx = FAMILY_ORDER.index(col_idx)
        ax.add_patch(
            patches.Rectangle(
                (family_idx, row_idx),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.5,
            )
        )

    ax.set_title("Median macro_f1 by family across datasets")
    ax.set_xlabel("Model family")
    ax.set_ylabel("Dataset")
    ax.tick_params(axis="x", rotation=25, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "family_median_all_datasets_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset(
    points: pd.DataFrame,
    dataset_name: str,
    *,
    output_dir: Path,
    plot_type: str,
    overlay: str,
    dpi: int,
) -> Path:
    subset = points[points["dataset"] == dataset_name].copy()
    if subset.empty:
        raise ValueError(f"No plot points for dataset {dataset_name}")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(max(10, len(FAMILY_ORDER) * 1.4), 6))

    if overlay == "violin":
        sns.violinplot(
            data=subset,
            x="family",
            y="median_macro_f1",
            order=FAMILY_ORDER,
            inner=None,
            cut=0,
            linewidth=1,
            color="#DDDDDD",
            ax=ax,
        )
    elif overlay == "box":
        sns.boxplot(
            data=subset,
            x="family",
            y="median_macro_f1",
            order=FAMILY_ORDER,
            width=0.35,
            showcaps=True,
            boxprops={"facecolor": "#EEEEEE", "alpha": 0.8},
            whiskerprops={"linewidth": 1},
            medianprops={"color": "black", "linewidth": 1.5},
            ax=ax,
        )

    scatter_kwargs = {
        "data": subset,
        "x": "family",
        "y": "median_macro_f1",
        "order": FAMILY_ORDER,
        "size": 7,
        "alpha": 0.9,
        "ax": ax,
    }
    if plot_type == "swarm":
        sns.swarmplot(**scatter_kwargs, color="C0")
    else:
        sns.stripplot(**scatter_kwargs, jitter=0.25, color="C0")

    ax.set_title(f"{dataset_name}: median macro_f1 by family (point = feature config, aggregated over seeds)")
    ax.set_xlabel("Model family")
    ax.set_ylabel("Median macro_f1 over seeds")
    ax.set_ylim(max(0.0, subset["median_macro_f1"].min() - 0.05), min(1.0, subset["median_macro_f1"].max() + 0.05))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset_name}__family_{plot_type}plot.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> tuple[list[Path], Path, Path, Path]:
    df = read_summary(Path(args.summary))
    points = build_plot_points(df)
    family_vs_best_single = build_family_vs_best_single_table(df)
    top3_by_dataset = build_top3_by_dataset_table(family_vs_best_single)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(output_dir / "family_plot_points.csv", index=False)
    family_vs_best_single_path = output_dir / "family_vs_best_single.csv"
    family_vs_best_single.to_csv(family_vs_best_single_path, index=False)
    top3_path = output_dir / "family_top3_by_dataset.csv"
    top3_by_dataset.to_csv(top3_path, index=False)
    all_datasets_heatmap_path = plot_all_datasets_heatmap(
        family_vs_best_single,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    saved: list[Path] = []
    for dataset_name in sorted(points["dataset"].unique()):
        saved.append(
            plot_dataset(
                points,
                dataset_name,
                output_dir=output_dir,
                plot_type=args.plot_type,
                overlay=args.overlay,
                dpi=args.dpi,
            )
        )
    return saved, family_vs_best_single_path, top3_path, all_datasets_heatmap_path


if __name__ == "__main__":
    args = parse_args()
    paths, family_vs_best_single_path, top3_path, all_datasets_heatmap_path = run(args)
    print("Saved plots:")
    for path in paths:
        print(f"- {path}")
    print(f"- {all_datasets_heatmap_path}")
    print(f"- {family_vs_best_single_path}")
    print(f"- {top3_path}")
