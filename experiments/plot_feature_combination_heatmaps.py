from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.agregate_results import fusion_method, read_summary


DEFAULT_SUMMARY_PATH = Path("results/fusion_over_raw/summary.csv")
DEFAULT_OUTPUT_DIR = Path("results/fusion_over_raw/plots")

PANEL_FAMILIES = [
    "raw_centered_residual",
    "film",
]

MODALITY_ORDER = [
    "raw+stats",
    "raw+gaf",
    "raw+stft",
    "raw+stats+gaf",
    "raw+stats+stft",
    "raw+gaf+stft",
    "raw+stats+gaf+stft",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2-panel heatmaps: feature-combination sensitivity across datasets.",
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def format_modality_label(modalities: str) -> str:
    return modalities.replace("gaf", "GAF").replace("stft", "STFT")


def family_method_filter(family: str) -> set[str]:
    if family == "raw_centered_residual":
        return {"raw_centered_residual"}
    if family == "film":
        return {"film"}
    raise ValueError(f"Unknown family: {family}")


def build_family_matrix(
    df: pd.DataFrame,
    *,
    family: str,
    datasets: list[str],
) -> pd.DataFrame:
    methods = family_method_filter(family)
    subset = df[df["architecture"].map(lambda arch: fusion_method(arch) in methods)].copy()
    if subset.empty:
        raise ValueError(f"No rows for family {family}")

    per_combo = (
        subset.groupby(["dataset", "modalities"], dropna=False)["macro_f1"]
        .median()
        .reset_index(name="median_macro_f1")
    )

    matrix = per_combo.pivot(index="dataset", columns="modalities", values="median_macro_f1")
    matrix = matrix.reindex(index=datasets, columns=MODALITY_ORDER)
    matrix.index.name = "dataset"
    return matrix


def best_column_per_row(matrix: pd.DataFrame) -> pd.Series:
    best_cols: dict[str, str | None] = {}
    for dataset in matrix.index:
        row = matrix.loc[dataset]
        valid = row.dropna()
        if valid.empty:
            best_cols[dataset] = None
            continue
        best_cols[dataset] = valid.idxmax()
    return pd.Series(best_cols, name="best_modality")


def draw_heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    vmin: float,
    vmax: float,
    column_labels: list[str],
) -> None:
    display = matrix.copy()
    display.columns = column_labels
    mask = display.isna()

    sns.heatmap(
        display,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9},
    )

    best_cols = best_column_per_row(matrix)
    for row_idx, dataset in enumerate(matrix.index):
        best_modality = best_cols[dataset]
        if best_modality is None:
            continue
        col_idx = MODALITY_ORDER.index(best_modality)
        rect = patches.Rectangle(
            (col_idx, row_idx),
            1,
            1,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
        )
        ax.add_patch(rect)

    ax.set_title(title)
    ax.set_xlabel("Feature combinations")
    ax.set_ylabel("Dataset")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=10)


def build_long_table(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, matrix in matrices.items():
        best_cols = best_column_per_row(matrix)
        for dataset in matrix.index:
            for modality in MODALITY_ORDER:
                value = matrix.loc[dataset, modality]
                rows.append(
                    {
                        "family": family,
                        "dataset": dataset,
                        "modalities": modality,
                        "feature_combination": format_modality_label(modality),
                        "median_macro_f1": value,
                        "is_best_for_dataset": modality == best_cols[dataset],
                    }
                )
    table = pd.DataFrame(rows)
    table["family"] = pd.Categorical(table["family"], categories=PANEL_FAMILIES, ordered=True)
    return table.sort_values(["family", "dataset", "modalities"])


def plot_feature_combination_heatmaps(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    datasets = sorted(df["dataset"].unique())
    column_labels = [format_modality_label(modality) for modality in MODALITY_ORDER]

    matrices = {family: build_family_matrix(df, family=family, datasets=datasets) for family in PANEL_FAMILIES}
    values = np.concatenate([matrix.to_numpy().ravel() for matrix in matrices.values()])
    values = values[~np.isnan(values)]
    vmin = float(values.min())
    vmax = float(values.max())

    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(
        1,
        len(PANEL_FAMILIES),
        figsize=(6.2 * len(PANEL_FAMILIES), max(4.5, 0.55 * len(datasets) + 2)),
        constrained_layout=True,
    )
    if len(PANEL_FAMILIES) == 1:
        axes = [axes]

    for ax, family in zip(axes, PANEL_FAMILIES, strict=True):
        draw_heatmap_panel(
            ax,
            matrices[family],
            title=family,
            vmin=vmin,
            vmax=vmax,
            column_labels=column_labels,
        )

    fig.suptitle("Feature-combination sensitivity across datasets", y=1.02, fontsize=16)
    sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Median macro_f1 over seeds")

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "feature_combination_sensitivity.png"
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    table = build_long_table(matrices)
    table_path = output_dir / "feature_combination_sensitivity.csv"
    table.to_csv(table_path, index=False)
    return figure_path, table_path


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    df = read_summary(Path(args.summary))
    return plot_feature_combination_heatmaps(df, output_dir=Path(args.output_dir), dpi=args.dpi)


if __name__ == "__main__":
    args = parse_args()
    figure_path, table_path = run(args)
    print(f"- {figure_path}")
    print(f"- {table_path}")
