from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


from experiments.tools import FUSION_OVER_RAW_SUMMARY_CSV, FUSION_OVER_RAW_TABLES_DIR


DEFAULT_SUMMARY_PATH = FUSION_OVER_RAW_SUMMARY_CSV
DEFAULT_OUTPUT_DIR = FUSION_OVER_RAW_TABLES_DIR
STRICT_THRESHOLD = 0.005
PRACTICAL_THRESHOLD = 0.01

RAW_CENTERED_METHODS = {
    "film",
    "raw_centered_residual",
    "raw_residual_bottleneck",
    "context_only_residual_bottleneck",
    "raw_conditioned_context_bottleneck",
}
NON_RAW_CENTERED_METHODS = {
    "concat",
    "gated",
    "ordinary_bottleneck",
}
BOTTLENECK_METHODS = {
    "ordinary_bottleneck",
    "raw_residual_bottleneck",
    "context_only_residual_bottleneck",
    "raw_conditioned_context_bottleneck",
}
SINGLE_CONTEXT_MODALITIES = ("raw+stats", "raw+gaf", "raw+stft")
FULL_CONTEXT_MODALITIES = "raw+stats+gaf+stft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fusion-over-raw experiment results.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--strict-threshold", type=float, default=STRICT_THRESHOLD)
    parser.add_argument("--practical-threshold", type=float, default=PRACTICAL_THRESHOLD)
    parser.add_argument("--large-min-total-size", type=int, default=5_000)
    parser.add_argument("--high-raw-threshold", type=float, default=0.90)
    parser.add_argument("--low-raw-threshold", type=float, default=0.75)
    return parser.parse_args()


def fusion_method(architecture: str) -> str:
    if "__" in architecture:
        return architecture.split("__", 1)[0]
    return architecture


def architecture_family(method: str) -> str:
    if method in RAW_CENTERED_METHODS:
        return "raw_centered"
    if method in NON_RAW_CENTERED_METHODS:
        return "non_raw_centered"
    if method in {"raw_only", "raw_larger", "stats_only", "gaf_only", "stft_only"}:
        return "single_view"
    if method == "minirocket_ridge":
        return "external_baseline"
    return "other"


def context_representation(modalities: str) -> str:
    parts = [part for part in str(modalities).split("+") if part != "raw"]
    return "+".join(parts) if parts else "none"


def read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = [
        "seed",
        "train_size",
        "val_size",
        "test_size",
        "num_classes",
        "num_params",
        "best_epoch",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "delta_macro_f1_vs_raw",
        "delta_macro_f1_vs_raw_larger",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["dataset"] = df["dataset"].astype(str)
    df["architecture"] = df["architecture"].astype(str)
    df["modalities"] = df["modalities"].astype(str)
    df["fusion_method"] = df["architecture"].map(fusion_method)
    df["architecture_family"] = df["fusion_method"].map(architecture_family)
    df["context_representation"] = df["modalities"].map(context_representation)
    df["total_size"] = df["train_size"] + df["val_size"] + df["test_size"]
    return df


def save_table(table: pd.DataFrame, output_dir: Path, name: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"
    table.to_csv(csv_path, index=False)
    table.to_json(json_path, orient="records", indent=2, force_ascii=False)
    return {"csv": str(csv_path), "json": str(json_path)}


def flatten_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in table.columns
    ]
    return table


def make_basic_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(["dataset", "architecture", "modalities"], dropna=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_weighted_f1=("weighted_f1", "mean"),
            mean_delta_macro_f1_vs_raw=("delta_macro_f1_vs_raw", "mean"),
            mean_delta_macro_f1_vs_raw_larger=("delta_macro_f1_vs_raw_larger", "mean"),
            mean_num_params=("num_params", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            n_runs=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_macro_f1"], ascending=[True, False])
    )
    return table


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rank"] = ranked.groupby(["dataset", "seed"])["macro_f1"].rank(
        ascending=False,
        method="average",
    )
    return ranked


def make_ranking_tables(
    ranked: pd.DataFrame,
    *,
    large_min_total_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = ranked.copy()
    ranked["is_large_dataset"] = ranked["total_size"] >= large_min_total_size
    ranked["is_multiclass_dataset"] = ranked["num_classes"] > 2

    group_cols = ["architecture", "modalities", "fusion_method", "architecture_family"]
    overall = ranked.groupby(group_cols, dropna=False).agg(
        average_rank_all=("rank", "mean"),
        std_rank_all=("rank", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        n_ranked_runs=("rank", "count"),
    )
    large = ranked[ranked["is_large_dataset"]].groupby(group_cols, dropna=False).agg(
        average_rank_large_datasets=("rank", "mean"),
        n_large_ranked_runs=("rank", "count"),
    )
    multiclass = ranked[ranked["is_multiclass_dataset"]].groupby(group_cols, dropna=False).agg(
        average_rank_multiclass_datasets=("rank", "mean"),
        n_multiclass_ranked_runs=("rank", "count"),
    )
    table = (
        overall.join(large, how="left")
        .join(multiclass, how="left")
        .reset_index()
        .sort_values("average_rank_all")
    )

    by_dataset = (
        ranked.groupby(["dataset", *group_cols], dropna=False)
        .agg(
            average_rank=("rank", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            n_ranked_runs=("rank", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "average_rank"])
    )
    return table, by_dataset


def classify_wtl(delta: float, threshold: float) -> str:
    if pd.isna(delta):
        return "missing"
    if delta > threshold:
        return "win"
    if delta < -threshold:
        return "loss"
    return "tie"


def make_wtl_table(
    df: pd.DataFrame,
    *,
    delta_column: str,
    threshold: float,
    level: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["architecture", "modalities", "fusion_method", "architecture_family"]

    if level == "dataset":
        working = (
            df.groupby(["dataset", *group_cols], dropna=False)
            .agg(delta=(delta_column, "mean"))
            .reset_index()
        )
    elif level == "seed":
        working = df[[*group_cols, delta_column]].rename(columns={delta_column: "delta"}).copy()
    else:
        raise ValueError(f"Unknown win/tie/loss level: {level}")

    working["outcome"] = working["delta"].map(lambda value: classify_wtl(value, threshold))
    counts = (
        working.pivot_table(
            index=group_cols,
            columns="outcome",
            values="delta",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    for column in ("win", "tie", "loss", "missing"):
        if column not in counts.columns:
            counts[column] = 0
    counts["n_compared"] = counts["win"] + counts["tie"] + counts["loss"]
    counts["win_rate"] = counts["win"] / counts["n_compared"].replace(0, np.nan)
    counts["loss_rate"] = counts["loss"] / counts["n_compared"].replace(0, np.nan)
    counts["threshold"] = threshold
    counts["comparison_level"] = level
    counts["baseline"] = delta_column.replace("delta_macro_f1_vs_", "")
    return counts.sort_values(["win", "loss"], ascending=[False, True])


def make_stability_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_dataset = (
        df.groupby(["dataset", "architecture", "modalities"], dropna=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_minus_std=("macro_f1", lambda values: values.mean() - values.std(ddof=1)),
            n_seeds=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_minus_std"], ascending=[True, False])
    )
    overall = (
        by_dataset.groupby(["architecture", "modalities"], dropna=False)
        .agg(
            mean_macro_f1=("mean_macro_f1", "mean"),
            mean_seed_std_macro_f1=("std_macro_f1", "mean"),
            mean_minus_std=("mean_minus_std", "mean"),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
        .sort_values("mean_minus_std", ascending=False)
    )
    return overall, by_dataset


def make_modality_contribution_tables(df: pd.DataFrame, *, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    two_view = df[
        df["modalities"].isin(SINGLE_CONTEXT_MODALITIES)
        & df["fusion_method"].isin({"concat", "gated", "film", "raw_centered_residual"})
    ].copy()

    by_dataset = (
        two_view.groupby(["dataset", "fusion_method", "context_representation"], dropna=False)
        .agg(
            mean_delta_vs_raw=("delta_macro_f1_vs_raw", "mean"),
            mean_delta_vs_raw_larger=("delta_macro_f1_vs_raw_larger", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            n_runs=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "fusion_method", "mean_delta_vs_raw"], ascending=[True, True, False])
    )
    by_dataset["helps_vs_raw"] = by_dataset["mean_delta_vs_raw"] > threshold
    by_dataset["hurts_vs_raw"] = by_dataset["mean_delta_vs_raw"] < -threshold

    summary = (
        by_dataset.groupby(["fusion_method", "context_representation"], dropna=False)
        .agg(
            mean_delta_vs_raw=("mean_delta_vs_raw", "mean"),
            mean_delta_vs_raw_larger=("mean_delta_vs_raw_larger", "mean"),
            datasets_helped_vs_raw=("helps_vs_raw", "sum"),
            datasets_hurt_vs_raw=("hurts_vs_raw", "sum"),
            n_datasets=("dataset", "count"),
        )
        .reset_index()
        .sort_values(["fusion_method", "mean_delta_vs_raw"], ascending=[True, False])
    )
    summary["help_rate_vs_raw"] = summary["datasets_helped_vs_raw"] / summary["n_datasets"].replace(0, np.nan)
    summary["hurt_rate_vs_raw"] = summary["datasets_hurt_vs_raw"] / summary["n_datasets"].replace(0, np.nan)
    return summary, by_dataset


def make_full_context_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df[df["fusion_method"].isin({"concat", "gated", "film", "raw_centered_residual"})].copy()
    rows: list[dict[str, Any]] = []
    for (dataset, seed, fusion), group in working.groupby(["dataset", "seed", "fusion_method"], dropna=False):
        full = group[group["modalities"] == FULL_CONTEXT_MODALITIES]
        singles = group[group["modalities"].isin(SINGLE_CONTEXT_MODALITIES)]
        if full.empty or singles.empty:
            continue
        best_single = singles.sort_values("macro_f1", ascending=False).iloc[0]
        full_row = full.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "fusion_method": fusion,
                "full_context_macro_f1": full_row["macro_f1"],
                "best_single_context_macro_f1": best_single["macro_f1"],
                "best_single_context_modalities": best_single["modalities"],
                "delta_full_vs_best_single": full_row["macro_f1"] - best_single["macro_f1"],
            }
        )
    by_run = pd.DataFrame(rows)
    if by_run.empty:
        return by_run, by_run
    summary = (
        by_run.groupby("fusion_method", dropna=False)
        .agg(
            mean_delta_full_vs_best_single=("delta_full_vs_best_single", "mean"),
            std_delta_full_vs_best_single=("delta_full_vs_best_single", "std"),
            full_context_wins=("delta_full_vs_best_single", lambda values: (values > 0).sum()),
            full_context_losses=("delta_full_vs_best_single", lambda values: (values < 0).sum()),
            n_runs=("delta_full_vs_best_single", "count"),
        )
        .reset_index()
        .sort_values("mean_delta_full_vs_best_single", ascending=False)
    )
    return summary, by_run.sort_values(["fusion_method", "dataset", "seed"])


def make_family_comparison_tables(
    ranked: pd.DataFrame,
    df: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fusion_only = ranked[ranked["architecture_family"].isin({"raw_centered", "non_raw_centered"})].copy()
    rank_table = (
        fusion_only.groupby("architecture_family", dropna=False)
        .agg(
            average_rank=("rank", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_delta_vs_raw=("delta_macro_f1_vs_raw", "mean"),
            mean_delta_vs_raw_larger=("delta_macro_f1_vs_raw_larger", "mean"),
            n_runs=("rank", "count"),
        )
        .reset_index()
        .sort_values("average_rank")
    )
    wtl = make_wtl_table(
        df[df["architecture_family"].isin({"raw_centered", "non_raw_centered"})],
        delta_column="delta_macro_f1_vs_raw",
        threshold=threshold,
        level="dataset",
        group_cols=["architecture_family"],
    )
    return rank_table, wtl


def make_bottleneck_tables(
    ranked: pd.DataFrame,
    df: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bottleneck_ranked = ranked[ranked["fusion_method"].isin(BOTTLENECK_METHODS)].copy()
    rank_table = (
        bottleneck_ranked.groupby(["fusion_method", "modalities"], dropna=False)
        .agg(
            average_rank=("rank", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_delta_vs_raw=("delta_macro_f1_vs_raw", "mean"),
            mean_delta_vs_raw_larger=("delta_macro_f1_vs_raw_larger", "mean"),
            n_runs=("rank", "count"),
        )
        .reset_index()
        .sort_values("average_rank")
    )
    wtl = make_wtl_table(
        df[df["fusion_method"].isin(BOTTLENECK_METHODS)],
        delta_column="delta_macro_f1_vs_raw",
        threshold=threshold,
        level="dataset",
        group_cols=["fusion_method", "modalities"],
    )
    return rank_table, wtl


def make_dataset_difficulty_tables(
    df: pd.DataFrame,
    *,
    high_threshold: float,
    low_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = (
        df[df["architecture"] == "raw_only"]
        .groupby("dataset", dropna=False)
        .agg(
            raw_macro_f1=("macro_f1", "mean"),
            raw_macro_f1_std=("macro_f1", "std"),
            num_classes=("num_classes", "max"),
            total_size=("total_size", "max"),
        )
        .reset_index()
    )
    raw["difficulty_group"] = np.select(
        [raw["raw_macro_f1"] >= high_threshold, raw["raw_macro_f1"] < low_threshold],
        ["high_raw_baseline", "low_raw_baseline"],
        default="medium_raw_baseline",
    )

    fusion = df[df["architecture_family"].isin({"raw_centered", "non_raw_centered"})].merge(
        raw[["dataset", "raw_macro_f1", "difficulty_group"]],
        on="dataset",
        how="left",
    )
    by_difficulty = (
        fusion.groupby(["difficulty_group", "fusion_method", "modalities"], dropna=False)
        .agg(
            mean_delta_vs_raw=("delta_macro_f1_vs_raw", "mean"),
            mean_delta_vs_raw_larger=("delta_macro_f1_vs_raw_larger", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            n_runs=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values(["difficulty_group", "mean_delta_vs_raw"], ascending=[True, False])
    )
    return raw.sort_values("raw_macro_f1", ascending=False), by_difficulty


def make_parameter_efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    raw_params = (
        df[df["architecture"] == "raw_only"][["dataset", "seed", "num_params"]]
        .rename(columns={"num_params": "raw_num_params"})
    )
    raw_larger_params = (
        df[df["architecture"] == "raw_larger"][["dataset", "seed", "num_params"]]
        .rename(columns={"num_params": "raw_larger_num_params"})
    )
    working = df.merge(raw_params, on=["dataset", "seed"], how="left").merge(
        raw_larger_params,
        on=["dataset", "seed"],
        how="left",
    )
    working["extra_params_vs_raw"] = working["num_params"] - working["raw_num_params"]
    working["extra_params_vs_raw_larger"] = working["num_params"] - working["raw_larger_num_params"]
    working["delta_vs_raw_per_100k_extra_params"] = np.where(
        working["extra_params_vs_raw"] > 0,
        working["delta_macro_f1_vs_raw"] / (working["extra_params_vs_raw"] / 100_000),
        np.nan,
    )
    working["delta_vs_raw_larger_per_100k_extra_params"] = np.where(
        working["extra_params_vs_raw_larger"] > 0,
        working["delta_macro_f1_vs_raw_larger"] / (working["extra_params_vs_raw_larger"] / 100_000),
        np.nan,
    )

    table = (
        working.groupby(["architecture", "modalities", "fusion_method", "architecture_family"], dropna=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            mean_num_params=("num_params", "mean"),
            mean_extra_params_vs_raw=("extra_params_vs_raw", "mean"),
            mean_extra_params_vs_raw_larger=("extra_params_vs_raw_larger", "mean"),
            mean_delta_vs_raw=("delta_macro_f1_vs_raw", "mean"),
            mean_delta_vs_raw_larger=("delta_macro_f1_vs_raw_larger", "mean"),
            mean_delta_vs_raw_per_100k_extra_params=("delta_vs_raw_per_100k_extra_params", "mean"),
            mean_delta_vs_raw_larger_per_100k_extra_params=("delta_vs_raw_larger_per_100k_extra_params", "mean"),
            n_runs=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values("mean_delta_vs_raw_per_100k_extra_params", ascending=False, na_position="last")
    )
    return table


def pick_best_worst_modalities(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    by_modality = (
        df.groupby(["fusion_method", "modalities"], dropna=False)
        .agg(value=(value_column, "mean"))
        .reset_index()
    )
    rows = []
    for fusion, group in by_modality.groupby("fusion_method", dropna=False):
        best = group.sort_values("value", ascending=False).iloc[0]
        worst = group.sort_values("value", ascending=True).iloc[0]
        rows.append(
            {
                "fusion_method": fusion,
                "best_modality_combination": best["modalities"],
                "best_modality_value": best["value"],
                "worst_modality_combination": worst["modalities"],
                "worst_modality_value": worst["value"],
            }
        )
    return pd.DataFrame(rows)


def make_final_decision_table(
    df: pd.DataFrame,
    ranking: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    fusion_df = df[df["architecture_family"].isin({"raw_centered", "non_raw_centered"})].copy()
    rank_by_method = (
        ranking[ranking["architecture_family"].isin({"raw_centered", "non_raw_centered"})]
        .groupby("fusion_method", dropna=False)
        .agg(average_rank=("average_rank_all", "mean"))
        .reset_index()
    )
    base = (
        fusion_df.groupby(["fusion_method", "architecture_family"], dropna=False)
        .agg(
            mean_delta_vs_raw=("delta_macro_f1_vs_raw", "mean"),
            mean_delta_vs_raw_larger=("delta_macro_f1_vs_raw_larger", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_over_runs=("macro_f1", "std"),
            mean_num_params=("num_params", "mean"),
            n_runs=("macro_f1", "count"),
        )
        .reset_index()
    )
    stability_by_method = (
        stability.assign(fusion_method=stability["architecture"].map(fusion_method))
        .groupby("fusion_method", dropna=False)
        .agg(mean_seed_std_macro_f1=("mean_seed_std_macro_f1", "mean"))
        .reset_index()
    )
    modalities = pick_best_worst_modalities(fusion_df, "delta_macro_f1_vs_raw")
    wtl_raw = make_wtl_table(
        fusion_df,
        delta_column="delta_macro_f1_vs_raw",
        threshold=threshold,
        level="dataset",
        group_cols=["fusion_method"],
    ).rename(columns={"win": "wins_vs_raw", "tie": "ties_vs_raw", "loss": "losses_vs_raw"})
    wtl_larger = make_wtl_table(
        fusion_df,
        delta_column="delta_macro_f1_vs_raw_larger",
        threshold=threshold,
        level="dataset",
        group_cols=["fusion_method"],
    ).rename(
        columns={
            "win": "wins_vs_raw_larger",
            "tie": "ties_vs_raw_larger",
            "loss": "losses_vs_raw_larger",
        }
    )

    table = (
        base.merge(rank_by_method, on="fusion_method", how="left")
        .merge(stability_by_method, on="fusion_method", how="left")
        .merge(modalities, on="fusion_method", how="left")
        .merge(wtl_raw[["fusion_method", "wins_vs_raw", "ties_vs_raw", "losses_vs_raw"]], on="fusion_method", how="left")
        .merge(
            wtl_larger[["fusion_method", "wins_vs_raw_larger", "ties_vs_raw_larger", "losses_vs_raw_larger"]],
            on="fusion_method",
            how="left",
        )
        .sort_values("average_rank")
    )
    return table


def run(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    df = read_summary(summary_path)
    ranked = add_ranks(df)

    artifacts: dict[str, dict[str, str]] = {}
    artifacts["basic"] = save_table(make_basic_table(df), output_dir, "basic")

    ranking, ranking_by_dataset = make_ranking_tables(
        ranked,
        large_min_total_size=args.large_min_total_size,
    )
    artifacts["ranking"] = save_table(ranking, output_dir, "ranking")
    artifacts["ranking_by_dataset"] = save_table(ranking_by_dataset, output_dir, "ranking_by_dataset")

    for threshold_name, threshold in (
        ("strict_0_005", args.strict_threshold),
        ("practical_0_01", args.practical_threshold),
    ):
        for baseline, delta_column in (
            ("raw", "delta_macro_f1_vs_raw"),
            ("raw_larger", "delta_macro_f1_vs_raw_larger"),
        ):
            for level in ("seed", "dataset"):
                name = f"win_tie_loss_vs_{baseline}_{threshold_name}_{level}_level"
                artifacts[name] = save_table(
                    make_wtl_table(df, delta_column=delta_column, threshold=threshold, level=level),
                    output_dir,
                    name,
                )

    stability, stability_by_dataset = make_stability_tables(df)
    artifacts["stability"] = save_table(stability, output_dir, "stability")
    artifacts["stability_by_dataset"] = save_table(stability_by_dataset, output_dir, "stability_by_dataset")

    modality_summary, modality_by_dataset = make_modality_contribution_tables(
        df,
        threshold=args.strict_threshold,
    )
    artifacts["modality_contribution"] = save_table(modality_summary, output_dir, "modality_contribution")
    artifacts["modality_contribution_by_dataset"] = save_table(
        modality_by_dataset,
        output_dir,
        "modality_contribution_by_dataset",
    )

    full_summary, full_by_run = make_full_context_table(df)
    artifacts["full_context_vs_best_single_context"] = save_table(
        full_summary,
        output_dir,
        "full_context_vs_best_single_context",
    )
    artifacts["full_context_vs_best_single_context_by_run"] = save_table(
        full_by_run,
        output_dir,
        "full_context_vs_best_single_context_by_run",
    )

    family_rank, family_wtl = make_family_comparison_tables(
        ranked,
        df,
        threshold=args.strict_threshold,
    )
    artifacts["raw_centered_vs_non_raw_centered_ranking"] = save_table(
        family_rank,
        output_dir,
        "raw_centered_vs_non_raw_centered_ranking",
    )
    artifacts["raw_centered_vs_non_raw_centered_wtl"] = save_table(
        family_wtl,
        output_dir,
        "raw_centered_vs_non_raw_centered_wtl",
    )

    bottleneck_rank, bottleneck_wtl = make_bottleneck_tables(
        ranked,
        df,
        threshold=args.strict_threshold,
    )
    artifacts["bottleneck_variants_ranking"] = save_table(
        bottleneck_rank,
        output_dir,
        "bottleneck_variants_ranking",
    )
    artifacts["bottleneck_variants_wtl"] = save_table(
        bottleneck_wtl,
        output_dir,
        "bottleneck_variants_wtl",
    )

    difficulty, difficulty_effects = make_dataset_difficulty_tables(
        df,
        high_threshold=args.high_raw_threshold,
        low_threshold=args.low_raw_threshold,
    )
    artifacts["dataset_difficulty"] = save_table(difficulty, output_dir, "dataset_difficulty")
    artifacts["dataset_difficulty_fusion_effects"] = save_table(
        difficulty_effects,
        output_dir,
        "dataset_difficulty_fusion_effects",
    )

    artifacts["parameter_efficiency"] = save_table(
        make_parameter_efficiency_table(df),
        output_dir,
        "parameter_efficiency",
    )

    artifacts["final_decision_table"] = save_table(
        make_final_decision_table(
            df,
            ranking,
            stability,
            threshold=args.strict_threshold,
        ),
        output_dir,
        "final_decision_table",
    )

    manifest = {
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "n_rows": int(len(df)),
        "n_datasets": int(df["dataset"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "thresholds": {
            "strict": args.strict_threshold,
            "practical": args.practical_threshold,
        },
        "large_min_total_size": args.large_min_total_size,
        "raw_baseline_difficulty_thresholds": {
            "high": args.high_raw_threshold,
            "low": args.low_raw_threshold,
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    artifacts["manifest"] = {"json": str(manifest_path)}
    return artifacts


if __name__ == "__main__":
    produced = run(parse_args())
    print("Saved aggregate tables:")
    for name, paths in produced.items():
        print(f"- {name}: {paths}")
