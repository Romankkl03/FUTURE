from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def add_project_paths() -> None:
    for root in [Path.cwd(), *Path.cwd().parents]:
        src_dir = root / "src"
        if (root / "pyproject.toml").is_file() and src_dir.is_dir():
            for path in (str(root), str(src_dir)):
                if path not in sys.path:
                    sys.path.insert(0, path)
            return
    raise RuntimeError("Could not find project root with pyproject.toml and src/")


add_project_paths()

from experiments.architecture_evaluating import (
    DATASET_NAMES,
    GAF_PARAMS,
    METRIC_NAMES,
    MTF_PARAMS,
    STAT_PARAMS,
    STFT_PARAMS,
    DatasetContext,
    build_dataset_context,
    build_result,
    fit_and_evaluate,
    result_file_name,
)
from experiments.tools import make_json_safe, make_summary_row, save_json_result, save_summary_csv
from src.data import prepare_image_tensors, prepare_stat_features
from src.image_transformation.methods.gaf_transformation import GAF
from src.image_transformation.methods.mtf_transformation import MTF
from src.image_transformation.methods.stft_transformation import STFTSpectrogram
from src.models import (
    RawStatsGAFConcatClassifier,
    RawStatsGAFMTFSTFTConcatClassifier,
    RawStatsGAFSTFTConcatClassifier,
    RawStatsSTFTConcatClassifier,
)
from src.tools import get_device, set_seed


RESULT_DIR = Path("results/representation_concat_results")
BASELINE_MTF_METHOD = "raw_stats_GAF_STFT"
WITH_MTF_METHOD = "raw_stats_GAF_MTF_STFT"


@dataclass(frozen=True)
class PreparedRepresentations:
    F_train: torch.Tensor
    F_test: torch.Tensor
    stat_in_features: int
    stat_feature_names: list[str]
    dropped_feature_names: list[str]
    X_train_gaf: torch.Tensor
    X_test_gaf: torch.Tensor
    gaf_transform_shape: tuple[int, ...]
    gaf_cnn_shape: tuple[int, ...]
    X_train_mtf: torch.Tensor
    X_test_mtf: torch.Tensor
    mtf_transform_shape: tuple[int, ...]
    mtf_cnn_shape: tuple[int, ...]
    X_train_stft: torch.Tensor
    X_test_stft: torch.Tensor
    stft_transform_shape: tuple[int, ...]
    stft_cnn_shape: tuple[int, ...]


@dataclass(frozen=True)
class ConcatSpec:
    method: str
    classifier_cls: type[nn.Module]
    tensor_builder: Callable[[DatasetContext, PreparedRepresentations], tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]
    image_shape_key: str | None = None


def multi_input_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    *inputs, y = batch
    return tuple(x.to(device) for x in inputs), y.to(device)


def make_loaders(
    train_tensors: tuple[torch.Tensor, ...],
    test_tensors: tuple[torch.Tensor, ...],
    context: DatasetContext,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(*train_tensors, context.y_train),
        batch_size=context.params["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(*test_tensors, context.y_test),
        batch_size=context.params["batch_size"],
        shuffle=False,
    )
    return train_loader, test_loader


def prepare_representations(context: DatasetContext) -> PreparedRepresentations:
    F_train, F_test, feature_names, dropped_feature_names = prepare_stat_features(
        context.X_train_2d,
        context.X_test_2d,
        STAT_PARAMS,
        batch_size=context.feature_batch_size,
    )
    X_train_gaf, X_test_gaf, gaf_transform_shape, gaf_cnn_shape = prepare_image_tensors(
        GAF(GAF_PARAMS),
        context.X_train_2d,
        context.X_test_2d,
        image_batch_size=context.image_batch_size,
    )
    X_train_mtf, X_test_mtf, mtf_transform_shape, mtf_cnn_shape = prepare_image_tensors(
        MTF(MTF_PARAMS),
        context.X_train_2d,
        context.X_test_2d,
        image_batch_size=context.image_batch_size,
    )
    X_train_stft, X_test_stft, stft_transform_shape, stft_cnn_shape = prepare_image_tensors(
        STFTSpectrogram(STFT_PARAMS),
        context.X_train_2d,
        context.X_test_2d,
        image_batch_size=context.image_batch_size,
        use_log1p=True,
    )
    return PreparedRepresentations(
        F_train=F_train,
        F_test=F_test,
        stat_in_features=F_train.shape[1],
        stat_feature_names=feature_names,
        dropped_feature_names=dropped_feature_names,
        X_train_gaf=X_train_gaf,
        X_test_gaf=X_test_gaf,
        gaf_transform_shape=gaf_transform_shape,
        gaf_cnn_shape=gaf_cnn_shape,
        X_train_mtf=X_train_mtf,
        X_test_mtf=X_test_mtf,
        mtf_transform_shape=mtf_transform_shape,
        mtf_cnn_shape=mtf_cnn_shape,
        X_train_stft=X_train_stft,
        X_test_stft=X_test_stft,
        stft_transform_shape=stft_transform_shape,
        stft_cnn_shape=stft_cnn_shape,
    )


def raw_stats_stft_tensors(
    context: DatasetContext,
    reps: PreparedRepresentations,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    return (
        (context.X_train, reps.F_train, reps.X_train_stft),
        (context.X_test, reps.F_test, reps.X_test_stft),
    )


def raw_stats_gaf_tensors(
    context: DatasetContext,
    reps: PreparedRepresentations,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    return (
        (context.X_train, reps.F_train, reps.X_train_gaf),
        (context.X_test, reps.F_test, reps.X_test_gaf),
    )


def raw_stats_gaf_stft_tensors(
    context: DatasetContext,
    reps: PreparedRepresentations,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    return (
        (context.X_train, reps.F_train, reps.X_train_gaf, reps.X_train_stft),
        (context.X_test, reps.F_test, reps.X_test_gaf, reps.X_test_stft),
    )


def raw_stats_gaf_mtf_stft_tensors(
    context: DatasetContext,
    reps: PreparedRepresentations,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    return (
        (context.X_train, reps.F_train, reps.X_train_gaf, reps.X_train_mtf, reps.X_train_stft),
        (context.X_test, reps.F_test, reps.X_test_gaf, reps.X_test_mtf, reps.X_test_stft),
    )


def image_shapes_for_method(spec: ConcatSpec, reps: PreparedRepresentations) -> dict[str, Any]:
    shapes = {
        "raw_stats_STFT": {
            "stft_transform_shape": reps.stft_transform_shape,
            "stft_cnn_shape": reps.stft_cnn_shape,
        },
        "raw_stats_GAF": {
            "gaf_transform_shape": reps.gaf_transform_shape,
            "gaf_cnn_shape": reps.gaf_cnn_shape,
        },
        "raw_stats_GAF_STFT": {
            "gaf_transform_shape": reps.gaf_transform_shape,
            "gaf_cnn_shape": reps.gaf_cnn_shape,
            "stft_transform_shape": reps.stft_transform_shape,
            "stft_cnn_shape": reps.stft_cnn_shape,
        },
        "raw_stats_GAF_MTF_STFT": {
            "gaf_transform_shape": reps.gaf_transform_shape,
            "gaf_cnn_shape": reps.gaf_cnn_shape,
            "mtf_transform_shape": reps.mtf_transform_shape,
            "mtf_cnn_shape": reps.mtf_cnn_shape,
            "stft_transform_shape": reps.stft_transform_shape,
            "stft_cnn_shape": reps.stft_cnn_shape,
        },
    }
    return shapes[spec.method]


def build_model(
    spec: ConcatSpec,
    context: DatasetContext,
    reps: PreparedRepresentations,
) -> nn.Module:
    return spec.classifier_cls(
        raw_in_channels=context.params["channels"],
        stat_in_features=reps.stat_in_features,
        image_in_channels=1,
        num_classes=context.params["num_classes"],
    )


def run_concat_spec(
    context: DatasetContext,
    reps: PreparedRepresentations,
    spec: ConcatSpec,
) -> dict[str, Any]:
    train_tensors, test_tensors = spec.tensor_builder(context, reps)
    train_loader, test_loader = make_loaders(train_tensors, test_tensors, context)
    model = build_model(spec, context, reps)

    started_at = time.perf_counter()
    epoch_losses, metrics = fit_and_evaluate(
        model,
        train_loader,
        test_loader,
        context,
        batch_adapter=multi_input_batch,
    )
    return build_result(
        context,
        method=spec.method,
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={
            "modalities": spec.method.split("_"),
            "fusion": "concat",
            "stat_params": STAT_PARAMS,
            "stat_in_features": reps.stat_in_features,
            "stat_feature_names": reps.stat_feature_names,
            "dropped_feature_names": reps.dropped_feature_names,
            "transform_params": {
                "GAF": GAF_PARAMS,
                "MTF": MTF_PARAMS,
                "STFT": STFT_PARAMS,
            },
            **image_shapes_for_method(spec, reps),
        },
    )


def make_concat_specs() -> list[ConcatSpec]:
    return [
        ConcatSpec("raw_stats_STFT", RawStatsSTFTConcatClassifier, raw_stats_stft_tensors),
        ConcatSpec("raw_stats_GAF", RawStatsGAFConcatClassifier, raw_stats_gaf_tensors),
        ConcatSpec("raw_stats_GAF_STFT", RawStatsGAFSTFTConcatClassifier, raw_stats_gaf_stft_tensors),
        ConcatSpec(
            "raw_stats_GAF_MTF_STFT",
            RawStatsGAFMTFSTFTConcatClassifier,
            raw_stats_gaf_mtf_stft_tensors,
        ),
    ]


def make_mtf_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset_method = {(row["dataset_name"], row["method"]): row for row in rows}
    comparison_rows: list[dict[str, Any]] = []
    for dataset_name in sorted({row["dataset_name"] for row in rows}):
        baseline = by_dataset_method.get((dataset_name, BASELINE_MTF_METHOD))
        with_mtf = by_dataset_method.get((dataset_name, WITH_MTF_METHOD))
        if baseline is None or with_mtf is None:
            continue

        comparison = {
            "dataset_name": dataset_name,
            "dataset_size": baseline["total_size"],
            "baseline_method": BASELINE_MTF_METHOD,
            "with_mtf_method": WITH_MTF_METHOD,
        }
        for metric in METRIC_NAMES:
            comparison[f"baseline_{metric}"] = baseline[metric]
            comparison[f"with_mtf_{metric}"] = with_mtf[metric]
            comparison[f"delta_{metric}"] = with_mtf[metric] - baseline[metric]
        comparison_rows.append(comparison)
    return comparison_rows


def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    specs = make_concat_specs()
    for dataset_name in args.datasets:
        context = build_dataset_context(
            dataset_name,
            device=device,
            max_batch_size=args.max_batch_size,
            n_epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            feature_batch_size=args.feature_batch_size,
            image_batch_size=args.image_batch_size,
            use_cache=args.use_cache,
        )
        reps = prepare_representations(context)

        for spec in specs:
            print(f"[{dataset_name}] running {spec.method}")
            set_seed(args.seed)
            result = run_concat_spec(context, reps, spec)
            save_json_result(result, output_dir / result_file_name(dataset_name, spec.method))
            rows.append(make_summary_row(result, METRIC_NAMES))
            save_summary_csv(rows, output_dir / "summary.csv")
            save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")

            comparison_rows = make_mtf_comparison(rows)
            if comparison_rows:
                save_summary_csv(comparison_rows, output_dir / "mtf_comparison.csv")
                save_json_result({"rows": make_json_safe(comparison_rows)}, output_dir / "mtf_comparison.json")

    comparison_rows = make_mtf_comparison(rows)
    if comparison_rows:
        save_summary_csv(comparison_rows, output_dir / "mtf_comparison.csv")
        save_json_result({"rows": make_json_safe(comparison_rows)}, output_dir / "mtf_comparison.json")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-representation concat classifiers.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
