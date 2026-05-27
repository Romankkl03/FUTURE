from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def add_project_paths() -> None:
    for root in [Path.cwd(), *Path.cwd().parents]:
        src_dir = root / "src"
        if (root / "pyproject.toml").is_file() and src_dir.is_dir():
            import sys

            for path in (str(root), str(src_dir)):
                if path not in sys.path:
                    sys.path.insert(0, path)
            return
    raise RuntimeError("Could not find project root with pyproject.toml and src/")


add_project_paths()

from src.data import (  # noqa: E402
    as_univariate_feature_tensor,
    load_dataset,
    make_prepared_image_loaders,
    make_prepared_raw_image_loaders,
    make_prepared_raw_stats_loaders,
    make_prepared_stat_loaders,
    make_tensor_loaders,
)
from experiments.tools import (  # noqa: E402
    evaluate_with_batch_adapter,
    make_experiment_params,
    make_json_safe,
    make_summary_row,
    save_json_result,
    save_summary_csv,
)
from src.image_transformation.methods.gaf_transformation import GAF  # noqa: E402
from src.image_transformation.methods.mtf_transformation import MTF  # noqa: E402
from src.image_transformation.methods.stft_transformation import STFTSpectrogram  # noqa: E402
from src.models import (  # noqa: E402
    GAFClassifier,
    MTFClassifier,
    RawCNNClassifier,
    RawGAFConcatClassifier,
    RawMTFConcatClassifier,
    RawSTFTConcatClassifier,
    RawStatsConcatClassifier,
    STFTClassifier,
    StatisticalMLPClassifier,
    single_input_batch,
    train_with_batch_adapter,
    two_input_batch,
)
from src.tools import get_device, minmax_scale_pair, set_seed, z_normalize_pair  # noqa: E402


DATASET_NAMES = ("ECG5000", "MedicalImages", "FacesUCR", "SwedishLeaf")
RESULT_DIR = Path("results/architecture_results")

STAT_PARAMS = {
    "window_size": 12,
    "stride": 50,
    "add_global_features": True,
}

GAF_PARAMS = {
    "method": "summation",
    "overlapping": True,
    "image_size": 0.25,
    "sample_range": None,
}

MTF_PARAMS = {
    "image_size": 0.25,
    "n_bins": 8,
    "strategy": "quantile",
    "overlapping": True,
    "flatten": False,
}

STFT_PARAMS = {
    "window_size": 64,
    "hop_length": 16,
    "n_fft": 64,
    "window_type": "hann",
    "center": False,
    "pad_mode": "reflect",
    "power": 2.0,
    "normalized": False,
}

METRIC_NAMES = (
    "loss",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)


@dataclass(frozen=True)
class DatasetContext:
    dataset_name: str
    dataset: Any
    params: dict[str, Any]
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    X_train_2d: torch.Tensor
    X_test_2d: torch.Tensor
    X_train_scaled_2d: torch.Tensor
    X_test_scaled_2d: torch.Tensor
    device: torch.device
    feature_batch_size: int
    image_batch_size: int

    @property
    def train_size(self) -> int:
        return len(self.y_train)

    @property
    def test_size(self) -> int:
        return len(self.y_test)

    @property
    def total_size(self) -> int:
        return self.train_size + self.test_size


@dataclass(frozen=True)
class ImageSpec:
    transform_name: str
    transformer_cls: type
    transform_params: dict[str, Any]
    classifier_cls: type[nn.Module]
    raw_classifier_cls: type[nn.Module]
    input_getter: Callable[[DatasetContext], tuple[torch.Tensor, torch.Tensor]]
    use_log1p: bool = False


@dataclass(frozen=True)
class ArchitectureSpec:
    method: str
    runner: Callable[[DatasetContext], dict[str, Any]]


def build_dataset_context(
    dataset_name: str,
    *,
    device: torch.device,
    max_batch_size: int,
    n_epochs: int,
    lr: float,
    seed: int,
    feature_batch_size: int,
    image_batch_size: int,
    use_cache: bool,
) -> DatasetContext:
    dataset = load_dataset(dataset_name, use_cache=use_cache)
    params = make_experiment_params(
        dataset_name,
        dataset,
        max_batch_size=max_batch_size,
        n_epochs=n_epochs,
        lr=lr,
        seed=seed,
    )
    X_train_raw = torch.from_numpy(dataset.X_train).float()
    y_train = torch.from_numpy(dataset.y_train).long()
    X_test_raw = torch.from_numpy(dataset.X_test).float()
    y_test = torch.from_numpy(dataset.y_test).long()
    X_train, X_test = z_normalize_pair(X_train_raw, X_test_raw)
    X_train_2d = as_univariate_feature_tensor(X_train)
    X_test_2d = as_univariate_feature_tensor(X_test)
    X_train_scaled_2d, X_test_scaled_2d = minmax_scale_pair(X_train_2d, X_test_2d)

    return DatasetContext(
        dataset_name=dataset_name,
        dataset=dataset,
        params=params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_2d=X_train_2d,
        X_test_2d=X_test_2d,
        X_train_scaled_2d=X_train_scaled_2d,
        X_test_scaled_2d=X_test_scaled_2d,
        device=device,
        feature_batch_size=feature_batch_size,
        image_batch_size=image_batch_size,
    )


def fit_and_evaluate(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    context: DatasetContext,
    *,
    batch_adapter: Callable[[Any, torch.device], tuple[tuple[torch.Tensor, ...], torch.Tensor]],
) -> tuple[list[float], dict[str, Any]]:
    model = model.to(context.device)
    epoch_losses = train_with_batch_adapter(
        model,
        train_loader,
        context.device,
        n_epochs=context.params["n_epochs"],
        lr=context.params["lr"],
        batch_adapter=batch_adapter,
    )
    metrics = evaluate_with_batch_adapter(
        model,
        test_loader,
        context.device,
        batch_adapter=batch_adapter,
    )
    return epoch_losses, metrics


def build_result(
    context: DatasetContext,
    *,
    method: str,
    epoch_losses: list[float],
    metrics: dict[str, Any],
    duration_sec: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "method": method,
        "dataset_name": context.dataset_name,
        "train_size": context.train_size,
        "test_size": context.test_size,
        "total_size": context.total_size,
        "channels": context.params["channels"],
        "time_steps": context.params["time_steps"],
        "num_classes": context.params["num_classes"],
        "batch_size": context.params["batch_size"],
        "n_epochs": context.params["n_epochs"],
        "lr": context.params["lr"],
        "seed": context.params["seed"],
        "duration_sec": duration_sec,
        "epoch_losses": epoch_losses,
    }
    if metadata:
        result.update(metadata)
    result.update(metrics)
    return result


def run_raw(context: DatasetContext) -> dict[str, Any]:
    train_loader, test_loader = make_tensor_loaders(
        context.X_train,
        context.y_train,
        context.X_test,
        context.y_test,
        batch_size=context.params["batch_size"],
    )
    model = RawCNNClassifier(
        in_channels=context.params["channels"],
        num_classes=context.params["num_classes"],
    )
    started_at = time.perf_counter()
    epoch_losses, metrics = fit_and_evaluate(
        model,
        train_loader,
        test_loader,
        context,
        batch_adapter=single_input_batch,
    )
    return build_result(
        context,
        method="raw",
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={"modalities": ["raw"], "encoder": "1D CNN"},
    )


def run_stats(context: DatasetContext) -> dict[str, Any]:
    train_loader, test_loader, stat_in_features, feature_names, dropped_feature_names = make_prepared_stat_loaders(
        context.X_train_2d,
        context.X_test_2d,
        context.y_train,
        context.y_test,
        STAT_PARAMS,
        batch_size=context.params["batch_size"],
        feature_batch_size=context.feature_batch_size,
    )
    model = StatisticalMLPClassifier(
        in_features=stat_in_features,
        num_classes=context.params["num_classes"],
    )
    started_at = time.perf_counter()
    epoch_losses, metrics = fit_and_evaluate(
        model,
        train_loader,
        test_loader,
        context,
        batch_adapter=single_input_batch,
    )
    return build_result(
        context,
        method="stats",
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={
            "modalities": ["stats"],
            "encoder": "MLP",
            "stat_params": STAT_PARAMS,
            "stat_in_features": stat_in_features,
            "stat_feature_names": feature_names,
            "dropped_feature_names": dropped_feature_names,
        },
    )


def run_raw_stats(context: DatasetContext) -> dict[str, Any]:
    train_loader, test_loader, stat_in_features, feature_names, dropped_feature_names = make_prepared_raw_stats_loaders(
        context.X_train,
        context.X_test,
        context.y_train,
        context.y_test,
        context.X_train_2d,
        context.X_test_2d,
        STAT_PARAMS,
        batch_size=context.params["batch_size"],
        feature_batch_size=context.feature_batch_size,
    )
    model = RawStatsConcatClassifier(
        raw_in_channels=context.params["channels"],
        stat_in_features=stat_in_features,
        num_classes=context.params["num_classes"],
    )
    started_at = time.perf_counter()
    epoch_losses, metrics = fit_and_evaluate(
        model,
        train_loader,
        test_loader,
        context,
        batch_adapter=two_input_batch,
    )
    return build_result(
        context,
        method="raw_stats",
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={
            "modalities": ["raw", "stats"],
            "fusion": "concat",
            "stat_params": STAT_PARAMS,
            "stat_in_features": stat_in_features,
            "stat_feature_names": feature_names,
            "dropped_feature_names": dropped_feature_names,
        },
    )


def run_image(context: DatasetContext, image_spec: ImageSpec) -> dict[str, Any]:
    transformer = image_spec.transformer_cls(image_spec.transform_params)
    X_train_image, X_test_image = image_spec.input_getter(context)
    train_loader, test_loader, transform_shape, cnn_shape = make_prepared_image_loaders(
        transformer,
        X_train_image,
        X_test_image,
        context.y_train,
        context.y_test,
        batch_size=context.params["batch_size"],
        image_batch_size=context.image_batch_size,
        use_log1p=image_spec.use_log1p,
    )
    model = image_spec.classifier_cls(
        in_channels=cnn_shape[0],
        num_classes=context.params["num_classes"],
    )
    started_at = time.perf_counter()
    epoch_losses, metrics = fit_and_evaluate(
        model,
        train_loader,
        test_loader,
        context,
        batch_adapter=single_input_batch,
    )
    return build_result(
        context,
        method=image_spec.transform_name,
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={
            "modalities": [image_spec.transform_name],
            "encoder": "2D CNN",
            "transform_params": image_spec.transform_params,
            "transform_shape": transform_shape,
            "cnn_shape": cnn_shape,
        },
    )


def run_raw_image(context: DatasetContext, image_spec: ImageSpec) -> dict[str, Any]:
    transformer = image_spec.transformer_cls(image_spec.transform_params)
    X_train_image, X_test_image = image_spec.input_getter(context)
    train_loader, test_loader, transform_shape, cnn_shape = make_prepared_raw_image_loaders(
        transformer,
        context.X_train,
        context.X_test,
        X_train_image,
        X_test_image,
        context.y_train,
        context.y_test,
        batch_size=context.params["batch_size"],
        image_batch_size=context.image_batch_size,
        use_log1p=image_spec.use_log1p,
    )
    model = image_spec.raw_classifier_cls(
        raw_in_channels=context.params["channels"],
        image_in_channels=cnn_shape[0],
        num_classes=context.params["num_classes"],
    )
    started_at = time.perf_counter()
    epoch_losses, metrics = fit_and_evaluate(
        model,
        train_loader,
        test_loader,
        context,
        batch_adapter=two_input_batch,
    )
    return build_result(
        context,
        method=f"raw_{image_spec.transform_name}",
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={
            "modalities": ["raw", image_spec.transform_name],
            "fusion": "concat",
            "transform_params": image_spec.transform_params,
            "transform_shape": transform_shape,
            "cnn_shape": cnn_shape,
        },
    )


def make_image_runner(image_spec: ImageSpec) -> Callable[[DatasetContext], dict[str, Any]]:
    def runner(context: DatasetContext) -> dict[str, Any]:
        return run_image(context, image_spec)

    return runner


def make_raw_image_runner(image_spec: ImageSpec) -> Callable[[DatasetContext], dict[str, Any]]:
    def runner(context: DatasetContext) -> dict[str, Any]:
        return run_raw_image(context, image_spec)

    return runner


def normalized_image_inputs(context: DatasetContext) -> tuple[torch.Tensor, torch.Tensor]:
    return context.X_train_2d, context.X_test_2d


def scaled_image_inputs(context: DatasetContext) -> tuple[torch.Tensor, torch.Tensor]:
    return context.X_train_scaled_2d, context.X_test_scaled_2d


def make_architecture_specs() -> list[ArchitectureSpec]:
    image_specs = (
        ImageSpec("GAF", GAF, GAF_PARAMS, GAFClassifier, RawGAFConcatClassifier, scaled_image_inputs),
        ImageSpec("MTF", MTF, MTF_PARAMS, MTFClassifier, RawMTFConcatClassifier, normalized_image_inputs),
        ImageSpec(
            "STFT",
            STFTSpectrogram,
            STFT_PARAMS,
            STFTClassifier,
            RawSTFTConcatClassifier,
            normalized_image_inputs,
            use_log1p=True,
        ),
    )
    image_architectures = [ArchitectureSpec(spec.transform_name, make_image_runner(spec)) for spec in image_specs]
    raw_image_architectures = [
        ArchitectureSpec(f"raw_{spec.transform_name}", make_raw_image_runner(spec)) for spec in image_specs
    ]
    return [
        ArchitectureSpec("raw", run_raw),
        ArchitectureSpec("stats", run_stats),
        ArchitectureSpec("raw_stats", run_raw_stats),
        *image_architectures,
        *raw_image_architectures,
    ]


def result_file_name(dataset_name: str, method: str) -> str:
    return f"{dataset_name}__{method}.json".replace("+", "_").replace(" ", "_")


def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    architecture_specs = make_architecture_specs()

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

        for spec in architecture_specs:
            print(f"[{dataset_name}] running {spec.method}")
            set_seed(args.seed)
            result = spec.runner(context)
            save_json_result(result, output_dir / result_file_name(dataset_name, spec.method))
            rows.append(make_summary_row(result, METRIC_NAMES))
            save_summary_csv(rows, output_dir / "summary.csv")
            save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")

    save_summary_csv(rows, output_dir / "summary.csv")
    save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw/stat/image architectures across UCR datasets.")
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
