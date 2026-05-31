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

from experiments.architecture_evaluating import (  # noqa: E402
    DATASET_NAMES,
    METRIC_NAMES,
    DatasetContext,
    build_dataset_context,
    build_result,
    fit_and_evaluate,
    result_file_name,
)
from experiments.representation_concat_evaluating import (  # noqa: E402
    PreparedRepresentations,
    prepare_representations,
)
from experiments.tools import make_json_safe, make_summary_row, save_json_result, save_summary_csv  # noqa: E402
from src.models import (  # noqa: E402
    RawCenteredStatsGAFSTFTClassifier,
    RawFiLMStatsGAFSTFTClassifier,
    RawGatedStatsGAFSTFTClassifier,
)
from src.tools import get_device, set_seed  # noqa: E402


RESULT_DIR = Path("results/multimodal_architecture_results")
MODALITY_COMBINATIONS = {
    "raw_STFT": ("stft",),
    "raw_GAF": ("gaf",),
    "raw_stats_STFT": ("stats", "stft"),
    "raw_stats_GAF_STFT": ("stats", "gaf", "stft"),
}


@dataclass(frozen=True)
class FusionSpec:
    name: str
    classifier_cls: type[nn.Module]
    model_kwargs: dict[str, Any]
    importance_kind: str


@dataclass(frozen=True)
class ExperimentSpec:
    method: str
    fusion: FusionSpec
    modality_label: str
    context_modalities: tuple[str, ...]


def full_model_batch(
    batch: tuple[Any, ...],
    device: torch.device,
    context_modalities: tuple[str, ...],
) -> tuple[tuple[Any, ...], torch.Tensor]:
    *values, y = batch
    x_raw = values[0].to(device)
    context_values = {modality: value.to(device) for modality, value in zip(context_modalities, values[1:])}
    return (
        x_raw,
        context_values.get("stats"),
        context_values.get("gaf"),
        context_values.get("stft"),
    ), y.to(device)


def make_batch_adapter(
    context_modalities: tuple[str, ...],
) -> Callable[[tuple[Any, ...], torch.device], tuple[tuple[Any, ...], torch.Tensor]]:
    def adapter(batch: tuple[Any, ...], device: torch.device) -> tuple[tuple[Any, ...], torch.Tensor]:
        return full_model_batch(batch, device, context_modalities)

    return adapter


def modality_tensors(
    context: DatasetContext,
    reps: PreparedRepresentations,
    context_modalities: tuple[str, ...],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    train_tensors: list[torch.Tensor] = [context.X_train]
    test_tensors: list[torch.Tensor] = [context.X_test]

    for modality in context_modalities:
        if modality == "stats":
            train_tensors.append(reps.F_train)
            test_tensors.append(reps.F_test)
        elif modality == "gaf":
            train_tensors.append(reps.X_train_gaf)
            test_tensors.append(reps.X_test_gaf)
        elif modality == "stft":
            train_tensors.append(reps.X_train_stft)
            test_tensors.append(reps.X_test_stft)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    return tuple(train_tensors), tuple(test_tensors)


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


def build_model(
    spec: ExperimentSpec,
    context: DatasetContext,
    reps: PreparedRepresentations,
) -> nn.Module:
    return spec.fusion.classifier_cls(
        raw_in_channels=context.params["channels"],
        stat_in_features=reps.stat_in_features,
        image_in_channels=1,
        num_classes=context.params["num_classes"],
        context_modalities=spec.context_modalities,
        **spec.fusion.model_kwargs,
    )


@torch.no_grad()
def collect_aux_outputs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    context_modalities: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {}
    batch_adapter = make_batch_adapter(context_modalities)

    for batch in dataloader:
        inputs, _ = batch_adapter(batch, device)
        aux = model(*inputs, return_aux=True)
        if isinstance(aux, tuple):
            raise TypeError("Expected aux dict from multimodal model, got tuple.")

        for key, value in aux.items():
            if key == "logits" or not isinstance(value, torch.Tensor):
                continue
            collected.setdefault(key, []).append(value.detach().cpu())

    return {key: torch.cat(values, dim=0) for key, values in collected.items()}


def summarize_gated_importance(
    aux_outputs: dict[str, torch.Tensor],
    spec: ExperimentSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gates = aux_outputs["gates"]
    modalities = ("raw", *spec.context_modalities)
    gate_mean = gates.mean(dim=0)
    gate_std = gates.std(dim=0, unbiased=False)

    rows = []
    by_modality = {}
    for idx, modality in enumerate(modalities):
        by_modality[modality] = {
            "mean": gate_mean[idx].item(),
            "std": gate_std[idx].item(),
        }
        rows.append(
            {
                "coefficient": "gate",
                "target": modality,
                "value": gate_mean[idx].item(),
                "std": gate_std[idx].item(),
            }
        )

    return {"gate_by_modality": by_modality}, rows


def summarize_residual_importance(
    aux_outputs: dict[str, torch.Tensor],
    spec: ExperimentSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    alpha = aux_outputs["alpha"]
    alpha_abs = alpha.abs()
    summary = {
        "alpha_shape": list(alpha.shape[1:]),
        "alpha_mean": alpha.mean().item(),
        "alpha_abs_mean": alpha_abs.mean().item(),
        "alpha_abs_std": alpha_abs.std(unbiased=False).item(),
    }
    rows = [
        {
            "coefficient": "alpha_abs_mean",
            "target": "residual_delta",
            "value": summary["alpha_abs_mean"],
            "std": summary["alpha_abs_std"],
        }
    ]
    if alpha.shape[1] > 1:
        summary["alpha_abs_mean_by_feature"] = alpha_abs.mean(dim=0).tolist()
    return summary, rows


def summarize_film_importance(
    aux_outputs: dict[str, torch.Tensor],
    spec: ExperimentSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gamma_abs = aux_outputs["gamma"].abs()
    beta_abs = aux_outputs["beta"].abs()
    summary = {
        "gamma_abs_mean": gamma_abs.mean().item(),
        "gamma_abs_std": gamma_abs.std(unbiased=False).item(),
        "beta_abs_mean": beta_abs.mean().item(),
        "beta_abs_std": beta_abs.std(unbiased=False).item(),
        "gamma_abs_mean_by_feature": gamma_abs.mean(dim=0).tolist(),
        "beta_abs_mean_by_feature": beta_abs.mean(dim=0).tolist(),
    }
    rows = [
        {
            "coefficient": "gamma_abs_mean",
            "target": "raw_modulation",
            "value": summary["gamma_abs_mean"],
            "std": summary["gamma_abs_std"],
        },
        {
            "coefficient": "beta_abs_mean",
            "target": "raw_shift",
            "value": summary["beta_abs_mean"],
            "std": summary["beta_abs_std"],
        },
    ]
    return summary, rows


def summarize_importance(
    aux_outputs: dict[str, torch.Tensor],
    spec: ExperimentSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if spec.fusion.importance_kind == "gated":
        return summarize_gated_importance(aux_outputs, spec)
    if spec.fusion.importance_kind == "residual":
        return summarize_residual_importance(aux_outputs, spec)
    if spec.fusion.importance_kind == "film":
        return summarize_film_importance(aux_outputs, spec)
    raise ValueError(f"Unknown importance kind: {spec.fusion.importance_kind}")


def run_experiment_spec(
    context: DatasetContext,
    reps: PreparedRepresentations,
    spec: ExperimentSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_tensors, test_tensors = modality_tensors(context, reps, spec.context_modalities)
    train_loader, test_loader = make_loaders(train_tensors, test_tensors, context)
    model = build_model(spec, context, reps)
    batch_adapter = make_batch_adapter(spec.context_modalities)

    started_at = time.perf_counter()
    epoch_losses, metrics = fit_and_evaluate(
        model,
        train_loader,
        test_loader,
        context,
        batch_adapter=batch_adapter,
    )
    aux_outputs = collect_aux_outputs(model, test_loader, context.device, spec.context_modalities)
    importance, importance_rows = summarize_importance(aux_outputs, spec)

    result = build_result(
        context,
        method=spec.method,
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={
            "fusion": spec.fusion.name,
            "modalities": ("raw", *spec.context_modalities),
            "context_modalities": spec.context_modalities,
            "model_kwargs": spec.fusion.model_kwargs,
            "importance": importance,
        },
    )

    rows = [
        {
            "dataset_name": context.dataset_name,
            "method": spec.method,
            "fusion": spec.fusion.name,
            "modality_combo": spec.modality_label,
            **row,
        }
        for row in importance_rows
    ]
    return result, rows


def make_fusion_specs() -> list[FusionSpec]:
    return [
        FusionSpec(
            name="gated_fusion",
            classifier_cls=RawGatedStatsGAFSTFTClassifier,
            model_kwargs={},
            importance_kind="gated",
        ),
        FusionSpec(
            name="residual_gated_fusion_vector_alpha",
            classifier_cls=RawCenteredStatsGAFSTFTClassifier,
            model_kwargs={"alpha_is_vector": True},
            importance_kind="residual",
        ),
        FusionSpec(
            name="residual_gated_fusion_scalar_alpha",
            classifier_cls=RawCenteredStatsGAFSTFTClassifier,
            model_kwargs={"alpha_is_vector": False},
            importance_kind="residual",
        ),
        FusionSpec(
            name="film_fusion",
            classifier_cls=RawFiLMStatsGAFSTFTClassifier,
            model_kwargs={},
            importance_kind="film",
        ),
    ]


def make_experiment_specs() -> list[ExperimentSpec]:
    specs = []
    for fusion in make_fusion_specs():
        for modality_label, context_modalities in MODALITY_COMBINATIONS.items():
            specs.append(
                ExperimentSpec(
                    method=f"{fusion.name}__{modality_label}",
                    fusion=fusion,
                    modality_label=modality_label,
                    context_modalities=context_modalities,
                )
            )
    return specs


def save_importance_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    save_summary_csv(rows, output_dir / "importance_summary.csv")
    save_json_result({"rows": make_json_safe(rows)}, output_dir / "importance_summary.json")


def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    specs = make_experiment_specs()

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
            result, result_importance_rows = run_experiment_spec(context, reps, spec)
            save_json_result(result, output_dir / result_file_name(dataset_name, spec.method))
            rows.append(make_summary_row(result, METRIC_NAMES))
            importance_rows.extend(result_importance_rows)

            save_summary_csv(rows, output_dir / "summary.csv")
            save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")
            save_importance_summary(importance_rows, output_dir)

    save_summary_csv(rows, output_dir / "summary.csv")
    save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")
    save_importance_summary(importance_rows, output_dir)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multimodal fusion architectures.")
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
