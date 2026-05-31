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
from src.models import RawStatsGAFSTFTBottleneckClassifier  # noqa: E402
from src.tools import get_device, set_seed  # noqa: E402


RESULT_DIR = Path("results/representative_learning_results")
ATTENTION_NOTE = (
    "Attention is a diagnostic view of latent-token interactions with modality "
    "tokens; it is not absolute causal importance."
)
BOTTLENECK_PARAMS = {
    "d_model": 128,
    "num_latents": 4,
    "num_heads": 4,
    "num_bottleneck_layers": 1,
    "bottleneck_dropout": 0.1,
    "pooling": "mean",
}
MODALITY_COMBINATIONS = {
    "raw_STFT": ("stft",),
    "raw_GAF": ("gaf",),
    "raw_stats_STFT": ("stats", "stft"),
    "raw_stats_GAF_STFT": ("stats", "gaf", "stft"),
}


@dataclass(frozen=True)
class BottleneckSpec:
    method: str
    modality_label: str
    context_modalities: tuple[str, ...]


def bottleneck_batch(
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
        return bottleneck_batch(batch, device, context_modalities)

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
    spec: BottleneckSpec,
    context: DatasetContext,
    reps: PreparedRepresentations,
) -> nn.Module:
    return RawStatsGAFSTFTBottleneckClassifier(
        raw_in_channels=context.params["channels"],
        stat_in_features=reps.stat_in_features,
        image_in_channels=1,
        num_classes=context.params["num_classes"],
        context_modalities=spec.context_modalities,
        **BOTTLENECK_PARAMS,
    )


@torch.no_grad()
def collect_aux_outputs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    context_modalities: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    attn_blocks: list[list[torch.Tensor]] = []
    latent_batches: list[torch.Tensor] = []
    batch_adapter = make_batch_adapter(context_modalities)
    modality_order: tuple[str, ...] | None = None

    for batch in dataloader:
        inputs, _ = batch_adapter(batch, device)
        aux = model(*inputs, return_aux=True)
        modality_order = tuple(aux["modality_order"])
        latent_batches.append(aux["latents"].detach().cpu())

        for layer_idx, layer_attn in enumerate(aux["attn"]):
            if len(attn_blocks) <= layer_idx:
                attn_blocks.append([])
            attn_blocks[layer_idx].append(layer_attn["cross_attn"].detach().cpu())

    return {
        "modality_order": modality_order,
        "latents": torch.cat(latent_batches, dim=0),
        "cross_attn_by_layer": [torch.cat(values, dim=0) for values in attn_blocks],
    }


def summarize_attention(
    aux_outputs: dict[str, Any],
    spec: BottleneckSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    modality_order = tuple(aux_outputs["modality_order"])
    cross_attn_by_layer = aux_outputs["cross_attn_by_layer"]
    stacked_attn = torch.stack(cross_attn_by_layer, dim=0)
    # [layers, batch, heads, latent, modality] -> [latent, modality]
    mean_attn = stacked_attn.mean(dim=(0, 1, 2))
    std_attn = stacked_attn.std(dim=(0, 1, 2), unbiased=False)

    rows: list[dict[str, Any]] = []
    by_latent = {}
    for latent_idx in range(mean_attn.shape[0]):
        latent_name = f"Z{latent_idx + 1}"
        latent_values = {}
        for modality_idx, modality in enumerate(modality_order):
            value = mean_attn[latent_idx, modality_idx].item()
            std = std_attn[latent_idx, modality_idx].item()
            latent_values[modality] = value
            rows.append(
                {
                    "latent": latent_name,
                    "modality": modality,
                    "attention_mean": value,
                    "attention_std": std,
                    "note": ATTENTION_NOTE,
                }
            )
        top_modality = max(latent_values, key=latent_values.get)
        by_latent[latent_name] = {
            "attention_by_modality": latent_values,
            "top_modality": top_modality,
            "top_attention": latent_values[top_modality],
        }

    summary = {
        "attention_note": ATTENTION_NOTE,
        "modality_order": modality_order,
        "num_layers": len(cross_attn_by_layer),
        "num_latents": int(mean_attn.shape[0]),
        "attention_by_latent": by_latent,
    }
    return summary, rows


def run_bottleneck_spec(
    context: DatasetContext,
    reps: PreparedRepresentations,
    spec: BottleneckSpec,
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
    attention_summary, attention_rows = summarize_attention(aux_outputs, spec)

    result = build_result(
        context,
        method=spec.method,
        epoch_losses=epoch_losses,
        metrics=metrics,
        duration_sec=time.perf_counter() - started_at,
        metadata={
            "fusion": "bottleneck_representation_encoder",
            "modalities": ("raw", *spec.context_modalities),
            "context_modalities": spec.context_modalities,
            "bottleneck_params": BOTTLENECK_PARAMS,
            "attention": attention_summary,
        },
    )

    rows = [
        {
            "dataset_name": context.dataset_name,
            "method": spec.method,
            "modality_combo": spec.modality_label,
            **row,
        }
        for row in attention_rows
    ]
    return result, rows


def make_bottleneck_specs() -> list[BottleneckSpec]:
    return [
        BottleneckSpec(
            method=f"bottleneck__{modality_label}",
            modality_label=modality_label,
            context_modalities=context_modalities,
        )
        for modality_label, context_modalities in MODALITY_COMBINATIONS.items()
    ]


def save_attention_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    save_summary_csv(rows, output_dir / "attention_summary.csv")
    save_json_result({"rows": make_json_safe(rows)}, output_dir / "attention_summary.json")


def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    specs = make_bottleneck_specs()

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
            result, result_attention_rows = run_bottleneck_spec(context, reps, spec)
            save_json_result(result, output_dir / result_file_name(dataset_name, spec.method))
            rows.append(make_summary_row(result, METRIC_NAMES))
            attention_rows.extend(result_attention_rows)

            save_summary_csv(rows, output_dir / "summary.csv")
            save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")
            save_attention_summary(attention_rows, output_dir)

    save_summary_csv(rows, output_dir / "summary.csv")
    save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")
    save_attention_summary(attention_rows, output_dir)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bottleneck representation encoder.")
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
