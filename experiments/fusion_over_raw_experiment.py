from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from experiments.tools import make_json_safe, save_json_result, save_summary_csv
from src.data import (
    as_univariate_feature_tensor,
    feats_batched,
    get_stat_feature_names,
    load_dataset,
    to_cnn_images,
    transform_images_batched,
)
from src.representations.image_transformation.methods.gaf_transformation import GAF
from src.representations.image_transformation.methods.stft_transformation import STFTSpectrogram
from src.metrics.metrics import compute_classification_metrics
from src.models.head.classification import ClassificationHead
from src.models.multimodals.bottleneck_fusion_clf import FlexibleBottleneckClassifier
from src.models.multimodals.concat_fusion_clf import FlexibleConcatClassifier
from src.models.multimodals.context_only_residual_bottleneck_clf import (
    FlexibleContextOnlyResidualBottleneckClassifier,
)
from src.models.multimodals.film_fusion_clf import FlexibleFiLMClassifier
from src.models.multimodals.gated_fusion_clf import FlexibleGatedClassifier
from src.models.multimodals.raw_conditioned_bottleneck_clf import (
    FlexibleRawConditionedContextBottleneckClassifier,
)
from src.models.multimodals.raw_residual_bottleneck_clf import FlexibleRawResidualBottleneckClassifier
from src.models.multimodals.raw_residual_centered_fusion_clf import FlexibleRawCenteredResidualClassifier
from src.models.registry.encoder_registry import ENCODER_REGISTRY
from src.representations.statistical.quantile_extractor import TorchQuantileExtractor
from src.tools import get_device, per_sample_minmax_scale, per_sample_z_normalize, set_seed


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "fusion_over_raw.json"
DEFAULT_OUTPUT_DIR = Path("results/fusion_over_raw")
TRACKED_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
ALL_MODALITIES = ("raw", "stats", "gaf", "stft")


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    dataset_name: str
    seed: int
    channels: int
    time_steps: int
    num_classes: int
    batch_size: int
    train_size: int
    val_size: int
    test_size: int
    tensors_train: dict[str, torch.Tensor]
    tensors_val: dict[str, torch.Tensor]
    tensors_test: dict[str, torch.Tensor]
    y_train: torch.Tensor
    y_val: torch.Tensor
    y_test: torch.Tensor
    encoder_shapes: dict[str, tuple[int, ...] | int]
    metadata: dict[str, Any]


class MappingClassifierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        y: torch.Tensor,
        modalities: tuple[str, ...],
    ) -> None:
        self.tensors = {name: tensors[name].float() for name in modalities}
        self.y = y.long()
        self.modalities = modalities

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        return {name: tensor[idx] for name, tensor in self.tensors.items()}, self.y[idx]


class SingleViewClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_modality: str,
        encoder_modality: str,
        num_classes: int,
        d_model: int,
        encoder_kwargs: dict[str, Any],
        head_hidden_dim: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.input_modality = input_modality
        self.encoder = ENCODER_REGISTRY[encoder_modality](d_model, encoder_kwargs)
        self.head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encoder(inputs[self.input_modality]))


def collate_mapping_batch(batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    keys = tuple(batch[0][0].keys())
    inputs = {key: torch.stack([item[0][key] for item in batch], dim=0) for key in keys}
    y = torch.stack([item[1] for item in batch], dim=0)
    return inputs, y


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def override_config_from_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    if args.datasets:
        config["datasets"] = args.datasets
    if args.seeds:
        config["seeds"] = args.seeds
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    return config


def ensure_stratify(y: np.ndarray, val_size: float, requested: bool) -> np.ndarray | None:
    if not requested:
        return None
    _, counts = np.unique(y, return_counts=True)
    n_val = int(np.ceil(len(y) * val_size))
    if counts.min(initial=0) < 2 or n_val < len(counts):
        return None
    return y


def make_train_val_split(y_train: np.ndarray, config: dict[str, Any], seed: int) -> SplitIndices:
    validation = config["validation"]
    indices = np.arange(len(y_train))
    stratify = ensure_stratify(
        y_train,
        float(validation["val_size"]),
        bool(validation.get("stratify", True)),
    )
    train_idx, val_idx = train_test_split(
        indices,
        test_size=float(validation["val_size"]),
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return SplitIndices(train=np.asarray(train_idx), val=np.asarray(val_idx))


def replace_and_standardize_three_way(
    train: torch.Tensor,
    val: torch.Tensor,
    test: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = train.float()
    val = val.float()
    test = test.float()

    finite_train = torch.isfinite(train)
    train_with_nan = torch.where(finite_train, train, torch.nan)
    col_mean = torch.nanmean(train_with_nan, dim=0, keepdim=True)
    col_mean = torch.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)

    train = torch.where(finite_train, train, col_mean)
    val = torch.where(torch.isfinite(val), val, col_mean)
    test = torch.where(torch.isfinite(test), test, col_mean)

    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    return (
        torch.nan_to_num((train - mean) / std).float(),
        torch.nan_to_num((val - mean) / std).float(),
        torch.nan_to_num((test - mean) / std).float(),
    )


def prepare_stats_three_way(
    train_2d: torch.Tensor,
    val_2d: torch.Tensor,
    test_2d: torch.Tensor,
    stat_params: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    extractor = TorchQuantileExtractor(stat_params)
    f_train = feats_batched(extractor, train_2d, batch_size=batch_size)
    f_val = feats_batched(extractor, val_2d, batch_size=batch_size)
    f_test = feats_batched(extractor, test_2d, batch_size=batch_size)
    feature_names = get_stat_feature_names(stat_params, f_train.shape[1])
    f_train, f_val, f_test = replace_and_standardize_three_way(f_train, f_val, f_test)
    return f_train, f_val, f_test, feature_names


def standardize_images_three_way(
    train: torch.Tensor,
    val: torch.Tensor,
    test: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=(0, 2, 3), keepdim=True)
    std = train.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(eps)
    return (
        torch.nan_to_num((train - mean) / std).float(),
        torch.nan_to_num((val - mean) / std).float(),
        torch.nan_to_num((test - mean) / std).float(),
    )


def prepare_image_three_way(
    transformer: Any,
    train_2d: torch.Tensor,
    val_2d: torch.Tensor,
    test_2d: torch.Tensor,
    *,
    image_batch_size: int,
    use_log1p: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    train_img = transform_images_batched(transformer, train_2d, batch_size=image_batch_size)
    val_img = transform_images_batched(transformer, val_2d, batch_size=image_batch_size)
    test_img = transform_images_batched(transformer, test_2d, batch_size=image_batch_size)

    if use_log1p:
        train_img = torch.log1p(train_img.float().clamp_min(0))
        val_img = torch.log1p(val_img.float().clamp_min(0))
        test_img = torch.log1p(test_img.float().clamp_min(0))

    transform_shape = tuple(train_img.shape[1:])
    train_img = to_cnn_images(train_img)
    val_img = to_cnn_images(val_img)
    test_img = to_cnn_images(test_img)
    cnn_shape = tuple(train_img.shape[1:])
    train_img, val_img, test_img = standardize_images_three_way(train_img, val_img, test_img)
    return train_img, val_img, test_img, transform_shape, cnn_shape


def make_stft_params_for_length(params: dict[str, Any], time_steps: int) -> dict[str, Any]:
    adjusted = dict(params)
    if time_steps <= 1:
        raise ValueError(f"STFT requires time_steps > 1, got {time_steps}")

    n_fft = min(int(adjusted.get("n_fft", adjusted.get("window_size", time_steps))), time_steps)
    window_size = min(int(adjusted.get("window_size", n_fft)), n_fft)
    hop_length = min(int(adjusted.get("hop_length", max(1, window_size // 4))), window_size)

    adjusted["n_fft"] = max(1, n_fft)
    adjusted["window_size"] = max(1, window_size)
    adjusted["hop_length"] = max(1, hop_length)
    return adjusted


def prepare_dataset_for_seed(
    dataset_name: str,
    seed: int,
    config: dict[str, Any],
    *,
    use_cache: bool,
) -> PreparedDataset:
    dataset = load_dataset(dataset_name, use_cache=use_cache)
    split = make_train_val_split(dataset.y_train, config, seed)

    x_official_train = torch.from_numpy(dataset.X_train).float()
    y_official_train = torch.from_numpy(dataset.y_train).long()
    x_test_raw = torch.from_numpy(dataset.X_test).float()
    y_test = torch.from_numpy(dataset.y_test).long()

    x_train_raw = x_official_train[split.train]
    y_train = y_official_train[split.train]
    x_val_raw = x_official_train[split.val]
    y_val = y_official_train[split.val]

    x_train = per_sample_z_normalize(x_train_raw)
    x_val = per_sample_z_normalize(x_val_raw)
    x_test = per_sample_z_normalize(x_test_raw)

    train_2d = as_univariate_feature_tensor(x_train)
    val_2d = as_univariate_feature_tensor(x_val)
    test_2d = as_univariate_feature_tensor(x_test)

    training_cfg = config["training"]
    repr_cfg = config["representations"]
    f_train, f_val, f_test, stat_feature_names = prepare_stats_three_way(
        train_2d,
        val_2d,
        test_2d,
        repr_cfg["stat_params"],
        batch_size=int(training_cfg["feature_batch_size"]),
    )

    gaf_train_2d = per_sample_minmax_scale(train_2d).clamp(-1.0, 1.0)
    gaf_val_2d = per_sample_minmax_scale(val_2d).clamp(-1.0, 1.0)
    gaf_test_2d = per_sample_minmax_scale(test_2d).clamp(-1.0, 1.0)
    x_train_gaf, x_val_gaf, x_test_gaf, gaf_transform_shape, gaf_cnn_shape = prepare_image_three_way(
        GAF(repr_cfg["gaf_params"]),
        gaf_train_2d,
        gaf_val_2d,
        gaf_test_2d,
        image_batch_size=int(training_cfg["image_batch_size"]),
        use_log1p=False,
    )
    stft_params = make_stft_params_for_length(repr_cfg["stft_params"], train_2d.shape[1])
    x_train_stft, x_val_stft, x_test_stft, stft_transform_shape, stft_cnn_shape = prepare_image_three_way(
        STFTSpectrogram(stft_params),
        train_2d,
        val_2d,
        test_2d,
        image_batch_size=int(training_cfg["image_batch_size"]),
        use_log1p=True,
    )

    batch_size = min(int(training_cfg["max_batch_size"]), len(y_train))
    return PreparedDataset(
        dataset_name=dataset_name,
        seed=seed,
        channels=x_train.shape[1],
        time_steps=x_train.shape[2],
        num_classes=int(y_official_train.max().item()) + 1,
        batch_size=batch_size,
        train_size=len(y_train),
        val_size=len(y_val),
        test_size=len(y_test),
        tensors_train={"raw": x_train, "raw_larger": x_train, "stats": f_train, "gaf": x_train_gaf, "stft": x_train_stft},
        tensors_val={"raw": x_val, "raw_larger": x_val, "stats": f_val, "gaf": x_val_gaf, "stft": x_val_stft},
        tensors_test={"raw": x_test, "raw_larger": x_test, "stats": f_test, "gaf": x_test_gaf, "stft": x_test_stft},
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        encoder_shapes={
            "raw": tuple(x_train.shape[1:]),
            "raw_larger": tuple(x_train.shape[1:]),
            "stats": f_train.shape[1],
            "gaf": gaf_cnn_shape,
            "stft": stft_cnn_shape,
        },
        metadata={
            "split_indices": {"train": split.train.tolist(), "val": split.val.tolist()},
            "stat_feature_names": stat_feature_names,
            "gaf_transform_shape": gaf_transform_shape,
            "gaf_cnn_shape": gaf_cnn_shape,
            "stft_params": stft_params,
            "stft_transform_shape": stft_transform_shape,
            "stft_cnn_shape": stft_cnn_shape,
            "normalization": {
                "raw": "per-sample z-normalization",
                "stats": "features generated on split tensors, imputed/standardized with train statistics",
                "gaf": "per-sample minmax to [-1, 1], image standardization with train statistics",
                "stft": "log1p spectrogram, image standardization with train statistics",
            },
        },
    )


def make_loader(
    tensors: dict[str, torch.Tensor],
    y: torch.Tensor,
    modalities: tuple[str, ...],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = MappingClassifierDataset(tensors, y, modalities)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_mapping_batch,
    )


def make_loaders(
    prepared: PreparedDataset,
    modalities: tuple[str, ...],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        make_loader(prepared.tensors_train, prepared.y_train, modalities, batch_size=prepared.batch_size, shuffle=True),
        make_loader(prepared.tensors_val, prepared.y_val, modalities, batch_size=prepared.batch_size, shuffle=False),
        make_loader(prepared.tensors_test, prepared.y_test, modalities, batch_size=prepared.batch_size, shuffle=False),
    )


def encoder_kwargs_for(
    prepared: PreparedDataset,
    config: dict[str, Any],
    modalities: tuple[str, ...],
    *,
    raw_larger: bool = False,
) -> dict[str, dict[str, Any]]:
    model_cfg = config["model"]
    raw_cfg = model_cfg["raw_larger_encoder"] if raw_larger else model_cfg["raw_encoder"]
    image_cfg = model_cfg["image_encoder"]
    kwargs: dict[str, dict[str, Any]] = {}

    for modality in modalities:
        if modality in {"raw", "raw_larger"}:
            kwargs["raw"] = {
                "in_channels": prepared.channels,
                "hidden_channels": tuple(raw_cfg["hidden_channels"]),
                "kernel_size": raw_cfg["kernel_size"],
                "dropout": raw_cfg["dropout"],
            }
        elif modality == "stats":
            kwargs["stats"] = {
                "in_features": int(prepared.encoder_shapes["stats"]),
                "hidden_dims": tuple(model_cfg["stats_encoder"]["hidden_dims"]),
                "dropout": model_cfg["stats_encoder"]["dropout"],
            }
        elif modality in {"gaf", "stft"}:
            shape = prepared.encoder_shapes[modality]
            if not isinstance(shape, tuple):
                raise TypeError(f"Expected CNN shape tuple for {modality}, got {shape!r}")
            kwargs[modality] = {
                "in_channels": shape[0],
                "hidden_channels": tuple(image_cfg["hidden_channels"]),
                "dropout": image_cfg["dropout"],
            }
        else:
            raise ValueError(f"Unknown modality: {modality}")

    return kwargs


def fusion_common_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    return {
        "d_model": model_cfg["d_model"],
        "head_hidden_dim": model_cfg["head_hidden_dim"],
        "head_dropout": model_cfg["head_dropout"],
    }


def build_single_view_model(
    prepared: PreparedDataset,
    config: dict[str, Any],
    modality: str,
) -> nn.Module:
    raw_larger = modality == "raw_larger"
    encoder_modality = "raw" if raw_larger else modality
    encoder_kwargs = encoder_kwargs_for(prepared, config, (modality,), raw_larger=raw_larger)[encoder_modality]
    common = fusion_common_kwargs(config)
    return SingleViewClassifier(
        input_modality=modality,
        encoder_modality=encoder_modality,
        num_classes=prepared.num_classes,
        encoder_kwargs=encoder_kwargs,
        **common,
    )


def build_fusion_model(
    prepared: PreparedDataset,
    config: dict[str, Any],
    modalities: tuple[str, ...],
    fusion_name: str,
) -> nn.Module:
    if "raw" not in modalities:
        raise ValueError(f"Fusion experiments must include raw, got {modalities}")

    model_cfg = config["model"]
    fusion_cfg = model_cfg["fusion"]
    encoder_kwargs = encoder_kwargs_for(prepared, config, modalities)
    common = fusion_common_kwargs(config)
    context_modalities = tuple(modality for modality in modalities if modality != "raw")

    if fusion_name == "concat":
        return FlexibleConcatClassifier(
            modalities=modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            fusion_hidden_dim=fusion_cfg["fusion_hidden_dim"],
            fusion_dropout=fusion_cfg["fusion_dropout"],
            **common,
        )
    if fusion_name == "gated":
        return FlexibleGatedClassifier(
            modalities=modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            fusion_hidden_dim=fusion_cfg["fusion_hidden_dim"],
            fusion_dropout=fusion_cfg["fusion_dropout"],
            **common,
        )
    if fusion_name == "film":
        return FlexibleFiLMClassifier(
            raw_modality="raw",
            context_modalities=context_modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            context_hidden_dim=fusion_cfg["context_hidden_dim"],
            film_hidden_dim=fusion_cfg["film_hidden_dim"],
            film_dropout=fusion_cfg["fusion_dropout"],
            **common,
        )
    if fusion_name == "raw_centered_residual":
        return FlexibleRawCenteredResidualClassifier(
            raw_modality="raw",
            context_modalities=context_modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            context_hidden_dim=fusion_cfg["context_hidden_dim"],
            delta_hidden_dim=fusion_cfg["delta_hidden_dim"],
            fusion_dropout=fusion_cfg["fusion_dropout"],
            **common,
        )
    if fusion_name == "ordinary_bottleneck":
        return FlexibleBottleneckClassifier(
            modalities=modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            num_latents=fusion_cfg["num_latents"],
            num_bottleneck_layers=fusion_cfg["num_bottleneck_layers"],
            num_heads=fusion_cfg["num_heads"],
            mlp_ratio=fusion_cfg["mlp_ratio"],
            pooling=fusion_cfg["pooling"],
            bottleneck_dropout=fusion_cfg["bottleneck_dropout"],
            **common,
        )
    if fusion_name == "raw_residual_bottleneck":
        return FlexibleRawResidualBottleneckClassifier(
            raw_modality="raw",
            context_modalities=context_modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            bottleneck_dropout=fusion_cfg["bottleneck_dropout"],
            residual_dropout=fusion_cfg["residual_dropout"],
            num_latents=fusion_cfg["num_latents"],
            num_bottleneck_layers=fusion_cfg["num_bottleneck_layers"],
            num_heads=fusion_cfg["num_heads"],
            mlp_ratio=fusion_cfg["mlp_ratio"],
            pooling=fusion_cfg["pooling"],
            **common,
        )
    if fusion_name == "context_only_residual_bottleneck":
        return FlexibleContextOnlyResidualBottleneckClassifier(
            raw_modality="raw",
            context_modalities=context_modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            bottleneck_dropout=fusion_cfg["bottleneck_dropout"],
            residual_dropout=fusion_cfg["residual_dropout"],
            num_latents=fusion_cfg["num_latents"],
            num_bottleneck_layers=fusion_cfg["num_bottleneck_layers"],
            num_heads=fusion_cfg["num_heads"],
            mlp_ratio=fusion_cfg["mlp_ratio"],
            pooling=fusion_cfg["pooling"],
            **common,
        )
    if fusion_name == "raw_conditioned_context_bottleneck":
        return FlexibleRawConditionedContextBottleneckClassifier(
            raw_modality="raw",
            context_modalities=context_modalities,
            num_classes=prepared.num_classes,
            encoder_kwargs=encoder_kwargs,
            bottleneck_dropout=fusion_cfg["bottleneck_dropout"],
            residual_dropout=fusion_cfg["residual_dropout"],
            num_latents=fusion_cfg["num_latents"],
            num_bottleneck_layers=fusion_cfg["num_bottleneck_layers"],
            num_heads=fusion_cfg["num_heads"],
            mlp_ratio=fusion_cfg["mlp_ratio"],
            pooling=fusion_cfg["pooling"],
            **common,
        )
    raise ValueError(f"Unknown fusion: {fusion_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def move_inputs(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in inputs.items()}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    for inputs, y in loader:
        y = y.to(device)
        logits = model(move_inputs(inputs, device))
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    metrics = compute_classification_metrics(
        y_true=np.concatenate(all_targets),
        y_pred=np.concatenate(all_preds),
    )
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def train_best_val_checkpoint(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    training_cfg = config["training"]
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["lr"]))
    patience = int(training_cfg["patience"])
    max_epochs = int(training_cfg["epochs"])

    best_val_macro_f1 = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for inputs, y in train_loader:
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(move_inputs(inputs, device))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

        val_metrics = evaluate_model(model, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_samples, 1),
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(epoch_row)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": checkpoint_path,
    }


def base_result_row(
    prepared: PreparedDataset,
    *,
    architecture: str,
    modalities: tuple[str, ...],
    seed: int,
    num_params: int,
    best_epoch: int | None,
    duration_sec: float,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    row = {
        "dataset": prepared.dataset_name,
        "dataset_name": prepared.dataset_name,
        "architecture": architecture,
        "method": architecture,
        "modalities": "+".join(modalities),
        "seed": seed,
        "train_size": prepared.train_size,
        "val_size": prepared.val_size,
        "test_size": prepared.test_size,
        "channels": prepared.channels,
        "time_steps": prepared.time_steps,
        "num_classes": prepared.num_classes,
        "batch_size": prepared.batch_size,
        "num_params": num_params,
        "best_epoch": best_epoch,
        "duration_sec": duration_sec,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    for metric in TRACKED_METRICS:
        row[metric] = test_metrics[metric]
        row[f"val_{metric}"] = val_metrics[metric]
    return row


def result_file_name(dataset_name: str, architecture: str, modalities: tuple[str, ...], seed: int) -> str:
    modality_label = "+".join(modalities)
    safe = f"{dataset_name}__{architecture}__{modality_label}__seed_{seed}"
    return safe.replace("+", "_").replace("/", "_").replace(" ", "_") + ".json"


def result_path(
    output_dir: Path,
    dataset_name: str,
    architecture: str,
    modalities: tuple[str, ...],
    seed: int,
) -> Path:
    return output_dir / "runs" / result_file_name(dataset_name, architecture, modalities, seed)


def run_deep_model(
    prepared: PreparedDataset,
    config: dict[str, Any],
    *,
    architecture: str,
    modalities: tuple[str, ...],
    model: nn.Module,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_loader, val_loader, test_loader = make_loaders(prepared, modalities)
    checkpoint_path = (
        output_dir
        / "checkpoints"
        / prepared.dataset_name
        / f"{architecture}__{'+'.join(modalities)}__seed_{prepared.seed}.pt".replace("+", "_")
    )
    started_at = time.perf_counter()
    train_state = train_best_val_checkpoint(
        model,
        train_loader,
        val_loader,
        device,
        config,
        checkpoint_path,
    )
    trained_model = train_state["model"]
    val_metrics = evaluate_model(trained_model, val_loader, device)
    test_metrics = evaluate_model(trained_model, test_loader, device)
    duration_sec = time.perf_counter() - started_at
    row = base_result_row(
        prepared,
        architecture=architecture,
        modalities=modalities,
        seed=prepared.seed,
        num_params=count_parameters(trained_model),
        best_epoch=train_state["best_epoch"],
        duration_sec=duration_sec,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        checkpoint_path=checkpoint_path,
    )
    result = {
        **row,
        "training_history": train_state["history"],
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "config": config,
        "prepared_metadata": prepared.metadata,
    }
    return row, result


def as_numpy_features(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def run_minirocket(
    prepared: PreparedDataset,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from sktime.transformations.panel.rocket import MiniRocket
    except ImportError as exc:
        raise RuntimeError("MiniROCKET baseline requires sktime. Install dependencies with `uv sync`.") from exc

    started_at = time.perf_counter()
    x_train = prepared.tensors_train["raw"].detach().cpu().numpy()
    x_val = prepared.tensors_val["raw"].detach().cpu().numpy()
    x_test = prepared.tensors_test["raw"].detach().cpu().numpy()
    y_train = prepared.y_train.detach().cpu().numpy()

    transformer = MiniRocket(random_state=prepared.seed)
    z_train = as_numpy_features(transformer.fit_transform(x_train))
    z_val = as_numpy_features(transformer.transform(x_val))
    z_test = as_numpy_features(transformer.transform(x_test))

    classifier_name = config["external_baseline"].get("classifier", "ridge")
    if classifier_name == "ridge":
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    elif classifier_name == "logistic":
        classifier = LogisticRegression(max_iter=2000, n_jobs=-1)
    else:
        raise ValueError(f"Unknown MiniROCKET classifier: {classifier_name}")

    classifier.fit(z_train, y_train)
    val_pred = classifier.predict(z_val)
    test_pred = classifier.predict(z_test)
    val_metrics = compute_classification_metrics(prepared.y_val.numpy(), val_pred)
    test_metrics = compute_classification_metrics(prepared.y_test.numpy(), test_pred)

    num_params = 0
    if hasattr(classifier, "coef_"):
        num_params += int(np.asarray(classifier.coef_).size)
    if hasattr(classifier, "intercept_"):
        num_params += int(np.asarray(classifier.intercept_).size)

    row = base_result_row(
        prepared,
        architecture=config["external_baseline"]["name"],
        modalities=("raw",),
        seed=prepared.seed,
        num_params=num_params,
        best_epoch=None,
        duration_sec=time.perf_counter() - started_at,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        checkpoint_path=None,
    )
    result = {
        **row,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "minirocket_num_features": int(z_train.shape[1]),
        "classifier": classifier_name,
        "config": config,
        "prepared_metadata": prepared.metadata,
    }
    return row, result


def add_baseline_deltas(
    row: dict[str, Any],
    *,
    raw_macro_f1: float | None,
    raw_larger_macro_f1: float | None,
) -> dict[str, Any]:
    row = dict(row)
    row["delta_macro_f1_vs_raw"] = None if raw_macro_f1 is None else row["macro_f1"] - raw_macro_f1
    row["delta_macro_f1_vs_raw_larger"] = (
        None if raw_larger_macro_f1 is None else row["macro_f1"] - raw_larger_macro_f1
    )
    return row


def persist_result(
    output_dir: Path,
    rows: list[dict[str, Any]],
    seen_rows: set[tuple[str, str, str, int]],
    result: dict[str, Any],
    row: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_result(result, result_path(output_dir, row["dataset"], row["architecture"], tuple(row["modalities"].split("+")), row["seed"]))
    key = row_key(row)
    if key not in seen_rows:
        rows.append(row)
        seen_rows.add(key)
    save_summary_csv(rows, output_dir / "summary.csv")
    save_json_result({"rows": make_json_safe(rows)}, output_dir / "summary.json")


def row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["dataset"]),
        str(row["architecture"]),
        str(row["modalities"]),
        int(row["seed"]),
    )


def load_existing_rows(output_dir: Path) -> list[dict[str, Any]]:
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return list(payload.get("rows", []))

    rows: list[dict[str, Any]] = []
    for path in sorted((output_dir / "runs").glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        if {"dataset", "architecture", "modalities", "seed"}.issubset(result):
            row = {key: result[key] for key in result if key not in {"training_history", "test_metrics", "val_metrics", "config", "prepared_metadata"}}
            rows.append(row)
    return rows


def find_existing_row(
    rows_by_key: dict[tuple[str, str, str, int], dict[str, Any]],
    dataset_name: str,
    architecture: str,
    modalities: tuple[str, ...],
    seed: int,
) -> dict[str, Any] | None:
    return rows_by_key.get((dataset_name, architecture, "+".join(modalities), seed))


def single_view_specs(config: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    specs = []
    for modalities in config["single_view_baselines"]:
        modality_tuple = tuple(modalities)
        if len(modality_tuple) != 1:
            raise ValueError(f"Single-view baseline must have one modality, got {modalities}")
        modality = modality_tuple[0]
        architecture = "raw_larger" if modality == "raw_larger" else f"{modality}_only"
        specs.append((architecture, modality_tuple))
    return specs


def fusion_specs(config: dict[str, Any]) -> list[tuple[str, tuple[str, ...], str]]:
    specs = []
    for block in config["fusion_experiments"]:
        modalities = tuple(block["modalities"])
        if any(modality not in ALL_MODALITIES for modality in modalities):
            raise ValueError(f"Unsupported modality in fusion block: {modalities}")
        for fusion_name in block["fusions"]:
            architecture = f"{fusion_name}__{'+'.join(modalities)}"
            specs.append((architecture, modalities, fusion_name))
    return specs


def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    config = override_config_from_args(load_config(args.config), args)
    output_dir = Path(args.output_dir)
    device = get_device()
    rows: list[dict[str, Any]] = load_existing_rows(output_dir) if args.skip_existing else []
    seen_rows = {row_key(row) for row in rows}
    rows_by_key = {row_key(row): row for row in rows}

    for dataset_name in config["datasets"]:
        for seed in config["seeds"]:
            single_specs = single_view_specs(config)
            external_specs = [
                (config["external_baseline"]["name"], ("raw",))
            ] if config.get("external_baseline", {}).get("enabled", True) else []
            fusion_run_specs = [
                (architecture, modalities)
                for architecture, modalities, _ in fusion_specs(config)
            ]
            planned_specs = [*single_specs, *external_specs, *fusion_run_specs]

            if args.skip_existing and all(
                result_path(output_dir, dataset_name, architecture, modalities, seed).is_file()
                for architecture, modalities in planned_specs
            ):
                print(f"[{dataset_name}][seed={seed}] all runs exist, skipping")
                continue

            print(f"[{dataset_name}][seed={seed}] preparing representations")
            set_seed(seed)
            prepared = prepare_dataset_for_seed(dataset_name, seed, config, use_cache=args.use_cache)

            raw_macro_f1: float | None = None
            raw_larger_macro_f1: float | None = None

            for architecture, modalities in single_specs:
                existing_row = find_existing_row(rows_by_key, dataset_name, architecture, modalities, seed)
                if args.skip_existing and result_path(output_dir, dataset_name, architecture, modalities, seed).is_file():
                    print(f"[{dataset_name}][seed={seed}] skipping existing {architecture}")
                    if existing_row is not None and architecture == "raw_only":
                        raw_macro_f1 = float(existing_row["macro_f1"])
                    if existing_row is not None and architecture == "raw_larger":
                        raw_larger_macro_f1 = float(existing_row["macro_f1"])
                    continue

                print(f"[{dataset_name}][seed={seed}] running {architecture}")
                set_seed(seed)
                model = build_single_view_model(prepared, config, modalities[0])
                row, result = run_deep_model(
                    prepared,
                    config,
                    architecture=architecture,
                    modalities=modalities,
                    model=model,
                    output_dir=output_dir,
                    device=device,
                )
                if architecture == "raw_only":
                    raw_macro_f1 = row["macro_f1"]
                if architecture == "raw_larger":
                    raw_larger_macro_f1 = row["macro_f1"]
                row = add_baseline_deltas(
                    row,
                    raw_macro_f1=raw_macro_f1,
                    raw_larger_macro_f1=raw_larger_macro_f1,
                )
                result.update({"delta_macro_f1_vs_raw": row["delta_macro_f1_vs_raw"]})
                result.update({"delta_macro_f1_vs_raw_larger": row["delta_macro_f1_vs_raw_larger"]})
                persist_result(output_dir, rows, seen_rows, result, row)
                rows_by_key[row_key(row)] = row

            if config.get("external_baseline", {}).get("enabled", True):
                architecture = config["external_baseline"]["name"]
                modalities = ("raw",)
                if args.skip_existing and result_path(output_dir, dataset_name, architecture, modalities, seed).is_file():
                    print(f"[{dataset_name}][seed={seed}] skipping existing MiniROCKET")
                    continue

                print(f"[{dataset_name}][seed={seed}] running MiniROCKET")
                row, result = run_minirocket(prepared, config)
                row = add_baseline_deltas(
                    row,
                    raw_macro_f1=raw_macro_f1,
                    raw_larger_macro_f1=raw_larger_macro_f1,
                )
                result.update({"delta_macro_f1_vs_raw": row["delta_macro_f1_vs_raw"]})
                result.update({"delta_macro_f1_vs_raw_larger": row["delta_macro_f1_vs_raw_larger"]})
                persist_result(output_dir, rows, seen_rows, result, row)
                rows_by_key[row_key(row)] = row

            for architecture, modalities, fusion_name in fusion_specs(config):
                if args.skip_existing and result_path(output_dir, dataset_name, architecture, modalities, seed).is_file():
                    print(f"[{dataset_name}][seed={seed}] skipping existing {architecture}")
                    continue

                print(f"[{dataset_name}][seed={seed}] running {architecture}")
                set_seed(seed)
                model = build_fusion_model(prepared, config, modalities, fusion_name)
                row, result = run_deep_model(
                    prepared,
                    config,
                    architecture=architecture,
                    modalities=modalities,
                    model=model,
                    output_dir=output_dir,
                    device=device,
                )
                row = add_baseline_deltas(
                    row,
                    raw_macro_f1=raw_macro_f1,
                    raw_larger_macro_f1=raw_larger_macro_f1,
                )
                result.update({"fusion": fusion_name})
                result.update({"delta_macro_f1_vs_raw": row["delta_macro_f1_vs_raw"]})
                result.update({"delta_macro_f1_vs_raw_larger": row["delta_macro_f1_vs_raw_larger"]})
                persist_result(output_dir, rows, seen_rows, result, row)
                rows_by_key[row_key(row)] = row

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether fusion helps over raw representations.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
