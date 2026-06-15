from __future__ import annotations

from typing import Any

import torch

from src.representations.statistical.quantile_extractor import (
    STAT_METHODS_GLOBAL_TORCH,
    STAT_METHODS_TORCH,
    TorchQuantileExtractor,
)


def as_univariate_feature_tensor(X: torch.Tensor) -> torch.Tensor:
    X = torch.as_tensor(X).float()
    if X.ndim == 3 and X.shape[1] == 1:
        return X.squeeze(1)
    return X


def feats_batched(
    extractor: TorchQuantileExtractor,
    X_2d: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(X_2d), batch_size):
        chunk = X_2d[start:start + batch_size]
        chunks.append(extractor.generate_features_from_ts(chunk))
    return torch.cat(chunks, dim=0)


def get_stat_feature_names(stat_params: dict[str, Any], num_features: int) -> list[str]:
    names = list(STAT_METHODS_GLOBAL_TORCH.keys()) if stat_params.get("add_global_features", True) else []
    if stat_params.get("window_size", 0) == 0:
        names.extend(STAT_METHODS_TORCH.keys())
        return names

    n_window_features = len(STAT_METHODS_TORCH)
    n_windows = (num_features - len(names)) // n_window_features
    for window_idx in range(n_windows):
        names.extend(f"window_{window_idx}_{name}" for name in STAT_METHODS_TORCH.keys())
    return names


def transform_images_batched(
    transformer: Any,
    X_2d: torch.Tensor,
    batch_size: int = 256,
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for start in range(0, len(X_2d), batch_size):
        chunk = X_2d[start:start + batch_size].float()
        images.append(transformer.transform(chunk).detach().cpu())
    return torch.cat(images, dim=0)


def to_cnn_images(X_img: torch.Tensor) -> torch.Tensor:
    if X_img.ndim != 3:
        raise ValueError(f"Expected image tensor [batch, height, width], got {tuple(X_img.shape)}")
    return X_img.unsqueeze(1).float()
