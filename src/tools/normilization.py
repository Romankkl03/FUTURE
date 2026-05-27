from __future__ import annotations

import torch


def per_sample_z_normalize(
    X: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    X = torch.as_tensor(X).float()
    mean = X.mean(dim=dim, keepdim=True)
    std = X.std(dim=dim, unbiased=False, keepdim=True).clamp_min(eps)
    return torch.nan_to_num((X - mean) / std)


def per_sample_minmax_scale(
    X: torch.Tensor,
    *,
    feature_range: tuple[float, float] = (-1.0, 1.0),
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    X = torch.as_tensor(X).float()
    min_value = X.amin(dim=dim, keepdim=True)
    max_value = X.amax(dim=dim, keepdim=True)
    scale = (max_value - min_value).clamp_min(eps)

    low, high = feature_range
    X_scaled = (X - min_value) / scale
    X_scaled = X_scaled * (high - low) + low
    return torch.nan_to_num(X_scaled)


def z_normalize_pair(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        per_sample_z_normalize(X_train, dim=dim, eps=eps),
        per_sample_z_normalize(X_test, dim=dim, eps=eps),
    )


def minmax_scale_pair(
    X_train: torch.Tensor,
    X_test: torch.Tensor,
    *,
    feature_range: tuple[float, float] = (-1.0, 1.0),
    dim: int = -1,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        per_sample_minmax_scale(X_train, feature_range=feature_range, dim=dim, eps=eps),
        per_sample_minmax_scale(X_test, feature_range=feature_range, dim=dim, eps=eps),
    )


def log1p_nonnegative(X: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.as_tensor(X).float().clamp_min(0))
