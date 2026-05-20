from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from image_transformation.methods.stft_transformation import STFTSpectrogram
from statistical.quantile_extractor import (
    STAT_METHODS_GLOBAL_TORCH,
    STAT_METHODS_TORCH,
    TorchQuantileExtractor,
)

IMAGE_LOG1P_TRANSFORMS = frozenset({"STFT", STFTSpectrogram})


class RawStatsTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_raw: torch.Tensor,
        X_stat: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self.X_raw = X_raw.float()
        self.X_stat = X_stat.float()
        self.y = y.long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_raw[idx], self.X_stat[idx], self.y[idx]


class RawImageTensorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X_raw: torch.Tensor,
        X_img: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self.X_raw = X_raw.float()
        self.X_img = X_img.float()
        self.y = y.long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_raw[idx], self.X_img[idx], self.y[idx]


def make_tensor_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    batch_size: int,
    train_shuffle: bool = True,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader]:
    X_train_t = torch.as_tensor(X_train).float()
    y_train_t = torch.as_tensor(y_train).long()
    X_test_t = torch.as_tensor(X_test).float()
    y_test_t = torch.as_tensor(y_test).long()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=train_shuffle,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def make_dataset_tensor_loaders(
    dataset: Any,
    *,
    batch_size: int,
    train_shuffle: bool = True,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader]:
    return make_tensor_loaders(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
        drop_last=drop_last,
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


def drop_stat_features(
    F_train: torch.Tensor,
    F_test: torch.Tensor,
    feature_names: list[str],
    drop_features: Iterable[str] = (),
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    drop_features = set(drop_features)
    drop_idx = [idx for idx, name in enumerate(feature_names) if name in drop_features]
    keep_idx = [idx for idx in range(len(feature_names)) if idx not in drop_idx]

    if not drop_idx:
        return F_train, F_test, feature_names, []

    keep_idx_t = torch.tensor(keep_idx, dtype=torch.long, device=F_train.device)
    kept_feature_names = [feature_names[idx] for idx in keep_idx]
    dropped_feature_names = [feature_names[idx] for idx in drop_idx]
    return F_train[:, keep_idx_t], F_test[:, keep_idx_t], kept_feature_names, dropped_feature_names


def replace_nonfinite_with_train_means(
    F_train: torch.Tensor,
    F_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    F_train = F_train.float()
    F_test = F_test.float()

    finite_train = torch.isfinite(F_train)
    train_with_nan = torch.where(finite_train, F_train, torch.nan)
    col_mean = torch.nanmean(train_with_nan, dim=0, keepdim=True)
    col_mean = torch.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)

    F_train = torch.where(finite_train, F_train, col_mean)
    F_test = torch.where(torch.isfinite(F_test), F_test, col_mean)
    return F_train, F_test


def standardize_from_train(
    F_train: torch.Tensor,
    F_test: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    F_train, F_test = replace_nonfinite_with_train_means(F_train, F_test)
    mean = F_train.mean(dim=0, keepdim=True)
    std = F_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    F_train = (F_train - mean) / std
    F_test = (F_test - mean) / std
    return torch.nan_to_num(F_train), torch.nan_to_num(F_test)


def prepare_stat_features(
    X_train_2d: torch.Tensor,
    X_test_2d: torch.Tensor,
    stat_params: dict[str, Any],
    *,
    batch_size: int = 512,
    drop_features: Iterable[str] = (),
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    extractor = TorchQuantileExtractor(stat_params)
    F_train = feats_batched(extractor, X_train_2d, batch_size=batch_size)
    F_test = feats_batched(extractor, X_test_2d, batch_size=batch_size)
    feature_names = get_stat_feature_names(stat_params, F_train.shape[1])
    F_train, F_test, feature_names, dropped_feature_names = drop_stat_features(
        F_train,
        F_test,
        feature_names,
        drop_features=drop_features,
    )
    F_train, F_test = standardize_from_train(F_train, F_test)
    return F_train.float(), F_test.float(), feature_names, dropped_feature_names


def make_stat_loaders(
    F_train: torch.Tensor,
    F_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    *,
    batch_size: int,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(F_train.float(), torch.as_tensor(y_train).long()),
        batch_size=batch_size,
        shuffle=train_shuffle,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(F_test.float(), torch.as_tensor(y_test).long()),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def make_prepared_stat_loaders(
    X_train_2d: torch.Tensor,
    X_test_2d: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    stat_params: dict[str, Any],
    *,
    batch_size: int,
    feature_batch_size: int = 512,
    drop_features: Iterable[str] = (),
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, int, list[str], list[str]]:
    F_train, F_test, feature_names, dropped_feature_names = prepare_stat_features(
        X_train_2d,
        X_test_2d,
        stat_params,
        batch_size=feature_batch_size,
        drop_features=drop_features,
    )
    train_loader, test_loader = make_stat_loaders(
        F_train,
        F_test,
        y_train,
        y_test,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
    )
    return train_loader, test_loader, F_train.shape[1], feature_names, dropped_feature_names


def make_raw_stats_loaders(
    X_train_raw: torch.Tensor,
    X_test_raw: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    F_train: torch.Tensor,
    F_test: torch.Tensor,
    *,
    batch_size: int,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        RawStatsTensorDataset(torch.as_tensor(X_train_raw), F_train, torch.as_tensor(y_train)),
        batch_size=batch_size,
        shuffle=train_shuffle,
        drop_last=False,
    )
    test_loader = DataLoader(
        RawStatsTensorDataset(torch.as_tensor(X_test_raw), F_test, torch.as_tensor(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def make_prepared_raw_stats_loaders(
    X_train_raw: torch.Tensor,
    X_test_raw: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    X_train_2d: torch.Tensor,
    X_test_2d: torch.Tensor,
    stat_params: dict[str, Any],
    *,
    batch_size: int,
    feature_batch_size: int = 512,
    drop_features: Iterable[str] = (),
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, int, list[str], list[str]]:
    F_train, F_test, feature_names, dropped_feature_names = prepare_stat_features(
        X_train_2d,
        X_test_2d,
        stat_params,
        batch_size=feature_batch_size,
        drop_features=drop_features,
    )
    train_loader, test_loader = make_raw_stats_loaders(
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        F_train,
        F_test,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
    )
    return train_loader, test_loader, F_train.shape[1], feature_names, dropped_feature_names


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


def standardize_images_from_train(
    X_train_img: torch.Tensor,
    X_test_img: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = X_train_img.mean(dim=(0, 2, 3), keepdim=True)
    std = X_train_img.std(dim=(0, 2, 3), unbiased=False, keepdim=True).clamp_min(eps)
    return (X_train_img - mean) / std, (X_test_img - mean) / std


def apply_image_log1p(X_train_img: torch.Tensor, X_test_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.log1p(X_train_img.clamp_min(0)), torch.log1p(X_test_img.clamp_min(0))


def prepare_image_tensors(
    transformer: Any,
    X_train_2d: torch.Tensor,
    X_test_2d: torch.Tensor,
    *,
    image_batch_size: int = 256,
    use_log1p: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    X_train_img = transform_images_batched(transformer, X_train_2d, batch_size=image_batch_size)
    X_test_img = transform_images_batched(transformer, X_test_2d, batch_size=image_batch_size)

    if use_log1p:
        X_train_img, X_test_img = apply_image_log1p(X_train_img, X_test_img)

    transform_shape = tuple(X_train_img.shape[1:])
    X_train_img = to_cnn_images(X_train_img)
    X_test_img = to_cnn_images(X_test_img)
    cnn_shape = tuple(X_train_img.shape[1:])
    X_train_img, X_test_img = standardize_images_from_train(X_train_img, X_test_img)
    return X_train_img.float(), X_test_img.float(), transform_shape, cnn_shape


def prepare_named_image_tensors(
    transform_name: str,
    transformer: Any,
    X_train_2d: torch.Tensor,
    X_test_2d: torch.Tensor,
    *,
    image_batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    return prepare_image_tensors(
        transformer,
        X_train_2d,
        X_test_2d,
        image_batch_size=image_batch_size,
        use_log1p=transform_name in IMAGE_LOG1P_TRANSFORMS,
    )


def make_image_loaders(
    X_train_img: torch.Tensor,
    X_test_img: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    *,
    batch_size: int,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(X_train_img.float(), torch.as_tensor(y_train).long()),
        batch_size=batch_size,
        shuffle=train_shuffle,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_img.float(), torch.as_tensor(y_test).long()),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def make_prepared_image_loaders(
    transformer: Any,
    X_train_2d: torch.Tensor,
    X_test_2d: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    *,
    batch_size: int,
    image_batch_size: int = 256,
    use_log1p: bool = False,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, tuple[int, ...], tuple[int, ...]]:
    X_train_img, X_test_img, transform_shape, cnn_shape = prepare_image_tensors(
        transformer,
        X_train_2d,
        X_test_2d,
        image_batch_size=image_batch_size,
        use_log1p=use_log1p,
    )
    train_loader, test_loader = make_image_loaders(
        X_train_img,
        X_test_img,
        y_train,
        y_test,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
    )
    return train_loader, test_loader, transform_shape, cnn_shape


def make_raw_image_loaders(
    X_train_raw: torch.Tensor,
    X_test_raw: torch.Tensor,
    X_train_img: torch.Tensor,
    X_test_img: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    *,
    batch_size: int,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        RawImageTensorDataset(torch.as_tensor(X_train_raw), X_train_img, torch.as_tensor(y_train)),
        batch_size=batch_size,
        shuffle=train_shuffle,
        drop_last=False,
    )
    test_loader = DataLoader(
        RawImageTensorDataset(torch.as_tensor(X_test_raw), X_test_img, torch.as_tensor(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def make_prepared_raw_image_loaders(
    transformer: Any,
    X_train_raw: torch.Tensor,
    X_test_raw: torch.Tensor,
    X_train_2d: torch.Tensor,
    X_test_2d: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    *,
    batch_size: int,
    image_batch_size: int = 256,
    use_log1p: bool = False,
    train_shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, tuple[int, ...], tuple[int, ...]]:
    X_train_img, X_test_img, transform_shape, cnn_shape = prepare_image_tensors(
        transformer,
        X_train_2d,
        X_test_2d,
        image_batch_size=image_batch_size,
        use_log1p=use_log1p,
    )
    train_loader, test_loader = make_raw_image_loaders(
        X_train_raw,
        X_test_raw,
        X_train_img,
        X_test_img,
        y_train,
        y_test,
        batch_size=batch_size,
        train_shuffle=train_shuffle,
    )
    return train_loader, test_loader, transform_shape, cnn_shape
