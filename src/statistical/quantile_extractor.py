"""Statistical feature extraction for time-series tensors."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from statistical.hankel import HankelMatrix
from statistical.stat_features import (
    autocorrelation_torch,
    ben_corr_torch,
    crest_factor_torch,
    energy_torch,
    hjorth_complexity_torch,
    hjorth_mobility_torch,
    hurst_exponent_torch,
    interquantile_range_torch,
    kurtosis_torch,
    max_torch,
    mean_ema_torch,
    mean_moving_median_torch,
    mean_ptp_distance_torch,
    mean_torch,
    median_torch,
    min_torch,
    n_peaks_torch,
    pfd_torch,
    ptp_amp_torch,
    q25_torch,
    q5_torch,
    q75_torch,
    q95_torch,
    shannon_entropy_torch,
    skewness_torch,
    slope_torch,
    std_torch,
    zero_crossing_rate_torch,
)

STAT_METHODS_TORCH: dict[str, Callable[..., Any]] = {
    "mean_": mean_torch,
    "median_": median_torch,
    "std_": std_torch,
    "max_": max_torch,
    "min_": min_torch,
    "q5_": q5_torch,
    "q25_": q25_torch,
    "q75_": q75_torch,
    "q95_": q95_torch,
}

STAT_METHODS_GLOBAL_TORCH: dict[str, Callable[..., Any]] = {
    "skewness_": skewness_torch,
    "kurtosis_": kurtosis_torch,
    "n_peaks_": n_peaks_torch,
    "slope_": slope_torch,
    "ben_corr_": ben_corr_torch,
    "interquartile_range_": interquantile_range_torch,
    "energy_": energy_torch,
    "cross_rate_": zero_crossing_rate_torch,
    "autocorrelation_": autocorrelation_torch,
    "shannon_entropy_": shannon_entropy_torch,
    "ptp_amplitude_": ptp_amp_torch,
    "mean_ptp_distance_": mean_ptp_distance_torch,
    "crest_factor_": crest_factor_torch,
    "mean_ema_": mean_ema_torch,
    "mean_moving_median_": mean_moving_median_torch,
    "hjorth_mobility_": hjorth_mobility_torch,
    "hjorth_complexity_": hjorth_complexity_torch,
    "hurst_exponent_": hurst_exponent_torch,
    "petrosian_fractal_dimension_": pfd_torch,
}


def _as_float_tensor_on_device(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Coerce scalar outputs from stat helpers to tensors for stacking or concatenation."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if value.device != device else value
    return torch.tensor(value, device=device, dtype=dtype)


def _stack_statistics_row(global_features: list[Any], *, ts: torch.Tensor) -> torch.Tensor:
    """Build ``(1, n_features)`` when the time axis was squeezed to length without a batch dimension."""
    tensors = [_as_float_tensor_on_device(f, device=ts.device, dtype=ts.dtype) for f in global_features]
    row = torch.stack([t.flatten()[0] for t in tensors], dim=0)
    return row.unsqueeze(0).to(ts.device)


class TorchQuantileExtractor:
    """
    Computes local (window-based) and optional global statistical features from
    time-series tensors.

    When ``window_size`` is non-zero, windows are built along the time axis (with an
    optional Hankel trajectory when ``stride > 1``); per-window statistics are stacked
    and optionally concatenated with series-level (global) statistics.

    Parameters are passed as a plain mapping (e.g. ``dict``); unknown keys are ignored.
    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        params = dict(params or {})
        self.window_size: int = int(params.get("window_size", 0))
        self.stride: int = int(params.get("stride", 1))
        self.add_global_features: bool = bool(params.get("add_global_features", True))
        self.use_sliding_window: bool = bool(params.get("use_sliding_window", True))
        self.max_elements: int = int(params.get("max_elements", 50_000_000))
        self.is_multichannel: bool = False

    def get_statistical_features_torch(
        self,
        time_series: torch.Tensor,
        add_global_features: bool = False,
        axis: int = -1,
        max_elements: int | None = None,
    ) -> list[torch.Tensor | float | int]:
        """
        Apply the configured set of statistical functions along ``axis``.

        Large batches are processed in chunks along the batch dimension to limit memory.
        """
        max_elements = self.max_elements if max_elements is None else max_elements
        if self.is_multichannel:
            if time_series.ndim == 3:
                batch = time_series.shape[0]
                time_series = time_series.reshape(batch, -1)
            elif time_series.ndim > 3:
                batch, b, *rest = time_series.shape
                time_series = time_series.reshape(batch, b, -1)

        list_of_methods = list(
            STAT_METHODS_GLOBAL_TORCH.items()
            if add_global_features
            else STAT_METHODS_TORCH.items()
        )

        if time_series.numel() <= max_elements:
            return [method(time_series, axis) for _, method in list_of_methods]

        bsz = time_series.shape[0]
        elems_per_sample = time_series[0].numel()
        batch_size = max(1, max_elements // max(elems_per_sample, 1))
        accumulators: list[list[Any]] = [[] for _ in list_of_methods]

        for start in range(0, bsz, batch_size):
            end = min(start + batch_size, bsz)
            ts_batch = time_series[start:end]
            for i, (_, method) in enumerate(list_of_methods):
                accumulators[i].append(method(ts_batch, axis))

        device = time_series.device
        dtype = time_series.dtype if torch.is_floating_point(time_series) else torch.float32
        merged: list[torch.Tensor | float | int] = []
        for parts in accumulators:
            ten_parts = [_as_float_tensor_on_device(p, device=device, dtype=dtype) for p in parts]
            normalized = [(tp.unsqueeze(0) if tp.ndim == 0 else tp) for tp in ten_parts]
            merged.append(torch.cat(normalized, dim=0))
        return merged

    def apply_window_for_stat_feature_torch(
        self,
        ts_data: torch.Tensor,
        feature_generator: Callable[..., Sequence[torch.Tensor | float | int]],
        window_size: int | None = None,
    ) -> torch.Tensor:
        """
        Segment the series into windows, compute statistics per window, and return a tensor
        shaped for concatenation with global features.
        """
        axis = ts_data.ndim - 1
        if window_size is None:
            window_size = round(ts_data.shape[axis] / 10)
        else:
            window_size = round(ts_data.shape[axis] * (window_size / 100))
        window_size = max(window_size, 5)

        if self.use_sliding_window:
            if self.stride > 1:
                subseq_set = HankelMatrix(
                    time_series=ts_data,
                    window_size=window_size,
                    strides=self.stride,
                ).trajectory_matrix
            else:
                window_length = ts_data.shape[axis] - window_size
                subseq_set = ts_data.unfold(
                    dimension=axis,
                    size=window_length,
                    step=self.stride,
                )
            if subseq_set.ndim > 2:
                subseq_set = subseq_set.transpose(1, 2)
            else:
                subseq_set = subseq_set.T
        else:
            t_len = ts_data.shape[1]
            num_windows = t_len // window_size
            t_eff = num_windows * window_size
            ts_cut = ts_data[:, :t_eff]
            subseq_set = ts_cut.reshape(2, num_windows, window_size)

        features = feature_generator(subseq_set)
        features = torch.stack(
            [_as_float_tensor_on_device(f, device=ts_data.device, dtype=ts_data.dtype) for f in features],
            dim=0,
        ).to(ts_data.device)
        if features.ndim > 2:
            features = features.permute(1, 2, 0)
        else:
            features = features.T
        return features

    def _flatten_sliding_window_features(self, window_stat_features: torch.Tensor) -> torch.Tensor:
        """Merge window and statistic dimensions into one feature dimension per batch row."""
        if window_stat_features.ndim > 2:
            out = window_stat_features.reshape(
                window_stat_features.shape[0],
                window_stat_features.shape[-1] * window_stat_features.shape[-2],
            )
        else:
            out = window_stat_features.reshape(
                window_stat_features.shape[-1] * window_stat_features.shape[-2]
            )
        return out.squeeze()

    def extract_stats_features_torch(self, ts: torch.Tensor, axis: int = -1) -> torch.Tensor:
        """
        Compute global (optional) and window-based statistics and concatenate them when
        ``add_global_features`` is True.
        """
        global_features = self.get_statistical_features_torch(
            ts, add_global_features=self.add_global_features, axis=axis
        )
        global_features = [f for f in global_features if f is not None]
        if ts.ndim == 4:
            global_features = torch.cat(global_features, dim=1).to(ts.device)
        elif ts.squeeze().ndim == 1:
            global_features = _stack_statistics_row(global_features, ts=ts)
        else:
            global_features = torch.stack(global_features, dim=0).T.to(ts.device)

        if self.window_size == 0:
            window_stat_features = self.get_statistical_features_torch(ts, axis=axis)
            if ts.squeeze().ndim == 1:
                window_stat_features = _stack_statistics_row(window_stat_features, ts=ts)
            else:
                window_stat_features = torch.stack(window_stat_features, dim=0).T.to(ts.device)
        else:
            window_stat_features = self.apply_window_for_stat_feature_torch(
                ts_data=ts,
                feature_generator=lambda x: self.get_statistical_features_torch(x, axis=axis),
                window_size=self.window_size,
            )

        if self.window_size != 0:
            window_stat_features = self._flatten_sliding_window_features(window_stat_features)

        if self.add_global_features:
            return torch.cat([global_features, window_stat_features], dim=-1)
        return window_stat_features

    def generate_features_from_ts(self, ts: torch.Tensor) -> torch.Tensor:
        """
        Public entry point: accepts shape ``(T,)``, ``(B, T)``, or higher-dimensional
        batch/channel layouts handled by ``is_multichannel`` reshaping internally.

        Returns feature vectors on CPU (same behavior as the previous implementation).
        """
        ts = ts if isinstance(ts, torch.Tensor) else torch.as_tensor(ts)
        if ts.ndim == 1:
            ts = ts.unsqueeze(0)
        self.is_multichannel = ts.ndim > 2

        features = self.extract_stats_features_torch(ts, axis=-1)
        return features.detach().cpu()
