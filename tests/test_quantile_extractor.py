"""Tests for :class:`~statistical.quantile_extractor.TorchQuantileExtractor`."""

import pytest
import torch

from statistical.quantile_extractor import (
    STAT_METHODS_GLOBAL_TORCH,
    STAT_METHODS_TORCH,
    TorchQuantileExtractor,
)


@pytest.fixture
def batch_ts() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(6, 80, dtype=torch.float32)


def test_generate_features_global_plus_local_window_zero(batch_ts: torch.Tensor) -> None:
    extractor = TorchQuantileExtractor(
        params={
            "window_size": 0,
            "stride": 1,
            "add_global_features": True,
        }
    )
    out = extractor.generate_features_from_ts(batch_ts)
    assert out.ndim == 2
    expected = len(STAT_METHODS_GLOBAL_TORCH) + len(STAT_METHODS_TORCH)
    assert out.shape == (batch_ts.shape[0], expected)
    assert out.device.type == "cpu"


def test_generate_features_locals_only_when_no_global(batch_ts: torch.Tensor) -> None:
    extractor = TorchQuantileExtractor(
        params={"window_size": 0, "add_global_features": False},
    )
    out = extractor.generate_features_from_ts(batch_ts)
    expected = len(STAT_METHODS_TORCH)
    assert out.shape == (batch_ts.shape[0], expected)


def test_1d_series_unsqueezed(batch_ts: torch.Tensor) -> None:
    extractor = TorchQuantileExtractor(params={"window_size": 0})
    series = batch_ts[0]
    assert series.ndim == 1
    out = extractor.generate_features_from_ts(series)
    assert out.ndim == 2 and out.shape[0] == 1


def test_sliding_windows_stride_one(batch_ts: torch.Tensor) -> None:
    extractor = TorchQuantileExtractor(
        params={
            "window_size": 25,
            "stride": 1,
            "add_global_features": True,
        },
    )
    out = extractor.generate_features_from_ts(batch_ts)
    assert out.ndim == 2
    assert out.shape[0] == batch_ts.shape[0]


def test_hankel_path_stride_above_one(batch_ts: torch.Tensor) -> None:
    extractor = TorchQuantileExtractor(
        params={
            "window_size": 30,
            "stride": 3,
            "add_global_features": False,
        },
    )
    out = extractor.generate_features_from_ts(batch_ts)
    assert out.ndim == 2
    assert out.shape[0] == batch_ts.shape[0]


def test_deterministic_repeated_call(batch_ts: torch.Tensor) -> None:
    extractor = TorchQuantileExtractor(params={"window_size": 0})
    replica = batch_ts.clone()
    assert torch.equal(
        extractor.generate_features_from_ts(batch_ts),
        extractor.generate_features_from_ts(replica),
    )
