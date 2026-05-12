import pytest
import torch

from statistical.hankel import HankelMatrix


@pytest.fixture
def series_1d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(128, dtype=torch.float32)


@pytest.fixture
def batch_2d() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(4, 96, dtype=torch.float32)


def test_hankel_1d_strided_trajectory(series_1d: torch.Tensor) -> None:
    h = HankelMatrix(time_series=series_1d, window_size=40, strides=2)
    h_mat = h.trajectory_matrix
    assert isinstance(h_mat, torch.Tensor)
    assert h_mat.device == series_1d.device
    assert h_mat.ndim == 2
    assert h_mat.shape[0] == h.window_length


def test_hankel_batch_concat(batch_2d: torch.Tensor) -> None:
    h = HankelMatrix(time_series=batch_2d, window_size=36, strides=1)
    h_mat = h.trajectory_matrix
    assert h_mat.ndim == 3
    assert h_mat.shape[0] == batch_2d.shape[0]


def test_hankel_window_length_fallback() -> None:
    ts = torch.arange(40, dtype=torch.float32)
    h = HankelMatrix(time_series=ts, window_size=250, strides=1)
    assert h.window_length == int(ts.numel() / 3)
