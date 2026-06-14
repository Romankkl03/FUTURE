"""Hankel (trajectory) matrix construction from PyTorch tensors."""

from __future__ import annotations

import torch


class HankelMatrix:
    """
    Builds a Hankel-like trajectory matrix for a batch or a single time series represented
    as a ``torch.Tensor``.

    Rows (after internal construction and transposition where applicable) correspond to
    lagged subsequences extracted along the time dimension; strides subsample sliding
    starting positions along the series.
    """

    def __init__(
        self,
        time_series: torch.Tensor,
        window_size: int | None = None,
        strides: int = 1,
    ) -> None:
        self.__time_series = time_series.squeeze()
        if self.__time_series.ndim > 1:
            self.__ts_length = self.__time_series.shape[-1]
        else:
            self.__ts_length = self.__time_series.numel()

        self.__strides = strides
        if window_size is None:
            self.__window_length = round(self.__ts_length * 0.35)
        else:
            self.__window_length = round(window_size - 1)

        self.__subseq_length = self.__ts_length - self.__window_length + 1

        self.__check_windows_length()
        if len(self.__time_series.shape) > 1:
            self.__trajectory_matrix = self.__get_2d_trajectory_matrix()
        else:
            self.__trajectory_matrix = self.__get_1d_trajectory_matrix()

    def __check_windows_length(self) -> None:
        if not 2 <= self.__window_length <= self.__ts_length / 2:
            self.__window_length = int(self.__ts_length / 3)

    def __get_1d_trajectory_matrix(self, ts: torch.Tensor | None = None) -> torch.Tensor:
        ts = self.__time_series if ts is None else ts
        T = ts.shape[0]
        W = self.__window_length
        S = self.__strides
        if S > 1:
            num_windows = T - W + 1
            trajectory = ts.as_strided(
                size=(num_windows, W),
                stride=(ts.stride(0), ts.stride(0)),
            )
            idx = torch.arange(0, num_windows, S, device=ts.device)
            trajectory = trajectory[idx]
            return trajectory.T
        i = torch.arange(W + 1, device=ts.device).unsqueeze(1)
        j = torch.arange(T - W, device=ts.device).unsqueeze(0)
        idx = i + j
        source = torch.cat([ts[: W + 1], ts[W:][1:]], dim=0)
        return source[idx]

    def __get_2d_trajectory_matrix(self) -> torch.Tensor:
        matrices = [self.__get_1d_trajectory_matrix(ts) for ts in self.__time_series]
        tensor_stacked = [m.unsqueeze(0) for m in matrices]
        return torch.concat(tensor_stacked)

    @property
    def window_length(self) -> int:
        return self.__window_length

    @property
    def time_series(self) -> torch.Tensor:
        return self.__time_series

    @property
    def sub_seq_length(self) -> int:
        return self.__subseq_length

    @window_length.setter
    def window_length(self, window_length: int) -> None:
        self.__window_length = window_length

    @property
    def trajectory_matrix(self) -> torch.Tensor:
        return self.__trajectory_matrix

    @property
    def ts_length(self) -> int:
        return self.__ts_length

    @trajectory_matrix.setter
    def trajectory_matrix(self, trajectory_matrix: torch.Tensor) -> None:
        self.__trajectory_matrix = trajectory_matrix
