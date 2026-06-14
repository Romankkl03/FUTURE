from collections.abc import Mapping
from typing import Any

import torch

from src.recurrence.kernel_matrix import TorchTSTransformer
from src.recurrence.sequences import RecurrenceFeatureExtractorTorch
from src.statistical.hankel import HankelMatrix


RECURRENCE_FEATURE_NAMES = [
    "RR",
    "DET",
    "ADLL",
    "LDLL",
    "DIV",
    "EDL",
    "LAM",
    "AVLL",
    "LVLL",
    "EVL",
    "AWLL",
    "LWLL",
    "EWLL",
    "RDRR",
    "RLD",
]


class RecurrenceExtractor:
    """
    A feature extractor for time series based on recurrence plots and recurrence quantification analysis (RQA).

    This class transforms time series into recurrence matrices and extracts statistical features
    such as recurrence rate, determinism, laminarity, and line-based metrics.
    It supports both feature-based and image-based representations of recurrence plots.

    Attributes:
        window_size (int): The size of the sliding window for trajectory matrix construction.
        stride (int): The stride for the sliding window.
        rec_metric (str): The distance metric used for recurrence plot construction (e.g., 'cosine').
        image_mode (bool): If True, returns the recurrence plot as a 3D tensor (image-like representation).
                          If False, returns RQA features as a 1D tensor.
        transformer: The transformer class used to convert time series to recurrence matrices.
        extractor: The extractor class used to compute RQA features from recurrence matrices.
    """

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        params = dict(params or {})
        self.window_size = int(params.get("window_size", 0))
        self.stride = int(params.get("stride", 1))
        self.rec_metric = str(params.get("rec_metric", "cosine"))
        self.image_mode = bool(params.get("image_mode", False))
        self.threshold = params.get("threshold", None)
        self.p = float(params.get("p", 3))
        self.transformer = TorchTSTransformer
        self.extractor = RecurrenceFeatureExtractorTorch

    def __repr__(self) -> str:
        return "RecurrenceExtractor(tensor-only)"

    def _generate_features_from_ts(self, ts: torch.Tensor) -> torch.Tensor:
        ts = ts.float().squeeze()
        if self.window_size > 0:
            trajectory_transformer = HankelMatrix(
                time_series=ts,
                window_size=self.window_size,
                strides=self.stride,
            )
            ts = trajectory_transformer.trajectory_matrix
            if ts.ndim > 2:
                ts = ts.reshape(-1, ts.shape[-1])

        specter = self.transformer(
            time_series=ts,
            rec_metric=self.rec_metric,
            p=self.p,
        )

        if not self.image_mode:
            recurrence_matrix = specter.ts_to_recurrence_matrix(threshold=self.threshold)
            feature_dict = self.extractor(
                recurrence_matrix=recurrence_matrix,
            ).quantification_analysis()
            features = torch.stack(
                [
                    torch.as_tensor(
                        feature_dict[name],
                        device=recurrence_matrix.device,
                        dtype=recurrence_matrix.dtype,
                    ).flatten()[0]
                    for name in RECURRENCE_FEATURE_NAMES
                ]
            )
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            features = specter.ts_to_3d_recurrence_matrix().float()
        return features.detach().cpu()

    @staticmethod
    def _as_batch(ts: torch.Tensor) -> torch.Tensor:
        if not isinstance(ts, torch.Tensor):
            ts = torch.as_tensor(ts)
        ts = ts.float()
        if ts.ndim == 1:
            return ts.unsqueeze(0)
        if ts.ndim == 2:
            return ts
        if ts.ndim == 3:
            return ts
        raise ValueError(f"Expected 1D, 2D or 3D tensor, got shape={tuple(ts.shape)}")

    def generate_recurrence_features(self, ts: torch.Tensor) -> torch.Tensor:
        ts_batch = self._as_batch(ts)
        features = [self._generate_features_from_ts(sample) for sample in ts_batch]
        return torch.stack(features, dim=0)

    def generate_features_from_ts(self, ts: torch.Tensor) -> torch.Tensor:
        return self.generate_recurrence_features(ts)
