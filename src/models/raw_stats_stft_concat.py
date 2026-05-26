import torch
import torch.nn as nn

from .cnn_encoder import ClassificationHead, RawTimeSeriesEncoder
from .multi_concat import MultiConcatFusionMLP
from .stats_encoder import StatisticalEncoder
from .stft_encoder import STFTEncoder


class RawStatsSTFTConcatClassifier(nn.Module):
    """raw time series + statistical features + STFT spectrogram."""

    def __init__(
        self,
        raw_in_channels: int,
        stat_in_features: int,
        image_in_channels: int,
        num_classes: int,
        d_model: int = 128,
        raw_hidden_channels: tuple[int, ...] = (64, 128, 128),
        stat_hidden_dims: tuple[int, ...] = (128, 64),
        image_hidden_channels: tuple[int, ...] = (32, 64, 128),
        fusion_hidden_dim: int = 128,
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.raw_encoder = RawTimeSeriesEncoder(
            in_channels=raw_in_channels,
            d_model=d_model,
            hidden_channels=raw_hidden_channels,
            kernel_size=kernel_size,
            dropout=raw_dropout,
        )
        self.stat_encoder = StatisticalEncoder(
            in_features=stat_in_features,
            d_model=d_model,
            hidden_dims=stat_hidden_dims,
            dropout=stat_dropout,
        )
        self.stft_encoder = STFTEncoder(
            in_channels=image_in_channels,
            d_model=d_model,
            hidden_channels=image_hidden_channels,
            dropout=image_dropout,
        )
        self.fusion = MultiConcatFusionMLP(
            n_inputs=3,
            d_model=d_model,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )
        self.head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=d_model,
            dropout=head_dropout,
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        x_stat: torch.Tensor,
        x_stft: torch.Tensor,
    ) -> torch.Tensor:
        h_final = self.fusion(
            self.raw_encoder(x_raw),
            self.stat_encoder(x_stat),
            self.stft_encoder(x_stft),
        )
        return self.head(h_final)
