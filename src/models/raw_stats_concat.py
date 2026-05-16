import torch
import torch.nn as nn

from .cnn_encoder import ClassificationHead, RawTimeSeriesEncoder
from .stats_encoder import StatisticalEncoder


class ConcatFusionMLP(nn.Module):
    """
    concat(h_raw, h_stat) -> h_final
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_raw: torch.Tensor, h_stat: torch.Tensor) -> torch.Tensor:
        h_fused = torch.cat([h_raw, h_stat], dim=1)
        return self.fusion(h_fused)


class RawStatsConcatClassifier(nn.Module):
    """
    Experiment 2:
    raw time series + statistical features
    через simple concat fusion.

    Inputs:
        x_raw:  [batch, channels, time]
        x_stat: [batch, num_features]

    Output:
        logits: [batch, num_classes]
    """

    def __init__(
        self,
        raw_in_channels: int,
        stat_in_features: int,
        num_classes: int,
        d_model: int = 128,
        raw_hidden_channels: tuple[int, ...] = (64, 128, 128),
        stat_hidden_dims: tuple[int, ...] = (128, 64),
        fusion_hidden_dim: int = 128,
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
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

        self.fusion = ConcatFusionMLP(
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
    ) -> torch.Tensor:
        """
        x_raw:  [batch, channels, time]
        x_stat: [batch, num_features]
        """

        h_raw = self.raw_encoder(x_raw)
        h_stat = self.stat_encoder(x_stat)

        h_final = self.fusion(h_raw, h_stat)
        logits = self.head(h_final)

        return logits