import torch
import torch.nn as nn

from .cnn_encoder import ClassificationHead, RawTimeSeriesEncoder
from .gaf_encoder import GAFEncoder
from .raw_stats_concat import ConcatFusionMLP


class RawGAFConcatClassifier(nn.Module):
    """
    raw time series + GAF image через concat fusion.

    Inputs:
        x_raw: [batch, channels, time]
        x_gaf: [batch, channels, height, width]

    Output:
        logits: [batch, num_classes]
    """

    def __init__(
        self,
        raw_in_channels: int,
        image_in_channels: int,
        num_classes: int,
        d_model: int = 128,
        raw_hidden_channels: tuple[int, ...] = (64, 128, 128),
        image_hidden_channels: tuple[int, ...] = (32, 64, 128),
        fusion_hidden_dim: int = 128,
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
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
        self.image_encoder = GAFEncoder(
            in_channels=image_in_channels,
            d_model=d_model,
            hidden_channels=image_hidden_channels,
            dropout=image_dropout,
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

    def forward(self, x_raw: torch.Tensor, x_gaf: torch.Tensor) -> torch.Tensor:
        h_raw = self.raw_encoder(x_raw)
        h_gaf = self.image_encoder(x_gaf)
        h_final = self.fusion(h_raw, h_gaf)
        return self.head(h_final)
