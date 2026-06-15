"""2D-CNN encoder for Markov Transition Field (MTF) images."""

import torch
import torch.nn as nn


class MTFConvBlock(nn.Module):
    """2D convolution block: Conv2d → BatchNorm → GELU → MaxPool2d → Dropout2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MTFEncoder(nn.Module):
    """Encode MTF images into a fixed-size embedding.

    Architecture: stacked :class:`MTFConvBlock` layers → global average pool
    → linear projection to ``d_model``.

    Forward
    -------
    x_mtf : Tensor, shape ``(batch, channels, height, width)``
        MTF image tensor (typically ``channels=1``).
    Returns ``h_mtf`` of shape ``(batch, d_model)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in hidden_channels:
            layers.append(MTFConvBlock(current_channels, out_channels, dropout=dropout))
            current_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_mtf: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x_mtf)
        z = self.global_pool(z).flatten(1)
        return self.projection(z)
