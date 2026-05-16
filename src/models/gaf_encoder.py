import torch
import torch.nn as nn

from .cnn_encoder import ClassificationHead


class GAFConvBlock(nn.Module):
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


class GAFEncoder(nn.Module):
    """
    CNN encoder for GAF images.

    Input:
        x_gaf: [batch, channels, height, width]

    Output:
        h_gaf: [batch, d_model]
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
            layers.append(GAFConvBlock(current_channels, out_channels, dropout=dropout))
            current_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_gaf: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x_gaf)
        z = self.global_pool(z).flatten(1)
        return self.projection(z)


class GAFClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 128,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
        encoder_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GAFEncoder(
            in_channels=in_channels,
            d_model=d_model,
            hidden_channels=hidden_channels,
            dropout=encoder_dropout,
        )
        self.head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=d_model,
            dropout=head_dropout,
        )

    def forward(self, x_gaf: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x_gaf))
