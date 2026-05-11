import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """
    Базовый 1D CNN блок:
    Conv1d → BatchNorm → GELU → Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RawTimeSeriesEncoder(nn.Module):
    """
    Raw time series encoder.

    Input:
        x: [batch, channels, time]

    Output:
        h_raw: [batch, d_model]
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        hidden_channels: tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        for out_channels in hidden_channels:
            layers.append(
                ConvBlock1D(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dropout=dropout,
                )
            )
            current_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[-1], d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, channels, time]
        """

        z = self.cnn(x)
        # z: [batch, hidden_channels, time]

        z = self.global_pool(z)
        # z: [batch, hidden_channels, 1]

        z = z.squeeze(-1)
        # z: [batch, hidden_channels]

        h_raw = self.projection(z)
        # h_raw: [batch, d_model]

        return h_raw


class ClassificationHead(nn.Module):
    """
    Task head для классификации.

    Input:
        h: [batch, d_model]

    Output:
        logits: [batch, num_classes]
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class RawCNNClassifier(nn.Module):
    """
    Experiment 1:
    raw time series → raw encoder → task head
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 128,
        hidden_channels: tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 5,
        encoder_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = RawTimeSeriesEncoder(
            in_channels=in_channels,
            d_model=d_model,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=encoder_dropout,
        )

        self.head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=d_model,
            dropout=head_dropout,
        )

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: [batch, channels, time]
        """

        h_raw = self.encoder(x_raw)
        # h_raw: [batch, d_model]

        logits = self.head(h_raw)
        # logits: [batch, num_classes]

        return logits


CNN_Encoder = RawTimeSeriesEncoder
