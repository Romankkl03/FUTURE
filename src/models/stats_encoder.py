import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Базовый MLP блок:
    Linear -> LayerNorm -> GELU -> Dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StatisticalEncoder(nn.Module):
    """
    Encoder для статистических признаков.

    Input:
        x_stat: [batch, num_features]

    Output:
        h_stat: [batch, d_model]
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = [nn.LayerNorm(in_features)]
        current_features = in_features

        for hidden_dim in hidden_dims:
            layers.append(
                MLPBlock(
                    in_features=current_features,
                    out_features=hidden_dim,
                    dropout=dropout,
                )
            )
            current_features = hidden_dim

        layers.extend(
            [
                nn.Linear(current_features, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        self.encoder = nn.Sequential(*layers)

    def forward(self, x_stat: torch.Tensor) -> torch.Tensor:
        """
        x_stat: [batch, num_features]
        """
        return self.encoder(x_stat)


class StatisticalMLPClassifier(nn.Module):
    """
    Классификатор для статистических признаков.

    Input:
        x_stat: [batch, num_features]

    Output:
        logits: [batch, num_classes]
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = [nn.LayerNorm(in_features)]
        current_features = in_features

        for hidden_dim in hidden_dims:
            layers.append(
                MLPBlock(
                    in_features=current_features,
                    out_features=hidden_dim,
                    dropout=dropout,
                )
            )
            current_features = hidden_dim

        layers.append(nn.Linear(current_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x_stat: torch.Tensor) -> torch.Tensor:
        """
        x_stat: [batch, num_features]
        """
        return self.net(x_stat)
