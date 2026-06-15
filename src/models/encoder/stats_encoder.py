"""Encoders for hand-crafted statistical feature vectors."""

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """MLP block: Linear → LayerNorm → GELU → Dropout."""

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
    """Encode tabular statistical features into a fixed-size embedding.

    Architecture: input LayerNorm → stacked :class:`MLPBlock` layers →
    linear projection to ``d_model``.

    Forward
    -------
    x_stat : Tensor, shape ``(batch, num_features)``
        Per-sample statistical descriptors.
    Returns ``h_stat`` of shape ``(batch, d_model)``.
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
        return self.encoder(x_stat)
