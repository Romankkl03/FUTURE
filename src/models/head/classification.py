import torch
import torch.nn as nn


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
