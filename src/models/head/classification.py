"""Classification head for fused or single-modality embeddings."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Two-layer MLP head that maps embeddings to class logits.

    Architecture: ``Linear → LayerNorm → GELU → Dropout → Linear``.

    Parameters
    ----------
    d_model:
        Input embedding size.
    num_classes:
        Number of output classes.
    hidden_dim:
        Width of the hidden layer (default 128).
    dropout:
        Dropout rate applied after the activation.

    Forward
    -------
    h : Tensor, shape ``(batch, d_model)``
        Fused or single-modality representation.
    Returns logits of shape ``(batch, num_classes)``.
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
