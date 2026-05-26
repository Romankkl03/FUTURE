import torch
import torch.nn as nn


class MultiConcatFusionMLP(nn.Module):
    """concat(h_1, ..., h_n) -> h_final for equal-size modality embeddings."""

    def __init__(
        self,
        n_inputs: int,
        d_model: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(n_inputs * d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat(embeddings, dim=1))
