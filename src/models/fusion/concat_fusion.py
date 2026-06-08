import torch
import torch.nn as nn


class MultiConcatFusionMLP(nn.Module):
    """
    concat(h_1, ..., h_n) -> h_final
    """

    def __init__(
        self,
        n_inputs: int,
        d_model: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        if n_inputs < 1:
            raise ValueError(f"n_inputs must be >= 1, got {n_inputs}")

        self.n_inputs = n_inputs
        self.d_model = d_model

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
        if len(embeddings) != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} embeddings, got {len(embeddings)}"
            )

        h = torch.cat(list(embeddings), dim=1)
        return self.fusion(h)
