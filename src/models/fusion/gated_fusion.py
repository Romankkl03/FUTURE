import torch
import torch.nn as nn


class MultiModalGatedFusion(nn.Module):
    """
    Gated fusion for multiple modality embeddings.

    Input:
        h_1, ..., h_n: each [batch, d_model]

    Internally:
        stack: [batch, n_inputs, d_model]
        gate logits: [batch, n_inputs]
        gates = softmax(logits): [batch, n_inputs]
        weighted sum: [batch, d_model]
    """

    def __init__(
        self,
        n_inputs: int,
        d_model: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.d_model = d_model

        # Gate network sees all modalities jointly.
        # Flattened input: [batch, n_inputs * d_model]
        self.gate_net = nn.Sequential(
            nn.Linear(n_inputs * d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_inputs),
        )

    def forward(
        self,
        *embeddings: torch.Tensor,
        return_gates: bool = False,
    ):
        if len(embeddings) != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} embeddings, got {len(embeddings)}."
            )

        # [batch, n_inputs, d_model]
        h_stack = torch.stack(embeddings, dim=1)

        batch_size, n_inputs, d_model = h_stack.shape

        if n_inputs != self.n_inputs:
            raise ValueError(
                f"Expected n_inputs={self.n_inputs}, got {n_inputs}."
            )

        if d_model != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, got {d_model}."
            )

        # [batch, n_inputs * d_model]
        h_flat = h_stack.reshape(batch_size, n_inputs * d_model)

        # [batch, n_inputs]
        gate_logits = self.gate_net(h_flat)

        # [batch, n_inputs]
        gates = torch.softmax(gate_logits, dim=1)

        # [batch, n_inputs, 1]
        gates_expanded = gates.unsqueeze(-1)

        # [batch, d_model]
        h_fused = torch.sum(gates_expanded * h_stack, dim=1)

        if return_gates:
            return h_fused, gates

        return h_fused
