"""Softmax-gated fusion of modality embeddings."""

import torch
import torch.nn as nn


class MultiModalGatedFusion(nn.Module):
    """Fuse modality embeddings with learned softmax gates.

    Stacks embeddings into ``(batch, n_inputs, d_model)``, predicts gate
    logits from the flattened stack, and returns their weighted sum.

    When ``gamma=0`` at initialization the fusion is near uniform; during
    training the model learns to up-weight informative modalities.

    Forward
    -------
    *embeddings : Tensor
        ``n_inputs`` tensors, each of shape ``(batch, d_model)``.
    return_gates : bool
        If ``True``, also return gate weights of shape ``(batch, n_inputs)``.
    Returns fused embedding of shape ``(batch, d_model)``.
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

        h_flat = h_stack.reshape(batch_size, n_inputs * d_model)
        gate_logits = self.gate_net(h_flat)
        gates = torch.softmax(gate_logits, dim=1)
        gates_expanded = gates.unsqueeze(-1)
        h_fused = torch.sum(gates_expanded * h_stack, dim=1)

        if return_gates:
            return h_fused, gates

        return h_fused
