import torch
import torch.nn as nn


class RawCenteredResidualFusion(nn.Module):
    """
    Main idea: make the model invariant to the context embeddings. 
    Where context embeddings are the additional inputs to the model (e.g. GAF, stats, etc.).

    Formulation:
    h_final = h_raw + alpha * delta

    alpha_is_vector: bool = True:
     If True, alpha is a vector of the same dimension as h_raw. 
     It's flexible to learn the importance of each dimension.

    Raw-centered residual gated fusion.

    h_raw: [batch, d_model]
    context embeddings: each [batch, d_model]

    Logic:
        h_ctx = concat(context_embeddings)
        h_ctx = context_projector(h_ctx)

        z = concat(h_raw, h_ctx)

        delta = delta_mlp(z)
        alpha = sigmoid(alpha_mlp(z))

        h_final = h_raw + alpha * delta
    """

    def __init__(
        self,
        n_context_inputs: int,
        d_model: int,
        context_hidden_dim: int = 128,
        delta_hidden_dim: int = 128,
        dropout: float = 0.2,
        alpha_is_vector: bool = True,
    ):
        super().__init__()

        self.n_context_inputs = n_context_inputs
        self.d_model = d_model
        self.alpha_is_vector = alpha_is_vector

        context_in_dim = n_context_inputs * d_model

        self.context_projector = nn.Sequential(
            nn.Linear(context_in_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

        fusion_in_dim = 2 * d_model

        self.delta_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, delta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden_dim, d_model),
        )

        alpha_out_dim = d_model if alpha_is_vector else 1

        self.alpha_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, delta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden_dim, alpha_out_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_raw: torch.Tensor,
        *context_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        if len(context_embeddings) != self.n_context_inputs:
            raise ValueError(
                f"Expected {self.n_context_inputs} context embeddings, "
                f"got {len(context_embeddings)}."
            )

        # [batch, n_context_inputs * d_model]
        h_context_cat = torch.cat(context_embeddings, dim=-1)

        # [batch, d_model]
        h_context = self.context_projector(h_context_cat)

        # [batch, 2 * d_model]
        fusion_input = torch.cat([h_raw, h_context], dim=-1)

        # [batch, d_model]
        delta = self.delta_mlp(fusion_input)

        # alpha:
        #   vector mode: [batch, d_model]
        #   scalar mode: [batch, 1]
        alpha = self.alpha_mlp(fusion_input)

        # [batch, d_model]
        h_final = h_raw + alpha * delta

        if return_aux:
            return {
                "h_final": h_final,
                "h_raw": h_raw,
                "h_context": h_context,
                "delta": delta,
                "alpha": alpha,
            }

        return h_final
