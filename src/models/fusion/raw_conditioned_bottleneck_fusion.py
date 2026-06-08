import torch
import torch.nn as nn

from src.models.fusion.bottleneck_fusion import BottleneckLatentBlock


class RawConditionedBottleneckRepresentationEncoder(nn.Module):
    """
    Raw-conditioned bottleneck representation encoder.

    Difference from BottleneckRepresentationEncoder:
        ordinary:
            latents = learnable_latents

        raw-conditioned:
            raw_condition = raw_to_latents(h_raw)
            latents = learnable_latents + raw_condition

    Cross-attention goes only to context tokens.
    RAW does not appear as a modality token.
    RAW controls the initial state of latent tokens.
    """

    def __init__(
        self,
        n_context_modalities: int,
        d_model: int,
        num_latents: int = 4,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pooling: str = "mean",
        latent_init_std: float = 0.02,
        raw_condition_scale: float = 1.0,
        zero_init_raw_condition: bool = False,
    ):
        super().__init__()

        if n_context_modalities <= 0:
            raise ValueError("n_context_modalities must be > 0.")

        if num_latents <= 0:
            raise ValueError("num_latents must be > 0.")

        if num_layers <= 0:
            raise ValueError("num_layers must be > 0.")

        if pooling not in {"mean", "cls", "concat"}:
            raise ValueError(
                f"Unknown pooling='{pooling}'. "
                "Expected one of: 'mean', 'cls', 'concat'."
            )

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}."
            )

        self.n_context_modalities = n_context_modalities
        self.d_model = d_model
        self.num_latents = num_latents
        self.pooling = pooling
        self.raw_condition_scale = raw_condition_scale

        self.modality_embedding = nn.Parameter(
            torch.zeros(1, n_context_modalities, d_model)
        )

        self.latent_tokens = nn.Parameter(
            torch.randn(num_latents, d_model) * latent_init_std
        )

        self.raw_to_latents = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_latents * d_model),
        )

        if zero_init_raw_condition:
            raw_linear = self.raw_to_latents[-1]
            nn.init.zeros_(raw_linear.weight)
            nn.init.zeros_(raw_linear.bias)

        self.blocks = nn.ModuleList(
            [
                BottleneckLatentBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        if pooling == "concat":
            self.concat_projection = nn.Sequential(
                nn.Linear(num_latents * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.concat_projection = None

    def forward(
        self,
        h_raw: torch.Tensor,
        *context_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        if len(context_embeddings) != self.n_context_modalities:
            raise ValueError(
                f"Expected {self.n_context_modalities} context embeddings, "
                f"got {len(context_embeddings)}."
            )

        context_tokens = torch.stack(context_embeddings, dim=1)

        batch_size, n_context_modalities, d_model = context_tokens.shape

        if n_context_modalities != self.n_context_modalities:
            raise ValueError(
                f"Expected n_context_modalities={self.n_context_modalities}, "
                f"got {n_context_modalities}."
            )

        if d_model != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, got {d_model}."
            )

        if h_raw.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected h_raw last dim={self.d_model}, got {h_raw.shape[-1]}."
            )

        context_tokens = context_tokens + self.modality_embedding

        base_latents = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        raw_condition = self.raw_to_latents(h_raw)
        raw_condition = raw_condition.view(
            batch_size,
            self.num_latents,
            self.d_model,
        )

        latents = base_latents + self.raw_condition_scale * raw_condition

        all_attn = []

        for block in self.blocks:
            if return_aux:
                latents, attn = block(
                    latents,
                    context_tokens,
                    return_attn=True,
                )
                all_attn.append(attn)
            else:
                latents = block(
                    latents,
                    context_tokens,
                    return_attn=False,
                )

        latents = self.final_norm(latents)

        if self.pooling == "mean":
            h_final = latents.mean(dim=1)

        elif self.pooling == "cls":
            h_final = latents[:, 0]

        elif self.pooling == "concat":
            h_flat = latents.reshape(batch_size, self.num_latents * self.d_model)
            h_final = self.concat_projection(h_flat)

        else:
            raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")

        if return_aux:
            return {
                "h_final": h_final,
                "context_tokens": context_tokens,
                "latents": latents,
                "base_latents": base_latents,
                "raw_condition": raw_condition,
                "attn": all_attn,
            }

        return h_final
