from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from src.models.head.classification import ClassificationHead
from src.models.registry.encoder_registry import ENCODER_REGISTRY
from src.models.fusion.bottleneck_fusion import BottleneckRepresentationEncoder


class FlexibleRawResidualBottleneckClassifier(nn.Module):
    """
    Flexible raw-residual bottleneck classifier.

    Bottleneck sees RAW + context tokens.

    Architecture:
        h_raw = encoder_raw(x_raw)
        h_context_i = encoder_i(x_i)

        h_bottleneck = Bottleneck(
            h_raw,
            h_context_1,
            ...
        )

        delta = delta_proj(h_bottleneck)
        alpha = sigmoid(alpha_mlp(concat(h_raw, h_bottleneck)))

        h_final = h_raw + alpha * delta
        logits = head(h_final)
    """

    def __init__(
        self,
        raw_modality: str = "raw",
        context_modalities: Sequence[str] = ("stats", "gaf", "stft"),
        num_classes: int = 2,
        d_model: int = 128,
        encoder_kwargs: Mapping[str, dict] | None = None,
        bottleneck_dropout: float = 0.1,
        residual_dropout: float = 0.2,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.2,
        num_latents: int = 4,
        num_bottleneck_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        pooling: str = "mean",
    ):
        super().__init__()

        if len(context_modalities) == 0:
            raise ValueError("context_modalities must contain at least one modality.")

        self.raw_modality = raw_modality
        self.context_modalities = tuple(context_modalities)
        self.modality_order = (self.raw_modality, *self.context_modalities)
        self.d_model = d_model

        if len(set(self.modality_order)) != len(self.modality_order):
            raise ValueError(
                f"Duplicate modalities are not allowed. Got: {self.modality_order}"
            )

        encoder_kwargs = encoder_kwargs or {}

        self.encoders = nn.ModuleDict()

        for name in self.modality_order:
            if name not in ENCODER_REGISTRY:
                available = ", ".join(sorted(ENCODER_REGISTRY.keys()))
                raise ValueError(
                    f"Unknown modality '{name}'. "
                    f"Available modalities: {available}"
                )

            if name not in encoder_kwargs:
                raise ValueError(f"Missing encoder config for modality '{name}'.")

            self.encoders[name] = ENCODER_REGISTRY[name](
                d_model,
                encoder_kwargs[name],
            )

        self.bottleneck = BottleneckRepresentationEncoder(
            n_modalities=len(self.modality_order),
            d_model=d_model,
            num_latents=num_latents,
            num_layers=num_bottleneck_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=bottleneck_dropout,
            pooling=pooling,
        )

        self.delta_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(residual_dropout),
            nn.Linear(d_model, d_model),
        )

        self.alpha = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(residual_dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim or d_model,
            dropout=head_dropout,
        )

    def encode(
        self,
        inputs: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        embeddings = {}

        for name in self.modality_order:
            if name not in inputs:
                raise ValueError(
                    f"Missing input for modality '{name}'. "
                    f"Expected inputs: {self.modality_order}. "
                    f"Got: {tuple(inputs.keys())}"
                )

            embeddings[name] = self.encoders[name](inputs[name])

        return embeddings

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        return_aux: bool = False,
    ):
        embeddings_dict = self.encode(inputs)

        h_raw = embeddings_dict[self.raw_modality]

        embeddings_list = [
            embeddings_dict[name]
            for name in self.modality_order
        ]

        if return_aux:
            bottleneck_aux = self.bottleneck(
                *embeddings_list,
                return_aux=True,
            )
            h_bottleneck = bottleneck_aux["h_final"]
        else:
            h_bottleneck = self.bottleneck(
                *embeddings_list,
                return_aux=False,
            )

        delta = self.delta_proj(h_bottleneck)
        alpha = self.alpha(torch.cat([h_raw, h_bottleneck], dim=-1))

        h_final = h_raw + alpha * delta
        logits = self.head(h_final)

        if return_aux:
            return {
                "logits": logits,
                "h_final": h_final,
                "h_raw": h_raw,
                "h_bottleneck": h_bottleneck,
                "delta": delta,
                "alpha": alpha,
                "embeddings": embeddings_dict,
                "modality_order": self.modality_order,
                **{
                    f"bottleneck_{k}": v
                    for k, v in bottleneck_aux.items()
                    if k != "h_final"
                },
            }

        return logits
