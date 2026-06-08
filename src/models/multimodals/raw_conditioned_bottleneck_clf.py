from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from src.models.head.classification import ClassificationHead
from src.models.registry.encoder_registry import ENCODER_REGISTRY
from src.models.fusion.raw_conditioned_bottleneck_fusion import (
    RawConditionedBottleneckRepresentationEncoder,
)


class FlexibleRawConditionedContextBottleneckClassifier(nn.Module):
    """
    Flexible raw-conditioned context bottleneck classifier.

    Bottleneck sees CONTEXT only.
    RAW controls latent queries.

    Architecture:
        h_raw = encoder_raw(x_raw)
        h_context_i = encoder_i(x_i)

        h_context = RawConditionedBottleneck(
            h_raw,
            h_context_1,
            h_context_2,
            ...
        )

        delta = delta_proj(h_context)
        alpha = sigmoid(alpha_mlp(concat(h_raw, h_context)))

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
        raw_condition_scale: float = 1.0,
        zero_init_raw_condition: bool = False,
    ):
        super().__init__()

        if len(context_modalities) == 0:
            raise ValueError("context_modalities must contain at least one modality.")

        self.raw_modality = raw_modality
        self.context_modalities = tuple(context_modalities)
        self.modality_order = self.context_modalities
        self.d_model = d_model

        all_modalities = (self.raw_modality, *self.context_modalities)

        if len(set(all_modalities)) != len(all_modalities):
            raise ValueError(
                f"Duplicate modalities are not allowed. Got: {all_modalities}"
            )

        encoder_kwargs = encoder_kwargs or {}

        self.encoders = nn.ModuleDict()

        for name in all_modalities:
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

        self.bottleneck = RawConditionedBottleneckRepresentationEncoder(
            n_context_modalities=len(self.context_modalities),
            d_model=d_model,
            num_latents=num_latents,
            num_layers=num_bottleneck_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=bottleneck_dropout,
            pooling=pooling,
            raw_condition_scale=raw_condition_scale,
            zero_init_raw_condition=zero_init_raw_condition,
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
        required_modalities = (self.raw_modality, *self.context_modalities)

        embeddings = {}

        for name in required_modalities:
            if name not in inputs:
                raise ValueError(
                    f"Missing input for modality '{name}'. "
                    f"Expected inputs: {required_modalities}. "
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

        context_embeddings = [
            embeddings_dict[name]
            for name in self.context_modalities
        ]

        if return_aux:
            bottleneck_aux = self.bottleneck(
                h_raw,
                *context_embeddings,
                return_aux=True,
            )
            h_context = bottleneck_aux["h_final"]
        else:
            h_context = self.bottleneck(
                h_raw,
                *context_embeddings,
                return_aux=False,
            )

        delta = self.delta_proj(h_context)
        alpha = self.alpha(torch.cat([h_raw, h_context], dim=-1))

        h_final = h_raw + alpha * delta
        logits = self.head(h_final)

        if return_aux:
            return {
                "logits": logits,
                "h_final": h_final,
                "h_raw": h_raw,
                "h_context": h_context,
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
