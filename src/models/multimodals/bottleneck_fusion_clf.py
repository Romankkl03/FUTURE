from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from src.models.head.classification import ClassificationHead
from src.models.fusion.bottleneck_fusion import BottleneckRepresentationEncoder
from src.models.registry.encoder_registry import ENCODER_REGISTRY


class FlexibleBottleneckClassifier(nn.Module):
    """
    Flexible multimodal bottleneck classifier.

    Each selected modality is encoded into h_i ∈ R^D.
    Then BottleneckRepresentationEncoder builds a fused representation
    through latent bottleneck tokens.

    Example:
        model = FlexibleBottleneckClassifier(
            modalities=("raw", "stats", "gaf", "stft"),
            num_classes=10,
            d_model=128,
            encoder_kwargs={
                "raw": {"in_channels": 1},
                "stats": {"in_features": 32},
                "gaf": {"in_channels": 1},
                "stft": {"in_channels": 1},
            },
        )

        logits = model({
            "raw": x_raw,
            "stats": x_stats,
            "gaf": x_gaf,
            "stft": x_stft,
        })
    """

    def __init__(
        self,
        modalities: Sequence[str],
        num_classes: int,
        d_model: int = 128,
        encoder_kwargs: Mapping[str, dict] | None = None,
        num_latents: int = 4,
        num_bottleneck_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        pooling: str = "mean",
        bottleneck_dropout: float = 0.1,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        if len(modalities) == 0:
            raise ValueError("modalities must contain at least one modality.")

        if len(set(modalities)) != len(modalities):
            raise ValueError(f"Duplicate modalities are not allowed. Got: {modalities}")

        self.modalities = tuple(modalities)
        self.d_model = d_model

        encoder_kwargs = encoder_kwargs or {}

        self.encoders = nn.ModuleDict()

        for name in self.modalities:
            if name not in ENCODER_REGISTRY:
                available = ", ".join(sorted(ENCODER_REGISTRY.keys()))
                raise ValueError(
                    f"Unknown modality '{name}'. "
                    f"Available modalities: {available}"
                )

            if name not in encoder_kwargs:
                raise ValueError(
                    f"Missing encoder config for modality '{name}'."
                )

            builder = ENCODER_REGISTRY[name]
            self.encoders[name] = builder(d_model, encoder_kwargs[name])

        self.bottleneck = BottleneckRepresentationEncoder(
            n_modalities=len(self.modalities),
            d_model=d_model,
            num_latents=num_latents,
            num_layers=num_bottleneck_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=bottleneck_dropout,
            pooling=pooling,
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
        embeddings: dict[str, torch.Tensor] = {}

        for name in self.modalities:
            if name not in inputs:
                raise ValueError(
                    f"Missing input for modality '{name}'. "
                    f"Expected inputs: {self.modalities}. "
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

        embeddings_list = [
            embeddings_dict[name]
            for name in self.modalities
        ]

        if return_aux:
            aux = self.bottleneck(
                *embeddings_list,
                return_aux=True,
            )

            logits = self.head(aux["h_final"])

            aux["logits"] = logits
            aux["embeddings"] = embeddings_dict
            aux["modality_order"] = self.modalities

            return aux

        h_final = self.bottleneck(
            *embeddings_list,
            return_aux=False,
        )

        return self.head(h_final)