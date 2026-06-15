"""Configurable multimodal classifiers built from encoders + fusion modules."""

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from src.models.fusion.concat_fusion import MultiConcatFusionMLP
from src.models.head.classification import ClassificationHead
from src.models.registry.encoder_registry import ENCODER_REGISTRY


class FlexibleConcatClassifier(nn.Module):
    """Multimodal classifier with concat fusion.

    Each modality is encoded to ``d_model``, concatenated, and passed through
    an MLP. All modalities are treated symmetrically.

    Example::

        model = FlexibleConcatClassifier(
            modalities=("raw", "stats", "gaf"),
            num_classes=4,
            d_model=128,
            encoder_kwargs={
                "raw": {"in_channels": 1},
                "stats": {"in_features": 32},
                "gaf": {"in_channels": 1},
            },
        )
        logits = model({"raw": x_raw, "stats": x_stats, "gaf": x_gaf})
    """

    def __init__(
        self,
        modalities: Sequence[str],
        num_classes: int,
        d_model: int = 128,
        encoder_kwargs: Mapping[str, dict] | None = None,
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.2,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        if len(modalities) == 0:
            raise ValueError("modalities must contain at least one modality")

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
                    f"Missing encoder config for modality '{name}'"
                )

            builder = ENCODER_REGISTRY[name]
            self.encoders[name] = builder(d_model, encoder_kwargs[name])

        self.fusion = MultiConcatFusionMLP(
            n_inputs=len(self.modalities),
            d_model=d_model,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )

        self.head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim or d_model,
            dropout=head_dropout,
        )

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
    ):
        embeddings: list[torch.Tensor] = []

        for name in self.modalities:
            if name not in inputs:
                raise ValueError(
                    f"Missing input for modality '{name}'. "
                    f"Expected inputs: {self.modalities}. "
                    f"Got: {tuple(inputs.keys())}"
                )

            h = self.encoders[name](inputs[name])
            embeddings.append(h)

        h_final = self.fusion(*embeddings)
        return self.head(h_final)
