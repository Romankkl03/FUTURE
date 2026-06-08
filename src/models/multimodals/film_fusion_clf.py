from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from src.models.head.classification import ClassificationHead
from src.models.fusion.film_fusion import FiLMFusion
from src.models.registry.encoder_registry import ENCODER_REGISTRY


class FlexibleFiLMClassifier(nn.Module):
    """
    Flexible raw-centered FiLM classifier.

    Raw is the main representation.
    Context modalities generate FiLM parameters gamma and beta.

    Example:
        model = FlexibleFiLMClassifier(
            raw_modality="raw",
            context_modalities=("stats", "gaf", "stft"),
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
        raw_modality: str = "raw",
        context_modalities: Sequence[str] = ("stats", "gaf", "stft"),
        num_classes: int = 2,
        d_model: int = 128,
        encoder_kwargs: Mapping[str, dict] | None = None,
        context_hidden_dim: int = 128,
        film_hidden_dim: int = 128,
        film_dropout: float = 0.2,
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.2,
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if len(context_modalities) == 0:
            raise ValueError("context_modalities must contain at least one modality.")

        self.raw_modality = raw_modality
        self.context_modalities = tuple(context_modalities)
        self.d_model = d_model

        encoder_kwargs = encoder_kwargs or {}

        all_modalities = (self.raw_modality,) + self.context_modalities

        # Check duplicates
        if len(set(all_modalities)) != len(all_modalities):
            raise ValueError(
                f"Duplicate modalities are not allowed. Got: {all_modalities}"
            )

        self.encoders = nn.ModuleDict()

        for name in all_modalities:
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

        self.film = FiLMFusion(
            n_context_inputs=len(self.context_modalities),
            d_model=d_model,
            context_hidden_dim=context_hidden_dim,
            film_hidden_dim=film_hidden_dim,
            dropout=film_dropout,
            gamma_scale=gamma_scale,
            beta_scale=beta_scale,
            use_layernorm=use_layernorm,
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
        return_aux: bool = False,
    ):
        if self.raw_modality not in inputs:
            raise ValueError(
                f"Missing input for raw modality '{self.raw_modality}'. "
                f"Got: {tuple(inputs.keys())}"
            )

        h_raw = self.encoders[self.raw_modality](inputs[self.raw_modality])

        context_embeddings: list[torch.Tensor] = []

        for name in self.context_modalities:
            if name not in inputs:
                raise ValueError(
                    f"Missing input for context modality '{name}'. "
                    f"Expected context modalities: {self.context_modalities}. "
                    f"Got: {tuple(inputs.keys())}"
                )

            h_context = self.encoders[name](inputs[name])
            context_embeddings.append(h_context)

        if return_aux:
            aux = self.film(
                h_raw,
                *context_embeddings,
                return_aux=True,
            )
            aux["logits"] = self.head(aux["h_final"])
            return aux

        h_final = self.film(
            h_raw,
            *context_embeddings,
            return_aux=False,
        )

        return self.head(h_final)
