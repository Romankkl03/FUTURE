"""Feature-wise Linear Modulation (FiLM) fusion with raw as the main stream."""

import torch
import torch.nn as nn

from src.models.encoder.gaf_encoder import GAFEncoder
from src.models.encoder.raw_encoder import RawTimeSeriesEncoder
from src.models.encoder.stats_encoder import StatisticalEncoder
from src.models.encoder.stft_encoder import STFTEncoder
from src.models.head.classification import ClassificationHead


class FiLMFusion(nn.Module):
    """Modulate a raw embedding with context-derived scale and shift.

    Context embeddings are concatenated, projected, and mapped to FiLM
    parameters ``gamma`` and ``beta``. The fused representation is:

    ``h_final = h_raw * (1 + gamma) + beta``

    With ``gamma=0`` and ``beta=0`` the output equals ``h_raw``, giving a
    stable raw-centered baseline at initialization.

    Forward
    -------
    h_raw : Tensor, shape ``(batch, d_model)``
        Main (raw) embedding to be modulated.
    *context_embeddings : Tensor
        ``n_context_inputs`` tensors, each of shape ``(batch, d_model)``.
    return_aux : bool
        If ``True``, return a dict with ``h_final``, ``gamma``, ``beta``, etc.
    """

    def __init__(
        self,
        n_context_inputs: int,
        d_model: int,
        context_hidden_dim: int = 128,
        film_hidden_dim: int = 128,
        dropout: float = 0.2,
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if n_context_inputs <= 0:
            raise ValueError("n_context_inputs must be > 0.")

        self.n_context_inputs = n_context_inputs
        self.d_model = d_model
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale

        context_in_dim = n_context_inputs * d_model

        context_layers: list[nn.Module] = [
            nn.Linear(context_in_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden_dim, d_model),
        ]

        if use_layernorm:
            context_layers.append(nn.LayerNorm(d_model))

        self.context_projector = nn.Sequential(*context_layers)

        self.gamma_mlp = nn.Sequential(
            nn.Linear(d_model, film_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, d_model),
            nn.Tanh(),
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(d_model, film_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(film_hidden_dim, d_model),
            nn.Tanh(),
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

        h_context_cat = torch.cat(context_embeddings, dim=-1)
        h_context = self.context_projector(h_context_cat)

        gamma = self.gamma_scale * self.gamma_mlp(h_context)
        beta = self.beta_scale * self.beta_mlp(h_context)
        h_final = h_raw * (1.0 + gamma) + beta

        if return_aux:
            return {
                "h_final": h_final,
                "h_raw": h_raw,
                "h_context": h_context,
                "gamma": gamma,
                "beta": beta,
            }

        return h_final


class RawFiLMStatsGAFSTFTClassifier(nn.Module):
    """End-to-end raw-centered FiLM classifier with fixed modality encoders.

    Encodes raw time series as the main stream and uses stats / GAF / STFT
    (configurable) to produce FiLM conditioning. Prefer
    :class:`~src.models.multimodals.film_fusion_clf.FlexibleFiLMClassifier`
    when modality sets should be chosen at runtime.

    Forward
    -------
    x_raw, x_stat, x_gaf, x_stft : Tensor | None
        Modality inputs; only tensors required by ``context_modalities``
        must be provided.
    """

    SUPPORTED_CONTEXT_MODALITIES = {"stats", "gaf", "stft"}

    def __init__(
        self,
        raw_in_channels: int,
        stat_in_features: int,
        image_in_channels: int,
        num_classes: int,
        context_modalities: tuple[str, ...] = ("stats", "gaf", "stft"),
        d_model: int = 128,
        raw_hidden_channels: tuple[int, ...] = (64, 128, 128),
        stat_hidden_dims: tuple[int, ...] = (128, 64),
        image_hidden_channels: tuple[int, ...] = (32, 64, 128),
        context_hidden_dim: int = 128,
        film_hidden_dim: int = 128,
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        film_dropout: float = 0.2,
        head_dropout: float = 0.2,
        gamma_scale: float = 0.5,
        beta_scale: float = 0.5,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if len(context_modalities) == 0:
            raise ValueError("context_modalities must contain at least one modality.")

        unknown = set(context_modalities) - self.SUPPORTED_CONTEXT_MODALITIES
        if unknown:
            raise ValueError(
                f"Unknown context modalities: {unknown}. "
                f"Supported: {self.SUPPORTED_CONTEXT_MODALITIES}"
            )

        self.context_modalities = context_modalities

        self.raw_encoder = RawTimeSeriesEncoder(
            in_channels=raw_in_channels,
            d_model=d_model,
            hidden_channels=raw_hidden_channels,
            kernel_size=kernel_size,
            dropout=raw_dropout,
        )

        if "stats" in context_modalities:
            self.stat_encoder = StatisticalEncoder(
                in_features=stat_in_features,
                d_model=d_model,
                hidden_dims=stat_hidden_dims,
                dropout=stat_dropout,
            )
        else:
            self.stat_encoder = None

        if "gaf" in context_modalities:
            self.gaf_encoder = GAFEncoder(
                in_channels=image_in_channels,
                d_model=d_model,
                hidden_channels=image_hidden_channels,
                dropout=image_dropout,
            )
        else:
            self.gaf_encoder = None

        if "stft" in context_modalities:
            self.stft_encoder = STFTEncoder(
                in_channels=image_in_channels,
                d_model=d_model,
                hidden_channels=image_hidden_channels,
                dropout=image_dropout,
            )
        else:
            self.stft_encoder = None

        self.film = FiLMFusion(
            n_context_inputs=len(context_modalities),
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
            hidden_dim=d_model,
            dropout=head_dropout,
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        x_stat: torch.Tensor | None = None,
        x_gaf: torch.Tensor | None = None,
        x_stft: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        h_raw = self.raw_encoder(x_raw)

        context_embeddings = []

        for modality in self.context_modalities:
            if modality == "stats":
                if x_stat is None:
                    raise ValueError("x_stat is required because 'stats' is enabled.")
                context_embeddings.append(self.stat_encoder(x_stat))

            elif modality == "gaf":
                if x_gaf is None:
                    raise ValueError("x_gaf is required because 'gaf' is enabled.")
                context_embeddings.append(self.gaf_encoder(x_gaf))

            elif modality == "stft":
                if x_stft is None:
                    raise ValueError("x_stft is required because 'stft' is enabled.")
                context_embeddings.append(self.stft_encoder(x_stft))

            else:
                raise RuntimeError(f"Unexpected modality: {modality}")

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
