import torch
import torch.nn as nn

from .cnn_encoder import ClassificationHead, RawTimeSeriesEncoder
from .gaf_encoder import GAFEncoder
from .stats_encoder import StatisticalEncoder
from .stft_encoder import STFTEncoder


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


class RawCenteredStatsGAFSTFTClassifier(nn.Module):
    """
    Raw-centered classifier.

    raw is the main representation.
    stats / GAF / STFT are optional context modalities.

    No MTF here.
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
        delta_hidden_dim: int = 128,
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.2,
        alpha_is_vector: bool = True,
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

        self.fusion = RawCenteredResidualFusion(
            n_context_inputs=len(context_modalities),
            d_model=d_model,
            context_hidden_dim=context_hidden_dim,
            delta_hidden_dim=delta_hidden_dim,
            dropout=fusion_dropout,
            alpha_is_vector=alpha_is_vector,
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
            aux = self.fusion(
                h_raw,
                *context_embeddings,
                return_aux=True,
            )
            logits = self.head(aux["h_final"])
            aux["logits"] = logits
            return aux

        h_final = self.fusion(
            h_raw,
            *context_embeddings,
            return_aux=False,
        )

        return self.head(h_final)