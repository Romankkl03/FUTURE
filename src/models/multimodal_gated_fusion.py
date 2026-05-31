import torch
import torch.nn as nn

from .cnn_encoder import ClassificationHead, RawTimeSeriesEncoder
from .gaf_encoder import GAFEncoder
from .mtf_encoder import MTFEncoder
from .stats_encoder import StatisticalEncoder
from .stft_encoder import STFTEncoder


class MultiModalGatedFusion(nn.Module):
    """
    Gated fusion for multiple modality embeddings.

    Input:
        h_1, ..., h_n: each [batch, d_model]

    Internally:
        stack: [batch, n_inputs, d_model]
        gate logits: [batch, n_inputs]
        gates = softmax(logits): [batch, n_inputs]
        weighted sum: [batch, d_model]
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

        # Gate network sees all modalities jointly.
        # Flattened input: [batch, n_inputs * d_model]
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

        # [batch, n_inputs, d_model]
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

        # [batch, n_inputs * d_model]
        h_flat = h_stack.reshape(batch_size, n_inputs * d_model)

        # [batch, n_inputs]
        gate_logits = self.gate_net(h_flat)

        # [batch, n_inputs]
        gates = torch.softmax(gate_logits, dim=1)

        # [batch, n_inputs, 1]
        gates_expanded = gates.unsqueeze(-1)

        # [batch, d_model]
        h_fused = torch.sum(gates_expanded * h_stack, dim=1)

        if return_gates:
            return h_fused, gates

        return h_fused


class RawStatsGAFMTFSTFTGatedClassifier(nn.Module):
    """
    raw time series + statistical features + GAF + MTF + STFT representations
    with softmax gated fusion.
    """

    def __init__(
        self,
        raw_in_channels: int,
        stat_in_features: int,
        image_in_channels: int,
        num_classes: int,
        d_model: int = 128,
        raw_hidden_channels: tuple[int, ...] = (64, 128, 128),
        stat_hidden_dims: tuple[int, ...] = (128, 64),
        image_hidden_channels: tuple[int, ...] = (32, 64, 128),
        fusion_hidden_dim: int = 128,
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.raw_encoder = RawTimeSeriesEncoder(
            in_channels=raw_in_channels,
            d_model=d_model,
            hidden_channels=raw_hidden_channels,
            kernel_size=kernel_size,
            dropout=raw_dropout,
        )

        self.stat_encoder = StatisticalEncoder(
            in_features=stat_in_features,
            d_model=d_model,
            hidden_dims=stat_hidden_dims,
            dropout=stat_dropout,
        )

        self.gaf_encoder = GAFEncoder(
            in_channels=image_in_channels,
            d_model=d_model,
            hidden_channels=image_hidden_channels,
            dropout=image_dropout,
        )

        self.mtf_encoder = MTFEncoder(
            in_channels=image_in_channels,
            d_model=d_model,
            hidden_channels=image_hidden_channels,
            dropout=image_dropout,
        )

        self.stft_encoder = STFTEncoder(
            in_channels=image_in_channels,
            d_model=d_model,
            hidden_channels=image_hidden_channels,
            dropout=image_dropout,
        )

        self.fusion = MultiModalGatedFusion(
            n_inputs=5,
            d_model=d_model,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
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
        x_stat: torch.Tensor,
        x_gaf: torch.Tensor,
        x_mtf: torch.Tensor,
        x_stft: torch.Tensor,
        return_gates: bool = False,
    ):
        h_raw = self.raw_encoder(x_raw)
        h_stat = self.stat_encoder(x_stat)
        h_gaf = self.gaf_encoder(x_gaf)
        h_mtf = self.mtf_encoder(x_mtf)
        h_stft = self.stft_encoder(x_stft)

        if return_gates:
            h_fused, gates = self.fusion(
                h_raw,
                h_stat,
                h_gaf,
                h_mtf,
                h_stft,
                return_gates=True,
            )
            logits = self.head(h_fused)
            return logits, gates

        h_fused = self.fusion(
            h_raw,
            h_stat,
            h_gaf,
            h_mtf,
            h_stft,
        )

        return self.head(h_fused)


class RawGatedStatsGAFSTFTClassifier(nn.Module):
    """Raw-centered gated fusion with optional stats/GAF/STFT context modalities."""

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
        fusion_hidden_dim: int = 128,
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        head_dropout: float = 0.2,
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

        self.modalities = ("raw", *context_modalities)
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

        self.fusion = MultiModalGatedFusion(
            n_inputs=len(self.modalities),
            d_model=d_model,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
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
        embeddings = [self.raw_encoder(x_raw)]

        for modality in self.context_modalities:
            if modality == "stats":
                if x_stat is None:
                    raise ValueError("x_stat is required because 'stats' is enabled.")
                embeddings.append(self.stat_encoder(x_stat))
            elif modality == "gaf":
                if x_gaf is None:
                    raise ValueError("x_gaf is required because 'gaf' is enabled.")
                embeddings.append(self.gaf_encoder(x_gaf))
            elif modality == "stft":
                if x_stft is None:
                    raise ValueError("x_stft is required because 'stft' is enabled.")
                embeddings.append(self.stft_encoder(x_stft))
            else:
                raise RuntimeError(f"Unexpected modality: {modality}")

        if return_aux:
            h_fused, gates = self.fusion(*embeddings, return_gates=True)
            logits = self.head(h_fused)
            return {
                "h_final": h_fused,
                "gates": gates,
                "logits": logits,
            }

        h_fused = self.fusion(*embeddings)
        return self.head(h_fused)