import torch
import torch.nn as nn


from .cnn_encoder import ClassificationHead, RawTimeSeriesEncoder
from .gaf_encoder import GAFEncoder
from .multimodal_bottleneck_fusion import (
    BottleneckLatentBlock,
    BottleneckRepresentationEncoder,
)
from .stats_encoder import StatisticalEncoder
from .stft_encoder import STFTEncoder


class RawResidualBottleneckClassifier(nn.Module):
    """
    Bottleneck over [RAW, CONTEXT], then raw-centered residual correction.

    Architecture:
        h_raw = raw_encoder(x_raw)

        h_bottleneck = bottleneck(
            h_raw,
            h_stat,
            h_gaf,
            h_stft,
        )

        delta = delta_proj(h_bottleneck)
        alpha = sigmoid(alpha_mlp(concat(h_raw, h_bottleneck)))

        h_final = h_raw + alpha * delta
        logits = head(h_final)

    This version allows bottleneck latents to attend to RAW and context tokens,
    but final representation is still raw-centered.
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
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        bottleneck_dropout: float = 0.1,
        residual_dropout: float = 0.2,
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

        unknown = set(context_modalities) - self.SUPPORTED_CONTEXT_MODALITIES
        if unknown:
            raise ValueError(
                f"Unknown context modalities: {unknown}. "
                f"Supported: {self.SUPPORTED_CONTEXT_MODALITIES}"
            )

        self.context_modalities = context_modalities
        self.modality_order = ("raw", *context_modalities)

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
            hidden_dim=d_model,
            dropout=head_dropout,
        )

    def _encode_context(
        self,
        x_stat: torch.Tensor | None,
        x_gaf: torch.Tensor | None,
        x_stft: torch.Tensor | None,
    ):
        context_embeddings = []
        aux_embeddings = {}

        for modality in self.context_modalities:
            if modality == "stats":
                if x_stat is None:
                    raise ValueError("x_stat is required because 'stats' is enabled.")
                h_stat = self.stat_encoder(x_stat)
                context_embeddings.append(h_stat)
                aux_embeddings["h_stat"] = h_stat

            elif modality == "gaf":
                if x_gaf is None:
                    raise ValueError("x_gaf is required because 'gaf' is enabled.")
                h_gaf = self.gaf_encoder(x_gaf)
                context_embeddings.append(h_gaf)
                aux_embeddings["h_gaf"] = h_gaf

            elif modality == "stft":
                if x_stft is None:
                    raise ValueError("x_stft is required because 'stft' is enabled.")
                h_stft = self.stft_encoder(x_stft)
                context_embeddings.append(h_stft)
                aux_embeddings["h_stft"] = h_stft

            else:
                raise RuntimeError(f"Unexpected modality: {modality}")

        return context_embeddings, aux_embeddings

    def forward(
        self,
        x_raw: torch.Tensor,
        x_stat: torch.Tensor | None = None,
        x_gaf: torch.Tensor | None = None,
        x_stft: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        h_raw = self.raw_encoder(x_raw)

        context_embeddings, aux_embeddings = self._encode_context(
            x_stat=x_stat,
            x_gaf=x_gaf,
            x_stft=x_stft,
        )

        embeddings = [h_raw, *context_embeddings]

        if return_aux:
            bottleneck_aux = self.bottleneck(
                *embeddings,
                return_aux=True,
            )
            h_bottleneck = bottleneck_aux["h_final"]
        else:
            h_bottleneck = self.bottleneck(
                *embeddings,
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
                "modality_order": self.modality_order,
                **aux_embeddings,
                **{
                    f"bottleneck_{k}": v
                    for k, v in bottleneck_aux.items()
                    if k != "h_final"
                },
            }

        return logits


class ContextOnlyResidualBottleneckClassifier(nn.Module):
    """
    Bottleneck over CONTEXT only, then residual correction to RAW.

    Architecture:
        h_raw = raw_encoder(x_raw)

        h_context = bottleneck(
            h_stat,
            h_gaf,
            h_stft,
        )

        delta = delta_proj(h_context)
        alpha = sigmoid(alpha_mlp(concat(h_raw, h_context)))

        h_final = h_raw + alpha * delta
        logits = head(h_final)

    Difference from RawResidualBottleneckClassifier:
        RAW is not a modality token inside the bottleneck.
        RAW stays as a separate main highway.
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
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        bottleneck_dropout: float = 0.1,
        residual_dropout: float = 0.2,
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

        unknown = set(context_modalities) - self.SUPPORTED_CONTEXT_MODALITIES
        if unknown:
            raise ValueError(
                f"Unknown context modalities: {unknown}. "
                f"Supported: {self.SUPPORTED_CONTEXT_MODALITIES}"
            )

        self.context_modalities = context_modalities
        self.modality_order = context_modalities

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

        self.bottleneck = BottleneckRepresentationEncoder(
            n_modalities=len(context_modalities),
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
            hidden_dim=d_model,
            dropout=head_dropout,
        )

    def _encode_context(
        self,
        x_stat: torch.Tensor | None,
        x_gaf: torch.Tensor | None,
        x_stft: torch.Tensor | None,
    ):
        context_embeddings = []
        aux_embeddings = {}

        for modality in self.context_modalities:
            if modality == "stats":
                if x_stat is None:
                    raise ValueError("x_stat is required because 'stats' is enabled.")
                h_stat = self.stat_encoder(x_stat)
                context_embeddings.append(h_stat)
                aux_embeddings["h_stat"] = h_stat

            elif modality == "gaf":
                if x_gaf is None:
                    raise ValueError("x_gaf is required because 'gaf' is enabled.")
                h_gaf = self.gaf_encoder(x_gaf)
                context_embeddings.append(h_gaf)
                aux_embeddings["h_gaf"] = h_gaf

            elif modality == "stft":
                if x_stft is None:
                    raise ValueError("x_stft is required because 'stft' is enabled.")
                h_stft = self.stft_encoder(x_stft)
                context_embeddings.append(h_stft)
                aux_embeddings["h_stft"] = h_stft

            else:
                raise RuntimeError(f"Unexpected modality: {modality}")

        return context_embeddings, aux_embeddings

    def forward(
        self,
        x_raw: torch.Tensor,
        x_stat: torch.Tensor | None = None,
        x_gaf: torch.Tensor | None = None,
        x_stft: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        h_raw = self.raw_encoder(x_raw)

        context_embeddings, aux_embeddings = self._encode_context(
            x_stat=x_stat,
            x_gaf=x_gaf,
            x_stft=x_stft,
        )

        if return_aux:
            bottleneck_aux = self.bottleneck(
                *context_embeddings,
                return_aux=True,
            )
            h_context = bottleneck_aux["h_final"]
        else:
            h_context = self.bottleneck(
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
                "modality_order": self.modality_order,
                **aux_embeddings,
                **{
                    f"bottleneck_{k}": v
                    for k, v in bottleneck_aux.items()
                    if k != "h_final"
                },
            }

        return logits


class RawConditionedBottleneckRepresentationEncoder(nn.Module):
    """
    Raw-conditioned bottleneck representation encoder.

    Difference from BottleneckRepresentationEncoder:
        ordinary:
            latents = learnable_latents

        raw-conditioned:
            raw_condition = raw_to_latents(h_raw)
            latents = learnable_latents + raw_condition

    Cross-attention still goes only to context tokens:
        [STATS, GAF, STFT]

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

        # [B, M, D]
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

        # Add context modality identity embeddings.
        # [B, M, D]
        context_tokens = context_tokens + self.modality_embedding

        # Base learnable latent tokens.
        # [B, K, D]
        base_latents = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Raw-conditioned latent correction.
        # [B, K * D] -> [B, K, D]
        raw_condition = self.raw_to_latents(h_raw)
        raw_condition = raw_condition.view(
            batch_size,
            self.num_latents,
            self.d_model,
        )

        # [B, K, D]
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

        # [B, K, D]
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


class RawConditionedContextBottleneckClassifier(nn.Module):
    """
    Raw-conditioned context bottleneck classifier.

    Architecture:
        h_raw = raw_encoder(x_raw)

        context_tokens:
            h_stat
            h_gaf
            h_stft

        raw_condition = raw_to_latents(h_raw)
        latents = learnable_latents + raw_condition

        h_context = bottleneck(
            h_raw as latent conditioner,
            context embeddings as cross-attention K,V
        )

        delta = delta_proj(h_context)
        alpha = sigmoid(alpha_mlp(concat(h_raw, h_context)))

        h_final = h_raw + alpha * delta
        logits = head(h_final)

    RAW has privileged role:
        it does not compete as one token among other modality tokens;
        it directly shapes latent queries.
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
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        bottleneck_dropout: float = 0.1,
        residual_dropout: float = 0.2,
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

        unknown = set(context_modalities) - self.SUPPORTED_CONTEXT_MODALITIES
        if unknown:
            raise ValueError(
                f"Unknown context modalities: {unknown}. "
                f"Supported: {self.SUPPORTED_CONTEXT_MODALITIES}"
            )

        self.context_modalities = context_modalities
        self.modality_order = context_modalities

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

        self.bottleneck = RawConditionedBottleneckRepresentationEncoder(
            n_context_modalities=len(context_modalities),
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
            hidden_dim=d_model,
            dropout=head_dropout,
        )

    def _encode_context(
        self,
        x_stat: torch.Tensor | None,
        x_gaf: torch.Tensor | None,
        x_stft: torch.Tensor | None,
    ):
        context_embeddings = []
        aux_embeddings = {}

        for modality in self.context_modalities:
            if modality == "stats":
                if x_stat is None:
                    raise ValueError("x_stat is required because 'stats' is enabled.")
                h_stat = self.stat_encoder(x_stat)
                context_embeddings.append(h_stat)
                aux_embeddings["h_stat"] = h_stat

            elif modality == "gaf":
                if x_gaf is None:
                    raise ValueError("x_gaf is required because 'gaf' is enabled.")
                h_gaf = self.gaf_encoder(x_gaf)
                context_embeddings.append(h_gaf)
                aux_embeddings["h_gaf"] = h_gaf

            elif modality == "stft":
                if x_stft is None:
                    raise ValueError("x_stft is required because 'stft' is enabled.")
                h_stft = self.stft_encoder(x_stft)
                context_embeddings.append(h_stft)
                aux_embeddings["h_stft"] = h_stft

            else:
                raise RuntimeError(f"Unexpected modality: {modality}")

        return context_embeddings, aux_embeddings

    def forward(
        self,
        x_raw: torch.Tensor,
        x_stat: torch.Tensor | None = None,
        x_gaf: torch.Tensor | None = None,
        x_stft: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        h_raw = self.raw_encoder(x_raw)

        context_embeddings, aux_embeddings = self._encode_context(
            x_stat=x_stat,
            x_gaf=x_gaf,
            x_stft=x_stft,
        )

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
                "modality_order": self.modality_order,
                **aux_embeddings,
                **{
                    f"bottleneck_{k}": v
                    for k, v in bottleneck_aux.items()
                    if k != "h_final"
                },
            }

        return logits
