import torch
import torch.nn as nn


from .cnn_encoder import ClassificationHead, RawTimeSeriesEncoder
from .gaf_encoder import GAFEncoder
from .stats_encoder import StatisticalEncoder
from .stft_encoder import STFTEncoder


class BottleneckLatentBlock(nn.Module):
    """
    One bottleneck block:

    1. Cross-attention:
        Q = latent tokens
        K,V = modality tokens

    2. Self-attention:
        Q,K,V = latent tokens

    3. Feed-forward network over latent tokens
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.modality_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attn_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(d_model)

        hidden_dim = int(d_model * mlp_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        modality_tokens: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        latents:         [B, K, D]
        modality_tokens: [B, M, D]
        """

        # Cross-attention: latents gather information from modality tokens.
        q = self.cross_attn_norm(latents)
        kv = self.modality_norm(modality_tokens)

        cross_out, cross_attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        latents = latents + cross_out

        # Self-attention between latent tokens.
        qkv = self.self_attn_norm(latents)

        self_out, self_attn_weights = self.self_attn(
            query=qkv,
            key=qkv,
            value=qkv,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        latents = latents + self_out

        # Feed-forward.
        latents = latents + self.ffn(self.ffn_norm(latents))

        if return_attn:
            return latents, {
                "cross_attn": cross_attn_weights,
                "self_attn": self_attn_weights,
            }

        return latents


class BottleneckRepresentationEncoder(nn.Module):
    """
    Bottleneck Representation Encoder.

    Input:
        modality embeddings:
            h_raw, h_stats, h_gaf, h_stft
            each [B, D]

    Steps:
        1. stack modality embeddings -> [B, M, D]
        2. add learnable modality embeddings
        3. create learnable latent tokens -> [B, K, D]
        4. latent tokens cross-attend to modality tokens
        5. latent tokens self-attend
        6. pool latent tokens into one representation

    pooling:
        "mean"   -> latents.mean(dim=1)
        "cls"    -> latents[:, 0]
        "concat" -> projection(flatten(latents))
    """

    def __init__(
        self,
        n_modalities: int,
        d_model: int,
        num_latents: int = 4,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pooling: str = "mean",
        latent_init_std: float = 0.02,
    ):
        super().__init__()

        if n_modalities <= 0:
            raise ValueError("n_modalities must be > 0.")

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

        self.n_modalities = n_modalities
        self.d_model = d_model
        self.num_latents = num_latents
        self.pooling = pooling

        # Learnable modality embeddings:
        # RAW, STATS, GAF, STFT, ...
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, n_modalities, d_model)
        )

        # Learnable latent bottleneck tokens:
        # Z1, Z2, ..., ZK
        self.latent_tokens = nn.Parameter(
            torch.randn(num_latents, d_model) * latent_init_std
        )

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
        *modality_embeddings: torch.Tensor,
        return_aux: bool = False,
    ):
        if len(modality_embeddings) != self.n_modalities:
            raise ValueError(
                f"Expected {self.n_modalities} modality embeddings, "
                f"got {len(modality_embeddings)}."
            )

        # [B, M, D]
        modality_tokens = torch.stack(modality_embeddings, dim=1)

        batch_size, n_modalities, d_model = modality_tokens.shape

        if n_modalities != self.n_modalities:
            raise ValueError(
                f"Expected n_modalities={self.n_modalities}, got {n_modalities}."
            )

        if d_model != self.d_model:
            raise ValueError(
                f"Expected d_model={self.d_model}, got {d_model}."
            )

        # Add modality identity embeddings.
        # [B, M, D]
        modality_tokens = modality_tokens + self.modality_embedding

        # [B, K, D]
        latents = self.latent_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        all_attn = []

        for block in self.blocks:
            if return_aux:
                latents, attn = block(
                    latents,
                    modality_tokens,
                    return_attn=True,
                )
                all_attn.append(attn)
            else:
                latents = block(
                    latents,
                    modality_tokens,
                    return_attn=False,
                )

        # [B, K, D]
        latents = self.final_norm(latents)

        if self.pooling == "mean":
            # [B, D]
            h_final = latents.mean(dim=1)

        elif self.pooling == "cls":
            # [B, D]
            h_final = latents[:, 0]

        elif self.pooling == "concat":
            # [B, K * D]
            h_flat = latents.reshape(batch_size, self.num_latents * self.d_model)

            # [B, D]
            h_final = self.concat_projection(h_flat)

        else:
            raise RuntimeError(f"Unexpected pooling mode: {self.pooling}")

        if return_aux:
            return {
                "h_final": h_final,
                "modality_tokens": modality_tokens,
                "latents": latents,
                "attn": all_attn,
            }

        return h_final


class RawStatsGAFSTFTBottleneckClassifier(nn.Module):
    """
    RAW + STATS + GAF + STFT classifier with Bottleneck Representation Encoder.

    Modalities:
        RAW
        STATS
        GAF
        STFT

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
        kernel_size: int = 5,
        raw_dropout: float = 0.1,
        stat_dropout: float = 0.2,
        image_dropout: float = 0.1,
        bottleneck_dropout: float = 0.1,
        head_dropout: float = 0.2,
        num_latents: int = 4,
        num_bottleneck_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        pooling: str = "mean",
    ):
        super().__init__()

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
        x_stft: torch.Tensor,
        return_aux: bool = False,
    ):
        h_raw = self.raw_encoder(x_raw)
        embeddings = [h_raw]
        aux_embeddings = {"h_raw": h_raw}

        for modality in self.context_modalities:
            if modality == "stats":
                if x_stat is None:
                    raise ValueError("x_stat is required because 'stats' is enabled.")
                h_stat = self.stat_encoder(x_stat)
                embeddings.append(h_stat)
                aux_embeddings["h_stat"] = h_stat
            elif modality == "gaf":
                if x_gaf is None:
                    raise ValueError("x_gaf is required because 'gaf' is enabled.")
                h_gaf = self.gaf_encoder(x_gaf)
                embeddings.append(h_gaf)
                aux_embeddings["h_gaf"] = h_gaf
            elif modality == "stft":
                if x_stft is None:
                    raise ValueError("x_stft is required because 'stft' is enabled.")
                h_stft = self.stft_encoder(x_stft)
                embeddings.append(h_stft)
                aux_embeddings["h_stft"] = h_stft
            else:
                raise RuntimeError(f"Unexpected modality: {modality}")

        if return_aux:
            aux = self.bottleneck(
                *embeddings,
                return_aux=True,
            )

            logits = self.head(aux["h_final"])

            aux["logits"] = logits
            aux.update(aux_embeddings)
            aux["modality_order"] = self.modality_order

            return aux

        h_final = self.bottleneck(
            *embeddings,
            return_aux=False,
        )

        return self.head(h_final)