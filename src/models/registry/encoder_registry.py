"""Factory registry mapping modality names to encoder builders."""

from typing import Callable

import torch.nn as nn

from src.models.encoder.gaf_encoder import GAFEncoder
from src.models.encoder.mtf_encoder import MTFEncoder
from src.models.encoder.raw_encoder import RawTimeSeriesEncoder
from src.models.encoder.stats_encoder import StatisticalEncoder
from src.models.encoder.stft_encoder import STFTEncoder

EncoderBuilder = Callable[[int, dict], nn.Module]


def build_raw_encoder(d_model: int, cfg: dict) -> nn.Module:
    """Build a 1D-CNN encoder for raw time series."""
    return RawTimeSeriesEncoder(
        in_channels=cfg["in_channels"],
        d_model=d_model,
        hidden_channels=cfg.get("hidden_channels", (64, 128, 128)),
        kernel_size=cfg.get("kernel_size", 5),
        dropout=cfg.get("dropout", 0.1),
    )


def build_stats_encoder(d_model: int, cfg: dict) -> nn.Module:
    """Build an MLP encoder for tabular statistical features."""
    return StatisticalEncoder(
        in_features=cfg["in_features"],
        d_model=d_model,
        hidden_dims=cfg.get("hidden_dims", (128, 64)),
        dropout=cfg.get("dropout", 0.2),
    )


def build_gaf_encoder(d_model: int, cfg: dict) -> nn.Module:
    """Build a 2D-CNN encoder for GAF images."""
    return GAFEncoder(
        in_channels=cfg["in_channels"],
        d_model=d_model,
        hidden_channels=cfg.get("hidden_channels", (32, 64, 128)),
        dropout=cfg.get("dropout", 0.1),
    )


def build_mtf_encoder(d_model: int, cfg: dict) -> nn.Module:
    """Build a 2D-CNN encoder for MTF images."""
    return MTFEncoder(
        in_channels=cfg["in_channels"],
        d_model=d_model,
        hidden_channels=cfg.get("hidden_channels", (32, 64, 128)),
        dropout=cfg.get("dropout", 0.1),
    )


def build_stft_encoder(d_model: int, cfg: dict) -> nn.Module:
    """Build a 2D-CNN encoder for STFT spectrograms."""
    return STFTEncoder(
        in_channels=cfg["in_channels"],
        d_model=d_model,
        hidden_channels=cfg.get("hidden_channels", (32, 64, 128)),
        dropout=cfg.get("dropout", 0.1),
    )


ENCODER_REGISTRY: dict[str, EncoderBuilder] = {
    "raw": build_raw_encoder,
    "stats": build_stats_encoder,
    "gaf": build_gaf_encoder,
    "mtf": build_mtf_encoder,
    "stft": build_stft_encoder,
}
