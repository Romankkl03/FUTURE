from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

BatchAdapter = Callable[[Any, torch.device], tuple[tuple[torch.Tensor, ...], torch.Tensor]]


def train_with_batch_adapter(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    *,
    n_epochs: int,
    lr: float,
    batch_adapter: BatchAdapter,
) -> list[float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses: list[float] = []

    for _ in tqdm(range(n_epochs)):
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            inputs, y = batch_adapter(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            running += loss.item() * batch_size
            n_seen += batch_size
        epoch_losses.append(running / max(n_seen, 1))

    return epoch_losses


def single_input_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[tuple[torch.Tensor], torch.Tensor]:
    x, y = batch
    return (x.to(device),), y.to(device)


def two_input_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    x_left, x_right, y = batch
    return (x_left.to(device), x_right.to(device)), y.to(device)
