from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics.metrics import compute_classification_metrics

BatchAdapter = Callable[[Any, torch.device], tuple[tuple[torch.Tensor, ...], torch.Tensor]]


def make_experiment_params(
    dataset_name: str,
    dataset: Any,
    *,
    max_batch_size: int = 32,
    n_epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    batch_size = min(max_batch_size, len(dataset.X_train))
    channels = dataset.X_train.shape[1]
    time_steps = dataset.X_train.shape[2]
    num_classes = int(dataset.y_train.max()) + 1

    return {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "channels": channels,
        "time_steps": time_steps,
        "num_classes": num_classes,
        "n_epochs": n_epochs,
        "lr": lr,
        "seed": seed,
    }


@torch.no_grad()
def evaluate_with_batch_adapter(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    batch_adapter: BatchAdapter,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs, y = batch_adapter(batch, device)
        logits = model(*inputs)
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / total_samples
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    for x_raw, y in dataloader:
        x_raw = x_raw.to(device)
        y = y.to(device)

        logits = model(x_raw)
        loss = criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / total_samples
    return metrics


def display_evaluation_results(metrics: dict[str, Any]) -> None:
    print(f"Loss: {metrics['loss']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Macro F1: {metrics['macro_f1']}")
    print(f"Weighted F1: {metrics['weighted_f1']}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(metrics["classification_report"])


def make_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def save_json_result(result: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(result), file, ensure_ascii=False, indent=2)


def make_summary_row(
    result: dict[str, Any],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    row = {
        "method": result["method"],
        "dataset_name": result["dataset_name"],
        "train_size": result["train_size"],
        "test_size": result["test_size"],
        "total_size": result["total_size"],
        "channels": result["channels"],
        "time_steps": result["time_steps"],
        "num_classes": result["num_classes"],
    }
    row.update({metric_name: result[metric_name] for metric_name in metric_names})
    return row


def save_summary_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError("rows must not be empty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(make_json_safe(row) for row in rows)
