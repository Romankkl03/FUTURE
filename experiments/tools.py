from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

FUSION_OVER_RAW_DIR = Path("results/fusion_over_raw")
FUSION_OVER_RAW_TABLES_DIR = FUSION_OVER_RAW_DIR / "tables"
FUSION_OVER_RAW_PLOTS_DIR = FUSION_OVER_RAW_DIR / "plots"
FUSION_OVER_RAW_SUMMARY_CSV = FUSION_OVER_RAW_TABLES_DIR / "summary.csv"


def fusion_over_raw_tables_dir(output_dir: Path) -> Path:
    return output_dir / "tables"


def fusion_over_raw_plots_dir(output_dir: Path) -> Path:
    return output_dir / "plots"


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
