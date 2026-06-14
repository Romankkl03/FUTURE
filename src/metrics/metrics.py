"""Classification metrics for multiclass experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute a standard bundle of multiclass classification metrics.

    Aggregates sklearn scores used across fusion-over-raw experiments. Primary
    headline metrics are ``accuracy``, ``balanced_accuracy``, ``macro_f1``, and
    ``weighted_f1``; macro averages treat every class equally and are preferred
    when class counts differ, while weighted averages follow the empirical class
    distribution.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels, shape ``(n_samples,)``.
    y_pred:
        Predicted integer labels, shape ``(n_samples,)``.
    class_names:
        Optional human-readable class names passed to
        :func:`sklearn.metrics.classification_report`. When ``None``, sklearn
        uses numeric label strings.

    Returns
    -------
    metrics:
        Dictionary with the following keys:

        - ``accuracy`` — fraction of correctly classified samples.
        - ``balanced_accuracy`` — mean per-class recall; robust to imbalance.
        - ``macro_precision``, ``macro_recall``, ``macro_f1`` — unweighted
          mean over classes (equal weight per class).
        - ``weighted_precision``, ``weighted_recall``, ``weighted_f1`` —
          support-weighted mean over classes.
        - ``per_class_precision``, ``per_class_recall``, ``per_class_f1`` —
          one value per class (numpy arrays).
        - ``confusion_matrix`` — raw counts, shape ``(n_classes, n_classes)``.
        - ``normalized_confusion_matrix`` — row-normalized confusion matrix
          (each row sums to 1).
        - ``classification_report`` — human-readable sklearn report string.
        - ``classification_report_dict`` — same report as a nested dictionary.

        All precision/recall/F1 scores use ``zero_division=0``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, Any] = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    metrics["macro_precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_f1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics["weighted_precision"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["weighted_recall"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["weighted_f1"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["per_class_precision"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["per_class_recall"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["per_class_f1"] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["normalized_confusion_matrix"] = confusion_matrix(
        y_true, y_pred, normalize="true"
    )

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    metrics["classification_report_dict"] = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return metrics
