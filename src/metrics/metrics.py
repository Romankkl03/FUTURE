import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    class_names=None,
):
    """
    y_true: array-like, shape [num_samples]
    y_pred: array-like, shape [num_samples]
    """

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    metrics["macro_f1"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["weighted_f1"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
    )

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return metrics