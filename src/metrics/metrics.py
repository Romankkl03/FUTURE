import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
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
    Метрики для multiclass classification.

    Parameters
    ----------
    y_true:
        array-like, shape [num_samples]
        Истинные метки классов.

    y_pred:
        array-like, shape [num_samples]
        Предсказанные метки классов.

    class_names:
        list[str] | None
        Названия классов для classification_report.
        Если None, sklearn использует числовые labels.

    Returns
    -------
    metrics:
        dict
        Словарь с основными multiclass-метриками.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {}

    # --------------------------------------------------
    # Main metrics
    # --------------------------------------------------

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------
    # Macro metrics
    # Каждому классу дается одинаковый вес.
    # Особенно важно для несбалансированных классов.
    # --------------------------------------------------

    metrics["macro_precision"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["macro_recall"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["macro_f1"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------
    # Weighted metrics
    # Взвешены по количеству объектов каждого класса.
    # Полезно как дополнительная метрика.
    # --------------------------------------------------

    metrics["weighted_precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["weighted_recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["weighted_f1"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------
    # Per-class metrics
    # Нужны, чтобы понять, какие классы проседают.
    # --------------------------------------------------

    metrics["per_class_precision"] = precision_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    metrics["per_class_recall"] = recall_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    metrics["per_class_f1"] = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    metrics["confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
    )

    metrics["normalized_confusion_matrix"] = confusion_matrix(
        y_true,
        y_pred,
        normalize="true",
    )

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    metrics["classification_report_dict"] = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return metrics