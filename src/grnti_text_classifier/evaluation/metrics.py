"""Metrics computation for classification scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score


def compute_metrics(
    y_true: np.ndarray,
    logits: np.ndarray | object,
    num_classes: int,
) -> dict[str, Any]:
    """Return top-1/top-5 accuracy, macro/weighted F1, num_classes, and n.

    Parameters
    ----------
    y_true:
        Integer class indices, shape ``(n,)``.
    logits:
        Raw model outputs, shape ``(n, num_classes)``.  Accepts either a
        NumPy array or a torch.Tensor — tensors are converted to NumPy
        automatically.
    num_classes:
        Total number of label classes.

    Returns
    -------
    dict with keys: top1_accuracy, top5_accuracy, macro_f1, weighted_f1,
    num_classes, n.
    """
    # Accept torch tensors without importing torch at module level.
    if hasattr(logits, "cpu"):
        logits = logits.cpu().numpy()
    logits = np.asarray(logits, dtype=np.float32)

    labels = list(range(num_classes))
    preds = logits.argmax(axis=-1)

    top1 = float(top_k_accuracy_score(y_true, logits, k=1, labels=labels))
    top5 = float(top_k_accuracy_score(y_true, logits, k=min(5, num_classes), labels=labels))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    return {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_classes": int(num_classes),
        "n": len(y_true),
    }
