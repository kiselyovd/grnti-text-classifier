"""Confusion matrix visualisation — saves a seaborn heatmap PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def save_confusion_matrix(
    y_true: np.ndarray,
    preds: np.ndarray,
    labels: list[str],
    out_path: "str | Path",
) -> None:
    """Save a row-normalised confusion matrix heatmap to *out_path* (PNG).

    Parameters
    ----------
    y_true:
        Ground-truth integer labels, shape ``(n,)``.
    preds:
        Predicted integer labels, shape ``(n,)``.
    labels:
        Human-readable class names (e.g. ``["Математика", "Информатика"]``).
        Length must equal the number of classes.
    out_path:
        Destination file path.  Parent directories are created if absent.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(
        y_true,
        preds,
        labels=list(range(len(labels))),
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Confusion matrix (main model, row-normalised)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close("all")
