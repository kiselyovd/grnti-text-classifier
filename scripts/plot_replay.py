"""Render numerical figures from the validated replay, without another model download."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from artifact_manifest import validate
from sklearn.metrics import confusion_matrix, f1_score


def main() -> None:
    """Check report/array parity before rendering all exact GRNTI class labels."""
    root = Path(__file__).resolve().parents[1]
    manifest, _ = validate(root)
    report = json.loads((root / manifest["evaluation"]["report"]).read_text(encoding="utf-8"))
    with np.load(root / "reports" / report["arrays"]["path"], allow_pickle=False) as arrays:
        targets, predicted = arrays["targets"], arrays["logits"].argmax(1)
    cm = confusion_matrix(targets, predicted, labels=list(range(28)))
    f1 = f1_score(targets, predicted, labels=list(range(28)), average=None, zero_division=0)
    np.testing.assert_array_equal(cm, report["confusion_matrix"])
    np.testing.assert_allclose(f1, report["per_class_f1"], atol=1e-12, rtol=0)
    contract = json.loads((root / "space/model-contract.json").read_text(encoding="utf-8"))
    labels = [contract["id2label"][str(i)] for i in range(28)]
    output = root / "docs/images"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.barh(np.arange(28), f1[order], color="#0891b2")
    ax.set_yticks(np.arange(28), [labels[i] for i in order], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-class F1")
    ax.axvline(report["metrics"]["macro_f1"], color="#475569", linestyle="--", label="Macro F1")
    ax.legend(loc="lower right")
    ax.set_title("Pinned native replay · 2,772 source abstracts · 256 tokens")
    fig.tight_layout()
    fig.savefig(output / "per_class_f1.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm / cm.sum(1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    codes = [label.split(":")[0] for label in labels]
    ax.set_xticks(range(28), codes, rotation=90, fontsize=7)
    ax.set_yticks(range(28), codes, fontsize=7)
    ax.set_xlabel("Predicted GRNTI code")
    ax.set_ylabel("Source GRNTI code")
    ax.set_title("Native replay · row-normalised confusion matrix")
    fig.colorbar(image, ax=ax, label="Fraction of source class")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
