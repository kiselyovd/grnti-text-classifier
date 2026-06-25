"""Generate model-card visualizations from real test predictions.

Produces a per-class top-1 F1 horizontal bar chart over the 28 GRNTI sections,
computed by running the HF-native XLM-RoBERTa model on the held-out test split.

The overall macro-F1 is printed so it can be cross-checked against
``reports/metrics.json`` (expected ~0.723).

Usage:
    uv run python scripts/make_plots.py \\
        --model-dir artifacts/main/hf \\
        --test data/processed/test.parquet \\
        --label-encoder data/processed/label_encoder.json \\
        --out reports/per_class_f1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# DejaVu Sans ships with matplotlib and renders Cyrillic correctly.
plt.rcParams["font.family"] = "DejaVu Sans"


def _section_labels(encoder_path: Path) -> dict[int, str]:
    """Return idx -> short Russian section name (code-disambiguated)."""
    enc = json.loads(encoder_path.read_text(encoding="utf-8"))
    idx_to_text = enc["idx_to_text"]
    idx_to_code = enc["idx_to_code"]
    # Detect duplicate names; append code to disambiguate (e.g. 680000 vs 683500).
    counts: dict[str, int] = {}
    for name in idx_to_text.values():
        counts[name] = counts.get(name, 0) + 1
    labels: dict[int, str] = {}
    for k, name in idx_to_text.items():
        idx = int(k)
        if counts[name] > 1:
            labels[idx] = f"{name} ({idx_to_code[k]})"
        else:
            labels[idx] = name
    return labels


def _predict(
    model_dir: Path,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Run the HF-native classifier on CPU and return argmax predictions."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to("cpu")
    model.train(False)

    preds: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).tolist())
            done = min(start + batch_size, len(texts))
            print(f"  inference {done}/{len(texts)}", end="\r", flush=True)
    print()
    return np.asarray(preds, dtype=np.int64)


def _plot_per_class_f1(
    f1_per_class: np.ndarray,
    labels: dict[int, str],
    macro_f1: float,
    n_test: int,
    out_path: Path,
) -> None:
    """Render a horizontal bar chart of per-class F1, sorted descending."""
    order = np.argsort(f1_per_class)  # ascending so largest ends up on top
    names = [labels[i] for i in order]
    values = f1_per_class[order]

    cmap = plt.get_cmap("viridis")
    colors = cmap(values / max(values.max(), 1e-9))

    fig, ax = plt.subplots(figsize=(10, 11))
    bars = ax.barh(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Top-1 F1", fontsize=11)
    ax.set_title(
        "Per-class top-1 F1 across 28 GRNTI sections\n"
        f"(XLM-RoBERTa-base, test n={n_test}, macro-F1 = {macro_f1:.3f})",
        fontsize=12,
    )
    for rect, val in zip(bars, values, strict=True):
        ax.text(
            rect.get_width() + 0.01,
            rect.get_y() + rect.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
            color="#333333",
        )
    ax.axvline(macro_f1, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        macro_f1,
        len(values) - 0.3,
        f" macro-F1 {macro_f1:.3f}",
        color="#d62728",
        fontsize=9,
        va="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Make per-class F1 plot from test predictions.")
    p.add_argument("--model-dir", default="artifacts/main/hf")
    p.add_argument("--test", default="data/processed/test.parquet")
    p.add_argument("--label-encoder", default="data/processed/label_encoder.json")
    p.add_argument("--out", default="reports/per_class_f1.png")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()

    df = pd.read_parquet(args.test)
    texts = df["text"].astype(str).tolist()
    y_true = df["label_idx"].to_numpy(dtype=np.int64)

    labels = _section_labels(Path(args.label_encoder))
    num_classes = len(labels)

    print(f"Running inference on {len(texts)} test rows (CPU)...")
    y_pred = _predict(
        Path(args.model_dir),
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    top1 = float((y_pred == y_true).mean())
    f1_per_class = f1_score(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )

    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Macro-F1:       {macro_f1:.4f}  (cross-check against reports/metrics.json)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_per_class_f1(np.asarray(f1_per_class), labels, macro_f1, len(texts), out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
