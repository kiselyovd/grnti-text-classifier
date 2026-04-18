"""CLI: score a saved HF checkpoint on a processed parquet split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .metrics import compute_metrics


def main() -> None:
    """Entry point for the scoring CLI."""
    parser = argparse.ArgumentParser(
        description="Score a saved HF checkpoint on a processed parquet split."
    )
    parser.add_argument("--hf-dir", required=True, help="HF model directory from train_one")
    parser.add_argument(
        "--split", required=True, help="Parquet file path (e.g. data/processed/test.parquet)"
    )
    parser.add_argument(
        "--label-encoder",
        required=True,
        help="Label encoder JSON (e.g. data/processed/label_encoder.json)",
    )
    parser.add_argument("--out", required=True, help="Output metrics JSON path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.hf_dir)
    model.to(device)
    model.train(False)

    df = pd.read_parquet(args.split)
    with open(args.label_encoder, encoding="utf-8") as fh:
        encoder = json.load(fh)
    num_classes = len(encoder)

    texts = df["text"].tolist()
    all_logits: list[np.ndarray] = []

    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model(**inputs)
        all_logits.append(out.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = df["label_idx"].to_numpy()

    metrics = compute_metrics(y_true, logits, num_classes=num_classes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(metrics)


if __name__ == "__main__":
    main()
