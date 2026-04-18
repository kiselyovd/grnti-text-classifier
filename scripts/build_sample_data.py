"""Build sample fixtures for HF widget and qualitative inspection.

Outputs:
  data/sample/sample.jsonl       — 20 sample abstracts (one per class)
  data/widget/sample_payload.json — single-text widget payload
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sample + widget fixtures.")
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--out-sample", default="data/sample/sample.jsonl", type=Path)
    parser.add_argument("--out-widget", default="data/widget/sample_payload.json", type=Path)
    parser.add_argument("--n", default=20, type=int, help="Number of classes to sample from")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    processed_dir: Path = args.processed_dir
    out_sample: Path = args.out_sample
    out_widget: Path = args.out_widget

    # Load data
    df = pd.read_parquet(processed_dir / "test.parquet")
    with open(processed_dir / "label_encoder.json", encoding="utf-8") as fh:
        le = json.load(fh)

    idx_to_text: dict[str, str] = le["idx_to_text"]
    idx_to_code: dict[str, int] = le["idx_to_code"]

    # Pick top-N most-populous classes
    class_counts = df["label_idx"].value_counts()
    n_classes = min(args.n, len(class_counts))
    top_idxs = class_counts.head(n_classes).index.tolist()

    # Sample one abstract per selected class
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []

    for idx in top_idxs:
        subset = df[df["label_idx"] == idx]
        row = subset.iloc[int(rng.integers(0, len(subset)))]
        records.append(
            {
                "id": int(row["id"]),
                "text": str(row["text"]),
                "label": str(row["label"]),
                "label_idx": int(row["label_idx"]),
                "label_text": idx_to_text[str(idx)],
            }
        )

    # Write sample.jsonl
    out_sample.parent.mkdir(parents=True, exist_ok=True)
    with open(out_sample, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {out_sample}")

    # Write widget payload (first sample's text)
    out_widget.parent.mkdir(parents=True, exist_ok=True)
    widget = {"text": records[0]["text"], "max_length": 256}
    with open(out_widget, "w", encoding="utf-8") as fh:
        json.dump(widget, fh, ensure_ascii=False, indent=2)

    print(f"Wrote widget payload to {out_widget}")
    print(f"Widget text length: {len(widget['text'])}")


if __name__ == "__main__":
    main()
