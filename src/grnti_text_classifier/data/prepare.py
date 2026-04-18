"""Data preparation CLI: raw JSONL → processed Parquet + label_encoder.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grnti import (
    ENCODED_COL,
    build_label_encoder,
    load_jsonl,
    split_stratified_train_val,
)


def prepare_data(
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> None:
    """Load raw JSONL splits, build encoder, write Parquet + JSON artefacts."""
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_raw = load_jsonl(raw / "train.jsonl")
    test_df = load_jsonl(raw / "test.jsonl")

    # Build encoder from all codes present in both splits combined.
    combined = train_raw._append(test_df, ignore_index=True)
    encoder = build_label_encoder(combined)

    # Stratified train / val split.
    train_df, val_df = split_stratified_train_val(
        train_raw, val_fraction=val_fraction, seed=seed
    )

    # Add dense label index to each split.
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[ENCODED_COL] = encoder.encode(train_df["label"])
    val_df[ENCODED_COL] = encoder.encode(val_df["label"])
    test_df[ENCODED_COL] = encoder.encode(test_df["label"])

    # Write Parquet files.
    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out / "val.parquet", index=False)
    test_df.to_parquet(out / "test.parquet", index=False)

    # Write label encoder JSON (ensure_ascii=False for readable Cyrillic).
    encoder_path = out / "label_encoder.json"
    encoder_path.write_text(
        json.dumps(encoder.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[prepare] train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        f" classes={encoder.num_classes}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare GRNTI data splits.")
    p.add_argument("--raw", default="data/raw", help="Path to raw data directory")
    p.add_argument("--out", default="data/processed", help="Path to output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--val-fraction", type=float, default=0.15, help="Val split ratio")
    args = p.parse_args()
    prepare_data(args.raw, args.out, val_fraction=args.val_fraction, seed=args.seed)
