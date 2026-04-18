"""Tests for grnti_text_classifier.data.grnti helpers."""

from __future__ import annotations

import json

import pandas as pd

from grnti_text_classifier.data.grnti import (
    FEATURES,
    LABEL_COL,
    LabelEncoder,
    build_label_encoder,
    load_jsonl,
    split_stratified_train_val,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_df(codes: list[int], n_per_code: int = 50) -> pd.DataFrame:
    """Create a toy DataFrame with *n_per_code* rows per code."""
    rows = []
    for i, code in enumerate(codes):
        for j in range(n_per_code):
            rows.append(
                {
                    "id": i * n_per_code + j,
                    "label": code,
                    "text": f"sample text {i} {j}",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: load_jsonl round-trip
# ---------------------------------------------------------------------------


def test_load_jsonl_round_trip(tmp_path):
    """Writing 5 rows as JSONL then loading should preserve shape and columns."""
    data = [{"id": i, "label": 20000 + i * 10000, "text": f"text {i}"} for i in range(5)]
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row) for row in data), encoding="utf-8")

    df = load_jsonl(jsonl_path)

    assert df.shape == (5, 3), f"Expected (5, 3), got {df.shape}"
    assert set(df.columns) == set(FEATURES)


# ---------------------------------------------------------------------------
# Test 2: LabelEncoder round-trip (encode → decode, serialise → deserialise)
# ---------------------------------------------------------------------------


def test_label_encoder_round_trip(tmp_path):
    """Encode then decode should be an identity; JSON serialisation round-trips."""
    codes = [100000, 270000, 500000]
    df = _make_toy_df(codes, n_per_code=10)
    encoder = build_label_encoder(df)

    assert encoder.num_classes == 3

    # encode → decode identity
    labels = pd.Series(codes)
    indices = encoder.encode(labels)
    decoded = [encoder.decode(int(idx)) for idx in indices]
    assert decoded == codes, f"decode mismatch: {decoded} != {codes}"

    # text labels are non-empty strings
    for idx in range(encoder.num_classes):
        assert isinstance(encoder.decode_text(idx), str)
        assert len(encoder.decode_text(idx)) > 0

    # JSON serialisation round-trip
    d = encoder.to_json_dict()
    json_str = json.dumps(d, ensure_ascii=False)
    restored = LabelEncoder.from_json_dict(json.loads(json_str))

    assert restored.num_classes == encoder.num_classes
    assert restored.code_to_idx == encoder.code_to_idx
    assert restored.idx_to_code == encoder.idx_to_code
    assert restored.idx_to_text == encoder.idx_to_text


# ---------------------------------------------------------------------------
# Test 3: stratified split preserves per-class proportions
# ---------------------------------------------------------------------------


def test_split_stratified_ratio(tmp_path):
    """Per-class rate in train ≈ 0.85 and in val ≈ 0.15 (delta ≤ 0.05)."""
    codes = [20000, 140000, 290000, 760000]
    df = _make_toy_df(codes, n_per_code=50)  # 200 rows, 50 per class

    train_df, val_df = split_stratified_train_val(df, val_fraction=0.15, seed=42)

    total = len(df)
    assert len(train_df) + len(val_df) == total

    for code in codes:
        n_code = (df[LABEL_COL] == code).sum()
        n_train = (train_df[LABEL_COL] == code).sum()
        n_val = (val_df[LABEL_COL] == code).sum()

        train_rate = n_train / n_code
        val_rate = n_val / n_code

        assert abs(train_rate - 0.85) <= 0.05, (
            f"code={code}: train_rate={train_rate:.3f} outside [0.80, 0.90]"
        )
        assert abs(val_rate - 0.15) <= 0.05, (
            f"code={code}: val_rate={val_rate:.3f} outside [0.10, 0.20]"
        )
