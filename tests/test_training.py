"""1-epoch smoke test of train_one on tiny in-memory splits with 2 labels.

Uses a 1-layer XLMRobertaForSequenceClassification built in-memory to avoid HF
downloads in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch  # noqa: F401
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification

from grnti_text_classifier.training.train import train_one


def _tiny_builder(num_labels: int):
    cfg = XLMRobertaConfig(
        vocab_size=250002,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=num_labels,
        max_position_embeddings=256,
        type_vocab_size=1,
    )
    return XLMRobertaForSequenceClassification(cfg)


@pytest.mark.slow
def test_train_one_smoke(tmp_path: Path) -> None:
    """train_one completes a 1-epoch run and writes HF artefacts."""
    processed = tmp_path / "processed"
    processed.mkdir()

    rows = [
        {
            "id": str(i),
            "text": f"пример текста {i}",
            "label": i % 2 * 10000,
            "label_idx": i % 2,
        }
        for i in range(20)
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(processed / "train.parquet", index=False)
    df.to_parquet(processed / "val.parquet", index=False)
    df.to_parquet(processed / "test.parquet", index=False)

    encoder = {
        "code_to_idx": {"0": 0, "10000": 1},
        "idx_to_code": {"0": 0, "1": 10000},
        "idx_to_text": {"0": "class0", "1": "class1"},
        "num_classes": 2,
    }
    (processed / "label_encoder.json").write_text(json.dumps(encoder), encoding="utf-8")

    out = tmp_path / "out"
    hf_dir = train_one(
        _tiny_builder,
        "FacebookAI/xlm-roberta-base",
        processed,
        out,
        max_epochs=1,
        batch_size=4,
        patience=1,
        max_length=16,
        num_workers=0,
    )

    assert (hf_dir / "config.json").is_file()
    assert (hf_dir / "tokenizer.json").is_file() or (hf_dir / "sentencepiece.bpe.model").is_file()
