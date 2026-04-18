"""Tests for GRNTIDataModule."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from grnti_text_classifier.data.datamodule import GRNTIDataModule
from grnti_text_classifier.data.grnti import ENCODED_COL, LABEL_COL, TEXT_COL


def _make_parquet(path, n_rows: int, n_classes: int) -> None:
    """Write a tiny parquet file with *n_rows* rows and *n_classes* labels."""
    rows = {
        "id": list(range(n_rows)),
        TEXT_COL: ["пример текста"] * n_rows,
        LABEL_COL: [i % n_classes for i in range(n_rows)],
        ENCODED_COL: [i % n_classes for i in range(n_rows)],
    }
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_datamodule_batch_shape(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    _make_parquet(processed / "train.parquet", n_rows=6, n_classes=3)
    _make_parquet(processed / "val.parquet", n_rows=4, n_classes=3)
    _make_parquet(processed / "test.parquet", n_rows=4, n_classes=3)

    dm = GRNTIDataModule(
        processed_dir=processed,
        model_name="FacebookAI/xlm-roberta-base",
        batch_size=4,
        max_length=32,
        num_workers=0,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    assert batch["input_ids"].shape == (4, 32)
    assert batch["attention_mask"].shape == (4, 32)
    assert batch["labels"].shape == (4,)
    assert batch["labels"].dtype == torch.long
