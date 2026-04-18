"""HF save_pretrained round-trip determinism for main (XLM-R) checkpoint.

Verifies that a model saved via save_pretrained, then reloaded via
from_pretrained, produces identical logits for the same input. Uses num_labels=2
for speed; real model has num_labels=28.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from grnti_text_classifier.models.factory import build_main


@pytest.mark.slow
def test_save_from_pretrained_round_trip(tmp_path: Path) -> None:
    model = build_main(num_labels=2)
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=True)
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    loaded = AutoModelForSequenceClassification.from_pretrained(str(tmp_path))
    loaded.train(False)
    model.train(False)
    enc = tok("Пример текста.", return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        a = model(**enc).logits
        b = loaded(**enc).logits
    assert torch.allclose(a, b, atol=1e-5)
