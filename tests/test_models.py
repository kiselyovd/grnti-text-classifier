"""Tests for model factories and GRNTIClassifier Lightning module."""
from __future__ import annotations

import os

import pytest
import torch
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification


@pytest.mark.skipif("HF_OFFLINE" in os.environ, reason="no internet")
def test_build_main_num_labels():
    from grnti_text_classifier.models.factory import build_main

    model = build_main(num_labels=28)
    assert model.config.num_labels == 28


def test_step_shapes():
    from grnti_text_classifier.models.lightning_module import GRNTIClassifier

    cfg = XLMRobertaConfig(
        vocab_size=250002,
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=28,
    )
    model = XLMRobertaForSequenceClassification(cfg)

    lit = GRNTIClassifier(model, num_classes=28, total_steps=100)
    lit.train(False)

    batch = {
        "input_ids": torch.randint(0, 250002, (2, 16)),
        "attention_mask": torch.ones((2, 16), dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }

    loss, logits, preds = lit._step(batch)

    assert loss.shape == (), f"expected scalar loss, got shape {loss.shape}"
    assert logits.shape == (2, 28), f"expected (2, 28) logits, got {logits.shape}"
    assert preds.shape == (2,), f"expected (2,) preds, got {preds.shape}"
