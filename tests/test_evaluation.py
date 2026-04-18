"""Tests for the evaluation metrics module."""
from __future__ import annotations

import numpy as np
import pytest

from grnti_text_classifier.evaluation.metrics import compute_metrics


def test_compute_metrics_keys_on_toy() -> None:
    """Verify compute_metrics returns correct keys and sensible values on toy data."""
    rng = np.random.default_rng(42)
    n = 200
    num_classes = 5

    y_true = rng.integers(0, num_classes, size=n)
    logits = rng.standard_normal((n, num_classes)).astype(np.float32)

    result = compute_metrics(y_true, logits, num_classes=num_classes)

    required_keys = {"top1_accuracy", "top5_accuracy", "macro_f1", "weighted_f1", "num_classes", "n"}
    assert required_keys == set(result.keys()), f"Missing or extra keys: {set(result.keys())}"

    assert result["n"] == 200
    assert result["num_classes"] == 5

    for key in ("top1_accuracy", "top5_accuracy", "macro_f1", "weighted_f1"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of [0, 1]: {result[key]}"

    assert result["top5_accuracy"] >= result["top1_accuracy"], (
        "top5_accuracy must be >= top1_accuracy"
    )
