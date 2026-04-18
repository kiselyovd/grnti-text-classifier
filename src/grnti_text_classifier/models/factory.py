"""Model factories — return a pretrained HuggingFace model ready for fine-tuning."""

from __future__ import annotations

from transformers import AutoModelForSequenceClassification, PreTrainedModel


def build_main(num_labels: int = 28) -> PreTrainedModel:
    """Return XLM-RoBERTa-base configured for sequence classification.

    Args:
        num_labels: Number of output classes (default 28 for GRNTI).

    Returns:
        AutoModelForSequenceClassification instance.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-base",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )


def build_baseline(num_labels: int = 28) -> PreTrainedModel:
    """Return ruBERT-base-cased configured for sequence classification.

    Args:
        num_labels: Number of output classes (default 28 for GRNTI).

    Returns:
        AutoModelForSequenceClassification instance.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        "DeepPavlov/rubert-base-cased",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
