"""Pinned, safetensors-only GRNTI inference shared with the standalone demo."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CONTRACT = json.loads((Path(__file__).parent / "model-contract.json").read_text(encoding="utf-8"))
MAX_LENGTH = CONTRACT["max_length"]
MAX_CHARACTERS = 20000


def parse_labels(mapping: Mapping[Any, str]) -> dict[int, dict[str, str]]:
    """Preserve the checkpoint's exact code strings and full names, including duplicates."""
    labels = {}
    for key, value in mapping.items():
        code, separator, title = value.partition(":")
        code, title = code.strip(), title.strip()
        if not separator or not code.isdecimal() or not title:
            raise ValueError("Malformed checkpoint GRNTI label")
        labels[int(key)] = {"code": code, "title": title}
    if sorted(labels) != list(range(CONTRACT["num_labels"])):
        raise ValueError("Checkpoint must define exactly 28 contiguous output labels")
    return labels


def rank_logits(logits: torch.Tensor, labels: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    """Rank real model logits; scores are uncalibrated softmax outputs."""
    if logits.shape != (1, CONTRACT["num_labels"]) or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite 28-class prediction")
    probabilities = logits.softmax(-1)[0].detach().cpu()
    indices = torch.argsort(probabilities, descending=True, stable=True)[:5].tolist()
    return [
        {"index": index, **labels[index], "score": float(probabilities[index])} for index in indices
    ]


def validate_text(text: str) -> str:
    """Bound the visitor input before tokenization without logging its content."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty_text")
    if len(cleaned) > MAX_CHARACTERS:
        raise ValueError("text_too_long")
    return cleaned


class NativeClassifier:
    """Load the reviewed main checkpoint once and execute its 256-token contract."""

    def __init__(self, device: str = "cpu", model_dir: str | Path | None = None) -> None:
        self.device = device
        if model_dir is None:
            source = CONTRACT["repo_id"]
            options: dict[str, Any] = {"revision": CONTRACT["revision"], "token": False}
            weights = Path(
                hf_hub_download(
                    source, CONTRACT["weights_file"], revision=CONTRACT["revision"], token=False
                )
            )
        else:
            source = str(model_dir)
            options = {"local_files_only": True}
            weights = Path(model_dir) / CONTRACT["weights_file"]
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != CONTRACT["weights_sha256"]:
            raise ValueError("Checkpoint weights SHA256 differs from reviewed main artifact")
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **options)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                source, use_safetensors=True, trust_remote_code=False, **options
            )
            .to(device)
            .eval()
        )
        if {str(k): v for k, v in self.model.config.id2label.items()} != CONTRACT["id2label"]:
            raise ValueError("Native label map differs from the reviewed checkpoint")
        self.labels = parse_labels(self.model.config.id2label)

    def classify(self, text: str) -> dict[str, Any]:
        """Return top five exact labels and an accurate truncation indicator."""
        text = validate_text(text)
        full_length = len(self.tokenizer(text, truncation=False, verbose=False)["input_ids"])
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        return {
            "predictions": rank_logits(logits, self.labels),
            "input_tokens": int(encoded["attention_mask"].sum().item()),
            "truncated": full_length > MAX_LENGTH,
            "model_revision": CONTRACT["revision"],
        }
