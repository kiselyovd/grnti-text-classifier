"""Replay the reviewed native checkpoint on a hash-identified source test JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from grnti_text_classifier.evaluation.metrics import compute_metrics
from grnti_text_classifier.inference.native import CONTRACT, MAX_LENGTH, NativeClassifier

SOURCE_REVISION = "a984f8a65b2e9fae7a347bf7761b0eaad79c73a1"
SOURCE_SHA256 = "f8b6a7c7050f4f8f76de8ab301b7b771e47b089900c06618d103369c5c10afbe"
log = logging.getLogger(__name__)


def read_test(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Reject changed bytes, duplicate IDs, missing classes and mismatched labels."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Test source differs from the pinned JSONL")
    rows = [json.loads(line) for line in raw.splitlines()]
    mapping = {value.split(":", 1)[0]: int(key) for key, value in CONTRACT["id2label"].items()}
    if len(rows) != 2772 or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Incomplete test split or duplicate source IDs")
    targets = np.asarray([mapping[str(row["label"])] for row in rows], dtype=np.int64)
    if set(targets) != set(range(28)):
        raise ValueError("Test source must contain all 28 checkpoint classes")
    return rows, targets


def main() -> None:
    """Execute CPU or explicitly selected CUDA inference with reproducible outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/replay-native.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and threads must be positive")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(args.threads)
    rows, targets = read_test(args.test_jsonl)
    classifier = NativeClassifier(device=args.device)
    started = time.monotonic()
    batches = []
    for start in range(0, len(rows), args.batch_size):
        encoded = classifier.tokenizer(
            [row["text"] for row in rows[start : start + args.batch_size]],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        with torch.inference_mode():
            batches.append(classifier.model(**encoded).logits.detach().cpu().numpy())
        if start % (args.batch_size * 25) == 0:
            log.info("Replayed %d/%d abstracts", min(start + args.batch_size, len(rows)), len(rows))
    logits = np.concatenate(batches)
    metrics = compute_metrics(targets, logits, num_classes=28)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays = args.out.with_suffix(".npz")
    np.savez_compressed(
        arrays,
        logits=logits,
        targets=targets,
        source_ids=np.asarray([str(row["id"]) for row in rows]),
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()
    report = {
        "schema_version": 1,
        "training_membership_verified": False,
        "artifact": {
            "repo_id": CONTRACT["repo_id"],
            "revision": CONTRACT["revision"],
            "file": CONTRACT["weights_file"],
            "sha256": CONTRACT["weights_sha256"],
        },
        "dataset": {
            "repo_id": "ai-forever/ru-scibench-grnti-classification",
            "revision": SOURCE_REVISION,
            "file": "test.jsonl",
            "sha256": SOURCE_SHA256,
            "examples": len(rows),
            "source_ids_in_arrays": True,
        },
        "preprocessing": {"max_length": MAX_LENGTH, "padding": "max_length", "truncation": True},
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            targets, logits.argmax(1), labels=list(range(28))
        ).tolist(),
        "per_class_f1": f1_score(
            targets, logits.argmax(1), labels=list(range(28)), average=None, zero_division=0
        ).tolist(),
        "arrays": {"path": arrays.name, "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest()},
        "source_commit": source_commit,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "packages": {
            name: version(name) for name in ["torch", "transformers", "numpy", "scikit-learn"]
        },
        "device": args.device,
        "seconds": time.monotonic() - started,
        "scope": (
            "Current pinned source test split. Original training dataset revision and "
            "membership were not recorded; held-out independence is not "
            "re-established by this replay."
        ),
    }
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
