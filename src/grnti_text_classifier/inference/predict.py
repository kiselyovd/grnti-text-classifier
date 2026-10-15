"""Run the pinned native GRNTI model on a title or abstract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .native import NativeClassifier


def load_model(model_dir: str | Path | None = None) -> NativeClassifier:
    """Load reviewed native safetensors; local directories must contain the same artifact."""
    return NativeClassifier(model_dir=model_dir)


def predict(model: NativeClassifier, input_path: str | Path) -> dict[str, Any]:
    """Classify UTF-8 file contents with the production tokenizer and label mapping."""
    return model.classify(Path(input_path).read_text(encoding="utf-8"))


def main() -> None:
    """Use --text for direct input or --input for a UTF-8 file."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    model = load_model(args.model_dir)
    result = model.classify(args.text) if args.text is not None else predict(model, args.input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
