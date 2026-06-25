"""Export trained model to HuggingFace-native format (safetensors + config.json).

Run BEFORE publish_to_hf.py so the HF repo gets proper pipeline pills / Inference
Providers instead of just a raw Lightning .ckpt.

Usage:
    python scripts/export_hf_native.py \\
        --checkpoint artifacts/checkpoints/best.ckpt \\
        --out artifacts/hf_export \\
        --base-model <HF_BASE_MODEL_ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_label_maps(
    encoder_path: Path,
) -> tuple[dict[str, str], dict[str, int]] | None:
    """Build id2label/label2id from the repo's label encoder, if present.

    Returns string-keyed maps (id2label keyed "0".."N-1", label2id its inverse)
    using human-readable "<code>: <section name>" labels, or None if the encoder
    file is missing.
    """
    if not encoder_path.exists():
        return None

    from grnti_text_classifier.data.grnti import LabelEncoder, _code_to_text

    enc = LabelEncoder.from_json_dict(json.loads(encoder_path.read_text(encoding="utf-8")))
    id2label = {
        str(i): f"{enc.idx_to_code[i]}: {_code_to_text(enc.idx_to_code[i])}"
        for i in range(enc.num_classes)
    }
    label2id = {label: int(i) for i, label in id2label.items()}
    return id2label, label2id


def main() -> None:
    p = argparse.ArgumentParser(description="Export NLP model to HF-native format.")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    p.add_argument("--out", default="artifacts/hf_export")
    p.add_argument(
        "--base-model",
        default=None,
        help="HF base model ID to copy tokenizer from (e.g. bert-base-uncased)",
    )
    p.add_argument(
        "--label-encoder",
        default="data/processed/label_encoder.json",
        help="Path to label_encoder.json used to inject id2label/label2id.",
    )
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from grnti_text_classifier.inference.predict import load_model

    lit = load_model(args.checkpoint)
    backbone = lit.model

    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit(
            "Backbone is not transformers-compatible; cannot export natively. "
            "Wrap your model in a transformers PreTrainedModel subclass first."
        )

    # Inject human-readable GRNTI labels so the Inference widget shows real
    # section names instead of LABEL_0..LABEL_N.
    maps = _load_label_maps(Path(args.label_encoder))
    if maps is not None:
        id2label, label2id = maps
        backbone.config.id2label = id2label
        backbone.config.label2id = label2id
        print(f"Injected id2label/label2id ({len(id2label)} classes) into config")
    else:
        print(
            f"WARNING: label encoder not found at {args.label_encoder}; "
            "config will keep default LABEL_* names"
        )

    backbone.save_pretrained(out)
    print(f"Saved model weights + config to {out}")

    if args.base_model:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(out)
        print(f"Copied tokenizer from {args.base_model} to {out}")

    print(f"HF-native export complete: {out}")


if __name__ == "__main__":
    main()
