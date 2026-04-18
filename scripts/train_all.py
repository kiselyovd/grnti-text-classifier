"""End-to-end training + scoring orchestrator.

Runs:
  1. Optuna 10-trial sweep on main (XLM-R) for best lr/wd/warmup_ratio.
  2. Main training with best params (5 epochs, patience 2).
  3. Baseline training (ruBERT) with fixed hyperparams.
  4. Test scoring for both models; save metrics JSON.
  5. Confusion matrix for main model with human-readable GRNTI labels.
  6. Flat summary JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from grnti_text_classifier.models.factory import build_baseline, build_main
from grnti_text_classifier.training.optuna_sweep import run_sweep
from grnti_text_classifier.training.train import train_one
from grnti_text_classifier.evaluation.confusion import save_confusion_matrix
from grnti_text_classifier.evaluation.metrics import compute_metrics
from grnti_text_classifier.evaluation.report import build_summary


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _score_hf_dir(
    hf_dir: Path,
    test_parquet: Path,
    num_classes: int,
    batch_size: int = 32,
    max_length: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, y_true) from scoring *hf_dir* on *test_parquet*.

    Parameters
    ----------
    hf_dir:
        HuggingFace model directory produced by ``train_one``.
    test_parquet:
        Path to the test split parquet file.
    num_classes:
        Number of output classes (used for shape assertions).
    batch_size:
        Inference batch size.
    max_length:
        Tokeniser max sequence length.

    Returns
    -------
    tuple of (logits, y_true) as float32 / int64 NumPy arrays.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(hf_dir)
    model = AutoModelForSequenceClassification.from_pretrained(hf_dir)
    model.to(device)
    model.train(False)

    df = pd.read_parquet(test_parquet)
    texts: list[str] = df["text"].tolist()
    y_true: np.ndarray = df["label_idx"].to_numpy()

    all_logits: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tok(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc)
        all_logits.append(out.logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    return logits, y_true


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the full training + evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train + evaluate GRNTI text classifiers end-to-end."
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                        help="Pre-processed data directory (default: data/processed)")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"),
                        help="Root artefacts directory (default: artifacts)")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"),
                        help="Reports output directory (default: reports)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Optuna trial count (default: 10)")
    parser.add_argument("--max-epochs", type=int, default=5,
                        help="Max training epochs per run (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip Optuna sweep; use spec §12 defaults instead")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline (ruBERT) training + scoring")
    args = parser.parse_args(argv)

    processed_dir: Path = args.processed_dir
    artifacts_dir: Path = args.artifacts_dir
    reports_dir: Path = args.reports_dir

    # Ensure output directories exist
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load label encoder
    label_enc = json.loads((processed_dir / "label_encoder.json").read_text(encoding="utf-8"))
    num_classes: int = int(label_enc["num_classes"])
    idx_to_text: dict[int, str] = {int(k): v for k, v in label_enc["idx_to_text"].items()}
    print(f"[train_all] num_classes={num_classes}")

    # ------------------------------------------------------------------
    # Step 1: Optuna sweep (optional)
    # ------------------------------------------------------------------
    if not args.skip_sweep:
        print("[train_all] Step 1/6 — Optuna sweep …")
        sweep_out_dir = artifacts_dir / "sweep"
        result = run_sweep(
            processed_dir,
            sweep_out_dir,
            model_builder=build_main,
            model_name_for_tokenizer="FacebookAI/xlm-roberta-base",
            n_trials=args.n_trials,
            trial_epochs=3,
            batch_size=args.batch_size,
        )
        (reports_dir / "sweep_best.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        best_params = result["best_params"]
        print(f"[train_all] Best sweep params: {best_params}  val_f1={result['best_value']:.4f}")
    else:
        print("[train_all] Step 1/6 — sweep skipped; using spec §12 defaults.")
        best_params = {"lr": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.1}

    # ------------------------------------------------------------------
    # Step 2: Main training (XLM-R)
    # ------------------------------------------------------------------
    print("[train_all] Step 2/6 — training main model (XLM-R) …")
    main_hf: Path = train_one(
        build_main,
        "FacebookAI/xlm-roberta-base",
        processed_dir,
        artifacts_dir / "main",
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        patience=2,
    )
    print(f"[train_all] Main model saved to: {main_hf}")

    # ------------------------------------------------------------------
    # Step 3: Baseline training (ruBERT)
    # ------------------------------------------------------------------
    baseline_hf: Path | None = None
    if not args.skip_baseline:
        print("[train_all] Step 3/6 — training baseline model (ruBERT) …")
        baseline_hf = train_one(
            build_baseline,
            "DeepPavlov/rubert-base-cased",
            processed_dir,
            artifacts_dir / "baseline",
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )
        print(f"[train_all] Baseline model saved to: {baseline_hf}")
    else:
        print("[train_all] Step 3/6 — baseline skipped.")

    # ------------------------------------------------------------------
    # Step 4: Test scoring
    # ------------------------------------------------------------------
    print("[train_all] Step 4/6 — scoring on test split …")
    test_parquet = processed_dir / "test.parquet"

    main_logits, y_true = _score_hf_dir(main_hf, test_parquet, num_classes)
    main_preds = main_logits.argmax(-1)
    main_metrics = compute_metrics(y_true, main_logits, num_classes)
    (reports_dir / "metrics.json").write_text(
        json.dumps(main_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train_all] Main  — macro_f1={main_metrics['macro_f1']:.4f}  "
          f"top1={main_metrics['top1_accuracy']:.4f}")

    baseline_metrics: dict | None = None
    if baseline_hf is not None:
        baseline_logits, _ = _score_hf_dir(baseline_hf, test_parquet, num_classes)
        baseline_metrics = compute_metrics(y_true, baseline_logits, num_classes)
        (reports_dir / "metrics_baseline.json").write_text(
            json.dumps(baseline_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[train_all] Baseline — macro_f1={baseline_metrics['macro_f1']:.4f}  "
              f"top1={baseline_metrics['top1_accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Confusion matrix (main model)
    # ------------------------------------------------------------------
    print("[train_all] Step 5/6 — saving confusion matrix …")
    labels = [idx_to_text[i] for i in range(num_classes)]
    cm_path = reports_dir / "confusion_matrix.png"
    save_confusion_matrix(y_true, main_preds, labels, cm_path)
    print(f"[train_all] Confusion matrix saved to: {cm_path}")

    # ------------------------------------------------------------------
    # Step 6: Summary JSON
    # ------------------------------------------------------------------
    print("[train_all] Step 6/6 — building summary …")
    if baseline_metrics is not None:
        summary = build_summary(
            main_metrics, baseline_metrics, out_path=reports_dir / "metrics_summary.json"
        )
        print(f"[train_all] Summary: main_macro_f1={summary['main_macro_f1']}  "
              f"baseline_macro_f1={summary['baseline_macro_f1']}")
    else:
        print("[train_all] Baseline not run — skipping comparative summary.")

    print("\n[train_all] All done.")
    print(f"  Artefacts : {artifacts_dir.resolve()}")
    print(f"  Reports   : {reports_dir.resolve()}")
    print(f"  Metrics   : {(reports_dir / 'metrics.json').resolve()}")
    if cm_path.exists():
        print(f"  Confusion : {cm_path.resolve()}")


if __name__ == "__main__":
    main()
