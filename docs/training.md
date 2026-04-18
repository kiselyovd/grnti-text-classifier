# Training

End-to-end training is orchestrated by `scripts/train_all.py`, which runs the baseline (ruBERT) and main model (XLM-RoBERTa) in sequence, produces evaluation metrics, a confusion matrix, and a `metrics_summary.json`. GPU is recommended; the main model will fall back to CPU but training time increases ~10×.

## Prerequisites

```bash
uv sync --all-groups
bash scripts/sync_data.sh
uv run python -m grnti_text_classifier.data.prepare --raw data/raw --out data/processed
```

`sync_data.sh` calls `snapshot_download` for `ai-forever/ru-scibench-grnti-classification`. `prepare.py` writes `train.parquet`, `val.parquet`, `test.parquet`, and `label_encoder.json` under `data/processed/`.

## Commands

Full pipeline (baseline + main + evaluation):

```bash
uv run python scripts/train_all.py
```

Individual stages:

```bash
# Main model only (XLM-RoBERTa-base)
uv run python -m grnti_text_classifier.training.train model=main

# Baseline only (ruBERT-base-cased)
uv run python -m grnti_text_classifier.training.train model=baseline

# Optuna sweep (10 trials, 3 epochs each)
uv run python -m grnti_text_classifier.training.sweep model=main

# Evaluation (top-k accuracy, F1, confusion matrix)
uv run python -m grnti_text_classifier.evaluation.evaluate
```

## Hydra configuration layout

```
configs/
├── train.yaml          # top-level: data paths, trainer, logger, seed
└── model/
    ├── main.yaml       # XLM-RoBERTa-base: lr, batch_size, max_length, warmup_ratio
    └── baseline.yaml   # ruBERT-base-cased: same schema, different defaults
```

Override any parameter from the CLI without editing YAML:

```bash
uv run python -m grnti_text_classifier.training.train model=main model.lr=3e-5 trainer.max_epochs=5
```

## Optuna sweep

`training.sweep` runs **10 trials × 3 epochs** each with the TPE sampler (seed=42). Search space:

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| `lr` | log-uniform | 1e-5 – 5e-5 |
| `weight_decay` | log-uniform | 1e-4 – 1e-1 |
| `warmup_ratio` | uniform | 0.0 – 0.15 |

The best trial is automatically written to `artifacts/best_params.json` and used as the starting point for the final full-epoch training run.

## RTX 3080 notes

Reference hardware for the v0.1.0 run: RTX 3080 (10 GB VRAM), 32 GB RAM, Ubuntu 22.04.

| Setting | Value |
|---------|-------|
| `precision` | `bf16-mixed` |
| `batch_size` | 16 |
| `max_length` | 256 |
| `gradient_clip_val` | 1.0 |
| Estimated wall time (XLM-R, 5 epochs) | ~45–60 min |
| Estimated wall time (ruBERT, 5 epochs) | ~30–40 min |

`bf16-mixed` requires Ampere or newer (RTX 30xx / A-series). On older CUDA GPUs set `precision=16-mixed`; on CPU omit the flag entirely.

## Outputs

- `artifacts/main/hf/` — XLM-RoBERTa `save_pretrained` snapshot (model + tokenizer).
- `artifacts/baseline/hf/` — ruBERT `save_pretrained` snapshot.
- `artifacts/main/logs/version_0/metrics.csv` — CSVLogger epoch-level metrics (loss, top1_acc, top5_acc, f1_macro).
- `reports/metrics.json` — per-class precision/recall/F1 on test set.
- `reports/metrics_summary.json` — top-1/top-5 accuracy + macro/weighted F1 summary for both models.
- `reports/confusion_matrix.png` — 28×28 normalised confusion matrix for the main model.

## HF mirror runbook

Once training completes and artefacts are validated, publish to the HF Hub:

```bash
# Ensure HUGGING_FACE_HUB_TOKEN is set (or run `huggingface-cli login`)
uv run python scripts/publish_to_hf.py \
  --main-dir artifacts/main/hf \
  --baseline-dir artifacts/baseline/hf \
  --repo-id kiselyovd/grnti-text-classifier
```

The script calls `push_to_hub` for the main model and attaches the model card generated in Task 17. The baseline is pushed to a separate branch `baseline` within the same repo.
