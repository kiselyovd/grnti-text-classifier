# grnti-text-classifier

[![CI](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml/badge.svg)](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/kiselyovd/grnti-text-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/kiselyovd/grnti-text-classifier)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![HF Hub](https://img.shields.io/badge/HF_Hub-kiselyovd/grnti--text--classifier-yellow)](https://huggingface.co/kiselyovd/grnti-text-classifier)

Production-grade Russian scientific-text classifier over 28 top-level GRNTI (State Rubricator of Scientific and Technical Information) classes. Main model **XLM-RoBERTa-base** (multilingual transformer, fine-tuned on Russian abstracts); baseline **ruBERT-base-cased** (single-language BERT). Both are Hydra-configured, Optuna-tuned, evaluated with top-1 / top-5 accuracy and macro / weighted F1, and served by FastAPI as `/classify`.

> **Part of the [kiselyovd ML portfolio](https://github.com/kiselyovd#ml-portfolio)** — production-grade ML projects sharing one [cookiecutter template](https://github.com/kiselyovd/ml-project-template).

📖 [English docs](https://kiselyovd.github.io/grnti-text-classifier/) • 🇷🇺 [Русский README](README.ru.md) • 🤗 [HF Hub model](https://huggingface.co/kiselyovd/grnti-text-classifier)

## Dataset

[ai-forever/ru-scibench-grnti-classification](https://huggingface.co/datasets/ai-forever/ru-scibench-grnti-classification) — Russian scientific abstracts labelled with 28 GRNTI top-level sections. Split statistics:

| Split | Rows | Classes |
|-------|-----:|--------:|
| Train | 28 476 | 28 (balanced) |
| Test  | 2 772 | 28 (balanced) |

Median sequence length ~120 tokens under the XLM-RoBERTa tokenizer (`xlm-roberta-base`, `max_length=256`). Fetched by `scripts/sync_data.sh` via HF `snapshot_download`.

## Results

Filled in from `reports/metrics_summary.json` once the v0.1.0 run completes (Task 18).

| Model | Top-1 accuracy | Top-5 accuracy | Macro F1 | Weighted F1 |
|-------|---------------:|---------------:|---------:|------------:|
| XLM-RoBERTa-base (main) | — | — | — | — |
| ruBERT-base-cased (baseline) | — | — | — | — |

Test set n = 2 772.

## Quick Start

```bash
uv sync --all-groups
bash scripts/sync_data.sh
uv run python -m grnti_text_classifier.data.prepare --raw data/raw --out data/processed
uv run python scripts/train_all.py
```

## Serving

```bash
uvicorn grnti_text_classifier.serving.main:app --reload
```

Classify a text:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"Исследование квантовой электродинамики в кристаллах."}'
```

See [docs/serving.md](docs/serving.md) for full endpoint contracts, all request/response schemas, and environment variable reference.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GRNTI_MAIN_DIR` | Directory containing the `save_pretrained` snapshot for XLM-RoBERTa main model. |
| `GRNTI_BASELINE_DIR` | Directory containing the `save_pretrained` snapshot for ruBERT baseline. |
| `GRNTI_LABEL_ENCODER` | Path to `label_encoder.json` mapping int indices to GRNTI class codes. |
| `GRNTI_MODEL_VERSION` | Reported in `/health` response and classification output (e.g. `v0.1.0`). |

## Docs

Full documentation (architecture, training runbook, serving guide, API reference) is published at **[https://kiselyovd.github.io/grnti-text-classifier/](https://kiselyovd.github.io/grnti-text-classifier/)**.

## License

MIT — see [LICENSE](LICENSE).
