# grnti-text-classifier

[![CI](https://img.shields.io/github/actions/workflow/status/kiselyovd/grnti-text-classifier/test.yml?branch=main&style=for-the-badge&label=CI&logo=github)](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://kiselyovd.github.io/grnti-text-classifier/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kiselyovd/grnti-text-classifier/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Hub](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/kiselyovd/grnti-text-classifier)

Промышленный классификатор русских научных текстов по 28 разделам ГРНТИ (Государственный рубрикатор научно-технической информации). Основная модель — **XLM-RoBERTa-base** (многоязычный трансформер, дообученный на русских аннотациях); baseline — **ruBERT-base-cased** (одноязычный BERT). Обе модели конфигурируются через Hydra, настраиваются через Optuna, оцениваются по top-1 / top-5 accuracy и macro / weighted F1, и обслуживаются через FastAPI (`/classify`).

> **Часть [ML-портфолио kiselyovd](https://github.com/kiselyovd#ml-portfolio)** — промышленные ML-проекты, основанные на одном [cookiecutter-шаблоне](https://github.com/kiselyovd/ml-project-template).

📖 [Документация (EN)](https://kiselyovd.github.io/grnti-text-classifier/) • 🇬🇧 [English README](README.md) • 🤗 [Модель на HF Hub](https://huggingface.co/kiselyovd/grnti-text-classifier)

## Датасет

[ai-forever/ru-scibench-grnti-classification](https://huggingface.co/datasets/ai-forever/ru-scibench-grnti-classification) — русские научные аннотации, размеченные по 28 верхнеуровневым разделам ГРНТИ. Статистика разбиения:

| Сплит | Строк | Классов |
|-------|------:|--------:|
| Train | 28 476 | 28 (сбалансированные) |
| Test  | 2 772 | 28 (сбалансированные) |

Медианная длина последовательности ~120 токенов в токенизаторе XLM-RoBERTa (`xlm-roberta-base`, `max_length=256`). Загружается через `scripts/sync_data.sh` посредством HF `snapshot_download`.

## Результаты

Тестовая выборка: 2 772 аннотации по 28 рубрикам ГРНТИ.

| Модель | Top-1 accuracy | Top-5 accuracy | Macro F1 | Weighted F1 |
|--------|---------------:|---------------:|---------:|------------:|
| **XLM-RoBERTa-base (основная)** | **72,4%** | **96,8%** | **72,3%** | **72,3%** |
| ruBERT-base-cased (baseline) | 72,9% | 95,9% | 72,8% | 72,8% |

Лучший триал Optuna (20 попыток по val macro-F1): `lr=3,1e-5, weight_decay=0,012, warmup_ratio=0,147` → val macro-F1 = 73,1%.

Baseline чуть впереди по top-1, основная модель лучше на +0,9 п.п. по top-5 — многоязычный pre-training XLM-R даёт более точный rerank top-k, а моноязычная ruBERT слегка выигрывает на argmax. Обе модели опубликованы на HF Hub с единым model card.

## Быстрый старт

```bash
uv sync --all-groups
bash scripts/sync_data.sh
uv run python -m grnti_text_classifier.data.prepare --raw data/raw --out data/processed
uv run python scripts/train_all.py
```

## Сервинг

```bash
uvicorn grnti_text_classifier.serving.main:app --reload
```

Классифицировать текст:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"Исследование квантовой электродинамики в кристаллах."}'
```

Полные контракты эндпоинтов, схемы запросов/ответов и описание переменных окружения — в [docs/serving.md](docs/serving.md).

## Переменные окружения

| Переменная | Назначение |
|------------|-----------|
| `GRNTI_MAIN_DIR` | Директория со снапшотом `save_pretrained` основной модели XLM-RoBERTa. |
| `GRNTI_BASELINE_DIR` | Директория со снапшотом `save_pretrained` baseline ruBERT. |
| `GRNTI_LABEL_ENCODER` | Путь к `label_encoder.json` — маппинг целочисленных индексов на коды классов ГРНТИ. |
| `GRNTI_MODEL_VERSION` | Возвращается в ответе `/health` и в теле классификации (например, `v0.1.0`). |

## Документация

Полная документация (архитектура, руководство по обучению, сервинг, API reference) опубликована на **[https://kiselyovd.github.io/grnti-text-classifier/](https://kiselyovd.github.io/grnti-text-classifier/)**.

## Лицензия

MIT — см. [LICENSE](LICENSE).
