# grnti-text-classifier

[![CI](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml/badge.svg)](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/kiselyovd/grnti-text-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/kiselyovd/grnti-text-classifier)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![HF Hub](https://img.shields.io/badge/HF_Hub-kiselyovd/grnti--text--classifier-yellow)](https://huggingface.co/kiselyovd/grnti-text-classifier)

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

Заполняется из `reports/metrics_summary.json` после завершения запуска v0.1.0 (Task 18).

| Модель | Top-1 accuracy | Top-5 accuracy | Macro F1 | Weighted F1 |
|--------|---------------:|---------------:|---------:|------------:|
| XLM-RoBERTa-base (основная) | — | — | — | — |
| ruBERT-base-cased (baseline) | — | — | — | — |

Test set n = 2 772.

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
