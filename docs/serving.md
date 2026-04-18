# Serving

The main and baseline models are exposed behind a single FastAPI app (`grnti_text_classifier.serving.main:app`). Both are loaded lazily from `artifacts/{main,baseline}/hf/` at first request and held in memory thereafter. The active model is selected via a query parameter on each call.

## Run

```bash
# local dev (auto-reload)
uvicorn grnti_text_classifier.serving.main:app --host 0.0.0.0 --port 8000 --reload

# production (4 workers)
uvicorn grnti_text_classifier.serving.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok", "version": <model_version>}`. |
| `GET` | `/labels` | Returns the full list of 28 GRNTI label codes and human-readable names loaded from `label_encoder.json`. |
| `POST` | `/classify` | Classify a single Russian text. Query param `model` selects `main` (default) or `baseline`. |

`/classify` returns HTTP 422 if `text` is empty or whitespace-only, and HTTP 503 if the HF snapshot directory is missing from disk.

## Request — `TextPayload`

Schema source: `grnti_text_classifier.serving.schemas.TextPayload`.

```python
class TextPayload(BaseModel):
    text: str   # Russian scientific text to classify (required, non-empty)
```

## Response schemas

### `LabelProb`

```python
class LabelProb(BaseModel):
    label: str    # GRNTI class code, e.g. "27" (Mathematics)
    name: str     # Human-readable section name
    prob: float   # Softmax probability for this class
```

### `LabelEntry`

```python
class LabelEntry(BaseModel):
    code: str   # GRNTI top-level code (2-digit string)
    name: str   # Section name in Russian
```

### `ClassificationResponse`

```python
class ClassificationResponse(BaseModel):
    top1_label: str          # GRNTI code of the most likely class
    top1_name: str           # Human-readable name of the top-1 class
    top1_prob: float         # Softmax probability of the top-1 class
    top5: list[LabelProb]    # Top-5 classes with probabilities
    truncated: bool          # True if input exceeded max_length and was truncated
    input_length_tokens: int # Token count before any truncation
    request_id: str          # 12-char UUID prefix for tracing
    model_name: str          # "xlm-roberta-base" or "rubert-base-cased"
    model_version: str       # e.g. "v0.1.0"
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GRNTI_MAIN_DIR` | `artifacts/main/hf` | Path to XLM-RoBERTa `save_pretrained` snapshot. |
| `GRNTI_BASELINE_DIR` | `artifacts/baseline/hf` | Path to ruBERT `save_pretrained` snapshot. |
| `GRNTI_LABEL_ENCODER` | `data/processed/label_encoder.json` | Path to label encoder JSON. |
| `GRNTI_MODEL_VERSION` | `v0.1.0` | Reported in `/health` and response body. |

## curl examples

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "version": "v0.1.0"}
```

### GET /labels

```bash
curl http://localhost:8000/labels
```

```json
[
  {"code": "01", "name": "Общенаучное и междисциплинарное знание"},
  {"code": "03", "name": "История. Исторические науки"},
  ...
]
```

### POST /classify — main model

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"Исследование квантовой электродинамики в кристаллах."}'
```

```json
{
  "top1_label": "29",
  "top1_name": "Физика",
  "top1_prob": 0.923,
  "top5": [
    {"label": "29", "name": "Физика", "prob": 0.923},
    {"label": "30", "name": "Химия", "prob": 0.031},
    {"label": "44", "name": "Энергетика", "prob": 0.018},
    {"label": "27", "name": "Математика", "prob": 0.012},
    {"label": "50", "name": "Автоматика", "prob": 0.007}
  ],
  "truncated": false,
  "input_length_tokens": 14,
  "request_id": "a1b2c3d4e5f6",
  "model_name": "xlm-roberta-base",
  "model_version": "v0.1.0"
}
```

### POST /classify — baseline model

```bash
curl -X POST "http://localhost:8000/classify?model=baseline" \
  -H "Content-Type: application/json" \
  -d '{"text":"Исследование квантовой электродинамики в кристаллах."}'
```

## Response field notes

| Field | Notes |
|-------|-------|
| `truncated` | `True` when `input_length_tokens > max_length` (256). The model still produces a prediction but context beyond 256 tokens was dropped. |
| `input_length_tokens` | Raw token count before truncation, useful for monitoring distribution shift at inference time. |
| `request_id` | First 12 characters of a UUID4 generated per request. Log this for end-to-end tracing. |
| `model_name` | Reflects the actual HF model identifier, not the alias (`main`/`baseline`). |
| `model_version` | Read from the `GRNTI_MODEL_VERSION` environment variable; matches the git tag of the published checkpoint. |
