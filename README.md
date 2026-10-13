# Scientific-Text Classification

Classify scientific abstracts into 28 GRNTI subject codes. XLM-RoBERTa is the main model; ruBERT provides a second operating point.

[![CI](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml/badge.svg)](https://github.com/kiselyovd/grnti-text-classifier/actions/workflows/test.yml) [![Docs](https://img.shields.io/badge/docs-MkDocs-blue)](https://kiselyovd.github.io/grnti-text-classifier/) [![Model](https://img.shields.io/badge/Hugging_Face-model-yellow)](https://huggingface.co/kiselyovd/grnti-text-classifier) [![MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)

[![Interactive subject catalogue and source-abstract explorer](docs/images/showcase.png)](https://kiselyovd.github.io/grnti-text-classifier/showcase/)

**[Browse the subjects and abstracts](https://kiselyovd.github.io/grnti-text-classifier/showcase/)** — search the 28-code catalogue, inspect the 20 committed source records, and reveal their corpus labels. Original code strings and text are preserved, including separate codes with matching names.

## Published operating points

Reported test set: 2,772 abstracts across 28 sections.

| Model | Top-1 accuracy | Top-5 accuracy | Macro F1 |
|---|---|---|---|
| XLM-RoBERTa | 72.4% | 96.8% | 72.3% |
| ruBERT | 72.9% | 95.9% | 72.8% |

One automatic assignment and a candidate list for review are different use cases. These figures alone do not establish why the models differ. [Published model card](https://huggingface.co/kiselyovd/grnti-text-classifier).

## Use the model

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="kiselyovd/grnti-text-classifier", top_k=5)
print(classifier("Исследование квантовой электродинамики в кристаллах."))
```

Outputs use the model's original code/name label map. [Source abstracts](data/sample/sample.jsonl) · [Training and API setup](docs/project-guide.md#quick-start) · [Русский README](README.ru.md)

## Documentation

- [Corpus explorer](https://kiselyovd.github.io/grnti-text-classifier/showcase/)
- [Project guide](docs/project-guide.md): dataset, recipes, evaluation, and serving variables.
- [Architecture](docs/architecture.md) · [Training](docs/training.md) · [API serving](docs/serving.md)

## Limitations and terms

The classifier targets Russian scientific-text subject assignment. The smoke corpus preserves original multilingual source text and existing labels; the explorer presents those records rather than producing new predictions. Numerical per-class report arrays are absent from the tracked repository, so the published plots remain source-labelled figures.

Code is [MIT](LICENSE). Dataset and pretrained-model terms apply separately; see the [model card](https://huggingface.co/kiselyovd/grnti-text-classifier).
