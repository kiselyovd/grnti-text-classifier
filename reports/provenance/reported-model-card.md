---
language: ru
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: FacebookAI/xlm-roberta-base
tags:
- xlm-roberta
- russian-nlp
- grnti
- multiclass
- text-classification
datasets:
- ai-forever/ru-scibench-grnti-classification
widget:
- text: В работе исследовано влияние сильного магнитного поля на спектр электронных
    состояний в полупроводниковых квантовых точках. Методами численного моделирования
    показано, что энергетические уровни расщепляются и сдвигаются с ростом индукции
    поля, что открывает возможности для управления оптическими свойствами наноструктур.
model-index:
- name: kiselyovd/grnti-text-classifier
  results:
  - task:
      type: text-classification
      name: Top-level GRNTI classification
    dataset:
      name: ru-scibench-grnti-classification
      type: ai-forever/ru-scibench-grnti-classification
    metrics:
    - type: accuracy
      value: 0.7237
      name: Top-1 accuracy
    - type: accuracy
      value: 0.9675
      name: Top-5 accuracy
    - type: f1
      value: 0.723
      name: Macro F1
    - type: f1
      value: 0.723
      name: Weighted F1
---

# Russian Scientific-Text Classification

A research classifier for Russian scientific text over the **28 top-level GRNTI sections** (State Rubricator of Scientific and Technical Information). The main model is **XLM-RoBERTa-base** fine-tuned on Russian scientific abstracts; a single-language **ruBERT-base-cased** baseline is reported alongside it for comparison. The `id2label` map embeds the real GRNTI codes and Russian section names, so the inference widget returns human-readable predictions such as `290000: Физика`.

The model takes a Russian abstract or title and returns a probability distribution over the 28 sections. It is intended for cataloguing and triage of Russian scientific text, not for high-stakes or non-Russian inputs.

[Code & documentation](https://github.com/kiselyovd/grnti-text-classifier) · [Published test results](#metrics)

## Metrics

Test split: n = 2772 abstracts, 28 GRNTI sections, balanced.

| Model | Top-1 accuracy | Top-5 accuracy | Macro F1 | Weighted F1 |
|-------|---------------:|---------------:|---------:|------------:|
| **FacebookAI/xlm-roberta-base (main)** | **72.4%** | **96.8%** | **72.3%** | **72.3%** |
| DeepPavlov/rubert-base-cased (baseline) | 72.9% | 95.9% | 72.8% | 72.8% |

ruBERT reports slightly higher top-1 accuracy (+0.5 pp), while XLM-RoBERTa reports slightly higher top-5 accuracy (+0.9 pp). These are different operating choices: automatic assignment versus a candidate list for human review. The comparison alone does not identify why the difference occurs. The per-class metrics below are computed by re-running the published main model on the held-out test split; the resulting macro-F1 (0.723) matches the reported value exactly.

## Visualizations

**Per-class top-1 F1 across the 28 GRNTI sections.** Computed from real predictions of this model on the test split, sorted descending, with the macro-F1 reference line. Performance is strongest on well-separated domains (sport, food industry, literature) and weakest where section boundaries overlap (mechanical engineering, agriculture).

![Per-class top-1 F1](plots/per_class_f1.png)

**Confusion matrix (28 x 28).** Row-normalised over the test split. The diagonal and off-diagonal cells describe classification outcomes and confusion between sections. A confusion matrix does not establish probability calibration.

![Confusion matrix](plots/confusion_matrix.png)

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="kiselyovd/grnti-text-classifier", top_k=5)
clf("Исследование квантовой электродинамики в кристаллах.")
```

Each returned label is formatted as `<GRNTI code>: <section name>` (for example `290000: Физика`).

## The 28 GRNTI sections

The label space is the 28 top-level GRNTI sections present in the dataset, spanning the humanities, social sciences, natural sciences, and engineering: Философия; История. Исторические науки; Социология; Экономика. Экономические науки; Государство и право. Юридические науки; Политика. Политические науки; Культура. Культурология; Народное образование. Педагогика; Психология; Языкознание; Литература. Литературоведение. Устное народное творчество; Искусство; Математика; Физика; Химия; Биология; Геология; Энергетика; Автоматика. Вычислительная техника; Горное дело; Машиностроение; Пищевая промышленность; Строительство. Архитектура; Сельское и лесное хозяйство (codes 680000 and 683500); Транспорт; Медицина и здравоохранение; Физическая культура и спорт.

## Intended use and limitations

This model is trained for Russian-language top-level GRNTI section classification. It is not evaluated outside Russian scientific text and should not be used for generic multilingual classification. Outputs are probabilistic and subject to training-data biases; do not rely on this model for high-stakes decisions.

## Training

- Dataset: `ai-forever/ru-scibench-grnti-classification` (MIT, 28 476 train + 2 772 test).
- Base model: `FacebookAI/xlm-roberta-base`; baseline `DeepPavlov/rubert-base-cased`.
- Tokenizer: `xlm-roberta-base`, `max_length=256` (median sequence ~120 tokens).
- Precision: bf16-mixed on CUDA. Optimizer: AdamW with linear warmup and decay.
- Hyperparameters tuned with an Optuna sweep (val macro-F1), then a final training run with the best parameters (`lr=3.1e-5, weight_decay=0.012, warmup_ratio=0.147`, val macro-F1 = 73.1%).

## Source and license

- Source code, training pipeline, and documentation: https://github.com/kiselyovd/grnti-text-classifier
- License: MIT.
