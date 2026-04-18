# Architecture

Two independent `save_pretrained` artefacts sharing the same `{top1_label, top1_prob, top5, truncated, input_length_tokens, request_id, model_version}` contract.

```mermaid
flowchart LR
    A["data/raw/*.jsonl<br/>HF snapshot_download"]:::external -->|prepare.py| B["data/processed/<br/>train/val/test.parquet<br/>+ label_encoder.json"]:::data
    B -->|GRNTIDataModule| C["XLM-R tokenizer<br/>padding=max_length<br/>truncation=True"]:::code
    C --> D["GRNTIClassifier<br/>AdamW + linear warmup+decay<br/>bf16-mixed on CUDA"]:::model
    D -->|train_one| E["artifacts/{main,baseline}/hf/<br/>save_pretrained"]:::artifact
    E -->|evaluate.py| F["reports/metrics.json<br/>+ confusion_matrix.png<br/>+ metrics_summary.json"]:::artifact
    E -.->|publish_to_hf.py| G["HF Hub<br/>kiselyovd/grnti-text-classifier"]:::external
    E -.->|"FastAPI /classify"| H["JSON<br/>top-1 + top-5 + probs"]:::serve

    classDef external fill:#E0F2F1,stroke:#00796B,color:#004D40
    classDef data fill:#B2DFDB,stroke:#00796B,color:#004D40
    classDef code fill:#80CBC4,stroke:#00695C,color:#004D40
    classDef model fill:#4DB6AC,stroke:#004D40,color:#fff
    classDef artifact fill:#26A69A,stroke:#004D40,color:#fff
    classDef serve fill:#00897B,stroke:#004D40,color:#fff
```

## Data flow

`scripts/sync_data.sh` calls `snapshot_download` from the HF Hub to pull `ai-forever/ru-scibench-grnti-classification` into `data/raw/` as JSONL shards. `grnti_text_classifier.data.prepare` reads those shards, fits a `LabelEncoder` over the 28 GRNTI top-level class codes, writes stratified `train/val/test.parquet` splits under `data/processed/`, and serialises the encoder to `data/processed/label_encoder.json`. The `GRNTIDataModule` wraps those Parquet files: it tokenises on-the-fly with the HF `AutoTokenizer` (`padding="max_length"`, `truncation=True`, `max_length=256`), returning `input_ids`, `attention_mask`, and `labels` tensors.

## Why XLM-RoBERTa-base over ruBERT

XLM-RoBERTa-base was pre-trained on 2.5 TB of CommonCrawl covering 100 languages including Russian. Its sub-word vocabulary (250 k sentencepiece tokens) achieves much lower out-of-vocabulary rates on Russian scientific terminology than the 120 k WordPiece vocabulary of ruBERT-base-cased. In practice this translates to longer effective context per 256-token window and better transfer on low-resource GRNTI sections. ruBERT-base-cased is retained as a single-language baseline: it is lighter, faster to fine-tune, and provides a meaningful reference point for any XLM-R gains.

## Inverse-frequency class weights

The ru-scibench-grnti-classification dataset is designed to be class-balanced, but minor residual imbalances remain across 28 sections. `train_one` passes `class_weights` (computed per-batch from inverse normalised class frequencies on the train split) into `CrossEntropyLoss`. This has negligible effect on top-1 accuracy but meaningfully protects macro F1: without weighting the rarest classes tend to be under-penalised, dragging the macro average below the weighted average.

## HF-native `save_pretrained`

Both models are serialised via the standard `PreTrainedModel.save_pretrained` / `tokenizer.save_pretrained` pattern into `artifacts/{main,baseline}/hf/`. This means any downstream consumer can load them with `AutoModelForSequenceClassification.from_pretrained(repo_id)` — zero extra config, no custom unpickling, and fully compatible with the HF Hub widget system for live inference demos on the model card.
