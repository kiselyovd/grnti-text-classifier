# Artifact and evaluation contract

The native main checkpoint is pinned in [artifact-manifest.json](https://github.com/kiselyovd/grnti-text-classifier/blob/main/artifact-manifest.json) and the shared inference contract. The CLI and [live Space](https://huggingface.co/spaces/kiselyovd/grnti-text-classifier-demo) verify its safetensors checksum, disable remote model code, use 256 tokens and preserve all 28 exact checkpoint labels. The tokenizer's capacity of 512 is not the training/inference recipe.

Codes are strings as stored: five- and six-digit forms coexist. Codes `680000` and `683500` have the same stored title but remain separate classes.

The original metrics are retained as a hash-bound snapshot in `reports/reported-native.json` with its immutable source card. Missing reports now stop publication; no fabricated zero scores are emitted. The numerical replay is a separate artifact with source row IDs and logits. Original training dataset revision and membership were not recorded, so replay does not independently establish test-set separation from historical training.

## Reproduce the native replay

Use the project-declared environment (`uv sync --frozen --all-groups`). Obtain only `test.jsonl` from dataset `ai-forever/ru-scibench-grnti-classification` at revision `a984f8a65b2e9fae7a347bf7761b0eaad79c73a1`. Its SHA256 is `f8b6a7c7050f4f8f76de8ab301b7b771e47b089900c06618d103369c5c10afbe`.

```bash
uv run --frozen --all-groups python scripts/replay_native.py --test-jsonl data/raw/test.jsonl
python scripts/artifact_manifest.py --check
uv run --frozen --all-groups python scripts/plot_replay.py
```

The replay checks source bytes and unique IDs, maps exact source codes to checkpoint indices, applies the native 256-token transform and saves logits/targets/IDs plus aggregate and per-class metrics. Visitor examples in the Space are illustrations rather than labelled evaluation inputs. Model scores are uncalibrated softmax values.

## Component terms

Project code and the author-declared model license are MIT. The source dataset card declares MIT; XLM-RoBERTa-base declares MIT. These are separate component declarations, not a new license over upstream materials. Dependencies retain their individual licenses. Cite the software using `CITATION.cff` and cite the original corpus separately.
