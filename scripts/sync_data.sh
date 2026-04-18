#!/usr/bin/env bash
# Fetch ai-forever/ru-scibench-grnti-classification (train.jsonl + test.jsonl)
# into data/raw/. Idempotent: skips if both files already present.
set -euo pipefail

ROOT="${GRNTI_REPO_ROOT:-$(pwd)}"
RAW_DIR="${ROOT}/data/raw"
TRAIN="${RAW_DIR}/train.jsonl"
TEST="${RAW_DIR}/test.jsonl"
mkdir -p "${RAW_DIR}"

if [[ -s "${TRAIN}" && -s "${TEST}" ]]; then
  echo "[sync_data] both files present, skipping."
  exit 0
fi

uv run python - <<'PY'
import os, pathlib
from huggingface_hub import snapshot_download
root = pathlib.Path(os.environ.get("GRNTI_REPO_ROOT") or ".").resolve()
snapshot_download(
    repo_id="ai-forever/ru-scibench-grnti-classification",
    repo_type="dataset",
    local_dir=str(root / "data" / "raw"),
    allow_patterns=["train.jsonl", "test.jsonl"],
)
print(f"[sync_data] downloaded to {root / 'data' / 'raw'}")
PY
