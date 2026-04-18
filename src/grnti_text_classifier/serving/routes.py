"""GRNTI classifier routes — /health, /classify, /labels."""
from __future__ import annotations
import json, os, uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Query
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .schemas import ClassificationResponse, LabelEntry, LabelProb, TextPayload

router = APIRouter()

_ENV_MAIN = "GRNTI_MAIN_DIR"
_ENV_BASE = "GRNTI_BASELINE_DIR"
_ENV_ENC = "GRNTI_LABEL_ENCODER"
_ENV_VER = "GRNTI_MODEL_VERSION"


def _env_path(var: str) -> Path:
    p = os.environ.get(var)
    if not p:
        raise HTTPException(status_code=503,
                            detail=f"Required env var {var} is not set.")
    path = Path(p)
    if not path.exists():
        raise HTTPException(status_code=503,
                            detail=f"{var} path does not exist: {path}")
    return path


def _load_labels() -> dict[int, str]:
    enc_path = _env_path(_ENV_ENC)
    data = json.loads(enc_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["idx_to_text"].items()}


@lru_cache(maxsize=2)
def _load_model(tag: Literal["main", "baseline"]):
    var = _ENV_MAIN if tag == "main" else _ENV_BASE
    model_dir = _env_path(var)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    if torch.cuda.is_available():
        model = model.cuda()
    model.train(False)
    return tok, model, str(model_dir)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/labels")
def labels() -> list[LabelEntry]:
    idx_to_text = _load_labels()
    return [LabelEntry(label=i, label_text=idx_to_text[i]) for i in sorted(idx_to_text)]


@router.post("/classify", response_model=ClassificationResponse)
def classify(
    payload: TextPayload,
    model: Literal["main", "baseline"] = Query(default="main"),
) -> ClassificationResponse:
    tok, mdl, model_dir = _load_model(model)
    idx_to_text = _load_labels()

    enc = tok(payload.text, return_tensors="pt",
              padding="max_length", truncation=True, max_length=payload.max_length)
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    input_length_tokens = int(enc["attention_mask"].sum().item())
    truncated = input_length_tokens >= payload.max_length

    with torch.inference_mode():
        out = mdl(**enc)
    probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu().numpy()

    top5_idx = np.argsort(probs)[-5:][::-1].tolist()
    top5 = [LabelProb(label=i, label_text=idx_to_text.get(i, f"GRNTI-{i}"),
                      probability=float(probs[i])) for i in top5_idx]

    return ClassificationResponse(
        top1=top5[0],
        top5=top5,
        input_length_tokens=input_length_tokens,
        truncated=truncated,
        model_version=os.environ.get(_ENV_VER, "unknown"),
        model_name=Path(model_dir).name,
        request_id=str(uuid.uuid4()),
    )
