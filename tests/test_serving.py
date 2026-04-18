"""E2E tests for the FastAPI /classify + /labels routes.

Uses a tiny 1-layer XLM-RoBERTa config built in-memory (no HF downloads beyond
the shared tokenizer cache) to keep test runtime <30s.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification

from grnti_text_classifier.serving.main import app


@pytest.fixture(scope="module", autouse=True)
def _setup_model(tmp_path_factory: pytest.TempPathFactory, monkeypatch_module: pytest.MonkeyPatch):
    tmp = tmp_path_factory.mktemp("grnti_serving")
    main_dir = tmp / "main"
    main_dir.mkdir()
    cfg = XLMRobertaConfig(
        vocab_size=250002,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=2,
        max_position_embeddings=256,
        type_vocab_size=1,
    )
    model = XLMRobertaForSequenceClassification(cfg)
    model.save_pretrained(main_dir)
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=True)
    tok.save_pretrained(main_dir)
    # baseline dir = same (simplification)
    baseline_dir = tmp / "baseline"
    baseline_dir.mkdir()
    model.save_pretrained(baseline_dir)
    tok.save_pretrained(baseline_dir)
    encoder = {
        "code_to_idx": {"0": 0, "10000": 1},
        "idx_to_code": {"0": 0, "1": 10000},
        "idx_to_text": {"0": "Класс ноль", "1": "Класс один"},
        "num_classes": 2,
    }
    enc_path = tmp / "label_encoder.json"
    enc_path.write_text(json.dumps(encoder, ensure_ascii=False), encoding="utf-8")

    monkeypatch_module.setenv("GRNTI_MAIN_DIR", str(main_dir))
    monkeypatch_module.setenv("GRNTI_BASELINE_DIR", str(baseline_dir))
    monkeypatch_module.setenv("GRNTI_LABEL_ENCODER", str(enc_path))
    monkeypatch_module.setenv("GRNTI_MODEL_VERSION", "test-0.0.1")
    # clear the lru_cache since we set env vars mid-session
    from grnti_text_classifier.serving.routes import _load_model

    _load_model.cache_clear()
    yield


@pytest.fixture(scope="module")
def monkeypatch_module():
    """module-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_labels(client):
    r = client.get("/labels")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    assert data[0]["label_text"] == "Класс ноль"


def test_classify_main(client):
    r = client.post("/classify", json={"text": "Пример абстракта."})
    assert r.status_code == 200
    body = r.json()
    assert "top1" in body and "top5" in body
    assert len(body["top5"]) == 2  # num_labels=2, min(5, 2) = 2
    assert body["model_version"] == "test-0.0.1"


def test_classify_baseline_toggle(client):
    r = client.post("/classify?model=baseline", json={"text": "Ещё один пример."})
    assert r.status_code == 200
    body = r.json()
    # model_name comes from Path(model_dir).name -> "baseline"
    assert body["model_name"] == "baseline"
