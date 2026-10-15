"""Native label, report-source and publication regression checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from grnti_text_classifier.inference.native import CONTRACT, parse_labels

ROOT = Path(__file__).resolve().parents[1]


def test_native_label_contract():
    labels = parse_labels(CONTRACT["id2label"])
    assert len(labels) == 28
    assert labels[0]["code"] == "20000"
    assert labels[23]["code"] != labels[24]["code"]
    assert labels[23]["title"] == labels[24]["title"]
    assert CONTRACT["max_length"] == 256


def test_manifest_report_is_hash_bound():
    spec = importlib.util.spec_from_file_location(
        "artifact_manifest", ROOT / "scripts/artifact_manifest.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest, scores = module.validate(ROOT)
    assert manifest["hub"]["revision"] == CONTRACT["revision"]
    assert scores["Top-5 accuracy"] > scores["Top-1 accuracy"]


def test_publisher_rejects_missing_report_before_upload(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("publisher", ROOT / "scripts/publish_to_hf.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys, "argv", ["publish", "--dry-run", "--metrics", str(tmp_path / "missing.json")]
    )
    with pytest.raises(SystemExit, match="Missing evaluation report"):
        module.main()


def test_api_default_labels_without_external_encoder(monkeypatch):
    from grnti_text_classifier.serving.routes import _load_labels

    monkeypatch.delenv("GRNTI_LABEL_ENCODER", raising=False)
    assert _load_labels() == {int(key): value for key, value in CONTRACT["id2label"].items()}
