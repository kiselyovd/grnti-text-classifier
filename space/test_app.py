"""Production-root and native-contract regression checks without a weight download."""

import json
from pathlib import Path

import app
import gradio as gr
import pytest
import torch
from native import CONTRACT, parse_labels, rank_logits, validate_text


def test_production_root_and_personal_identity():
    config = app.demo.get_config_file()
    serialized = json.dumps(config, ensure_ascii=False, default=str)
    assert config["title"] == "GRNTI Classifier"
    assert "Scientific text in Russian" in serialized
    assert "Full subject name" in serialized
    assert "https://huggingface.co/kiselyovd" in serialized
    assert '"value": "en"' in serialized
    assert app.DEVICE == "cpu"
    assert app.demo.enable_queue


def test_exact_codes_and_duplicate_titles():
    labels = parse_labels(CONTRACT["id2label"])
    assert labels[0]["code"] == "20000"
    assert labels[23]["code"] == "680000"
    assert labels[24]["code"] == "683500"
    assert labels[23]["title"] == labels[24]["title"]
    scores = rank_logits(torch.arange(28).float().unsqueeze(0), labels)
    assert [row["index"] for row in scores] == [27, 26, 25, 24, 23]
    rows, note = app.render_result({"predictions": scores, "truncated": True}, "ru")
    assert rows[-1][1] == labels[23]["title"]
    assert "256" in note
    with pytest.raises(ValueError):
        rank_logits(torch.full((1, 28), float("nan")), labels)


@pytest.mark.parametrize(
    "text, error", [("  ", "empty_text"), ("я" * 20001, "text_too_long")], ids=["empty", "too-long"]
)
def test_validation_before_model_load(text, error, monkeypatch):
    def forbidden(_text):
        pytest.fail("Invalid input must not allocate GPU or load weights")

    monkeypatch.setattr(app, "infer", forbidden)
    with pytest.raises(ValueError, match=error):
        validate_text(text)
    with pytest.raises(gr.Error):
        app.classify(text, "ru")


def test_real_result_rendering_preserves_every_full_name(monkeypatch):
    result = {
        "predictions": rank_logits(torch.zeros(1, 28), parse_labels(CONTRACT["id2label"])),
        "truncated": False,
        "input_tokens": 30,
        "model_revision": CONTRACT["revision"],
    }
    monkeypatch.setattr(app, "infer", lambda text: result)
    state, rows, note, raw = app.classify("Аннотация", "en")
    assert len(rows) == 5 and state == raw == result and note == ""
    assert rows[0][0] == "20000"
    assert "Аннотация" not in json.dumps(raw, ensure_ascii=False)


def test_space_core_matches_repository():
    source = Path(__file__).resolve().parents[1] / "src/grnti_text_classifier/inference"
    for name in ("native.py", "model-contract.json"):
        assert Path(__file__).with_name(name).read_bytes() == (source / name).read_bytes()


def test_space_metadata_uses_supported_palette():
    readme = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
    allowed = {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
    for line in readme.splitlines():
        if line.startswith(("colorFrom:", "colorTo:")):
            assert line.split(":", 1)[1].strip() in allowed


def test_actual_production_launch_uses_accessible_theme_and_owner_image(monkeypatch):
    import runpy

    captured = {}

    def launch(_self, **options):
        captured.update(options)

    monkeypatch.setattr(gr.Blocks, "launch", launch)
    runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
    assert captured["theme"].button_primary_background_fill == "*primary_700"
    assert captured["theme"].button_primary_background_fill_dark == "*primary_700"
    assert "image.alt" in captured["js"]
    assert "kiselyovd/avatar" in captured["js"]


def test_service_quota_has_a_localized_message(monkeypatch):
    def quota(_text):
        raise gr.Error("ZeroGPU quota exceeded")

    monkeypatch.setattr(app, "infer", quota)
    for language in ("en", "ru"):
        with pytest.raises(gr.Error, match=app.COPY[language]["quota"]):
            app.classify("Scientific abstract", language)
