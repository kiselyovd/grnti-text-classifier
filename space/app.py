"""Production Gradio root for the free GRNTI inference Space."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

# ZeroGPU must patch CUDA before torch is imported by native.py.
if os.environ.get("SPACE_ID"):
    import spaces
else:
    spaces = None

import gradio as gr
from native import CONTRACT, MAX_CHARACTERS, NativeClassifier, validate_text

CONTENT = json.loads(Path(__file__).with_name("copy.json").read_text(encoding="utf-8"))
EXAMPLES = CONTENT["EXAMPLES"]
COPY = CONTENT["COPY"]

DEVICE = os.environ.get("GRNTI_DEVICE", "cuda" if spaces else "cpu")


@lru_cache(maxsize=1)
def classifier() -> NativeClassifier:
    """Cache model weights, never visitor inputs or predictions."""
    return NativeClassifier(device=DEVICE)


if spaces and DEVICE == "cuda":
    classifier()


def infer(text: str) -> dict[str, Any]:
    """Run real inference; validation happens before a ZeroGPU allocation."""
    return classifier().classify(text)


if spaces and DEVICE == "cuda":
    infer = spaces.GPU(duration=10)(infer)


def render_result(result: dict[str, Any] | None, language: str) -> tuple[list[list[str]], str]:
    """Keep full names and exact code strings readable in both languages."""
    if not result:
        return [], ""
    rows = [
        [row["code"], row["title"], f"{row['score']:.2%}" if row["score"] >= 0.0001 else "<0.01%"]
        for row in result["predictions"]
    ]
    return rows, COPY[language]["cut"] if result["truncated"] else ""


def classify(
    text: str, language: str
) -> tuple[dict[str, Any], list[list[str]], str, dict[str, Any]]:
    """Validate before GPU scheduling and expose no visitor text in outputs."""
    copy = COPY[language]
    try:
        cleaned = validate_text(text)
    except ValueError as error:
        raise gr.Error(copy["empty"] if str(error) == "empty_text" else copy["long"]) from None
    try:
        result = infer(cleaned)
    except gr.Error as error:
        raise gr.Error(copy["quota"] if "ZeroGPU" in str(error) else copy["failed"]) from None
    except Exception:
        # Avoid framework tracebacks containing tokenized inputs or visitor text.
        raise gr.Error(copy["failed"]) from None
    rows, note = render_result(result, language)
    return result, rows, note, result


def build_demo() -> gr.Blocks:
    """Construct the actual production UI without downloading a model on CPU."""
    copy = COPY["en"]
    with gr.Blocks(title="GRNTI Classifier", analytics_enabled=False) as demo:
        result = gr.State(None)
        with gr.Row(elem_id="header"):
            title = gr.Markdown(f"# {copy['title']}", elem_id="intro", scale=5)
            language = gr.Radio(
                [("English", "en"), ("Русский", "ru")],
                value="en",
                show_label=False,
                container=False,
                min_width=220,
                scale=1,
            )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                examples = gr.Dropdown(copy["choices"], value=None, label=copy["examples"])
                text = gr.Textbox(
                    label=copy["input"],
                    placeholder=copy["placeholder"],
                    lines=10,
                    max_lines=18,
                    max_length=MAX_CHARACTERS,
                )
                run = gr.Button(copy["run"], variant="primary")
            with gr.Column(scale=1, min_width=300):
                table = gr.Dataframe(
                    value=[],
                    headers=copy["headers"],
                    datatype=["str"] * 3,
                    type="array",
                    column_count=3,
                    row_count=5,
                    interactive=False,
                    wrap=True,
                    max_height=600,
                    column_widths=["18%", "62%", "20%"],
                    label=copy["output"],
                    elem_id="subjects",
                    buttons=["copy"],
                )
                note = gr.Markdown("")
        with gr.Accordion(copy["about_title"], open=False) as about:
            explanation = gr.Markdown(copy["about"])
            raw = gr.JSON(label=copy["raw"], value=None)
            gr.Markdown(
                f"Checkpoint `{CONTRACT['revision']}` · SHA256 `{CONTRACT['weights_sha256']}`"
            )
        prediction = run.click(
            classify,
            [text, language],
            [result, table, note, raw],
            concurrency_limit=1,
            api_name="classify",
            show_progress="minimal",
        )
        examples.change(
            lambda key: EXAMPLES.get(key, ""), examples, text, queue=False, api_visibility="private"
        )
        text.change(
            lambda: (None, [], "", None),
            outputs=[result, table, note, raw],
            cancels=[prediction],
            queue=False,
            api_visibility="private",
        )

        def change_language(lang: str, current: dict[str, Any] | None) -> tuple[Any, ...]:
            localized = COPY[lang]
            rows, message = render_result(current, lang)
            return (
                gr.Markdown(f"# {localized['title']}"),
                gr.Textbox(label=localized["input"], placeholder=localized["placeholder"]),
                gr.Dropdown(choices=localized["choices"], label=localized["examples"]),
                gr.Button(localized["run"]),
                gr.Dataframe(value=rows, headers=localized["headers"], label=localized["output"]),
                message,
                gr.Accordion(label=localized["about_title"]),
                localized["about"],
                gr.JSON(label=localized["raw"]),
            )

        language.change(
            change_language,
            [language, result],
            [title, text, examples, run, table, note, about, explanation, raw],
            queue=False,
            api_visibility="private",
        )
    return demo.queue(default_concurrency_limit=1, max_size=12)


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Default(
            primary_hue="cyan",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        ).set(
            button_primary_background_fill="*primary_700",
            button_primary_background_fill_dark="*primary_700",
            button_primary_background_fill_hover="*primary_800",
            button_primary_background_fill_hover_dark="*primary_800",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
        ),
        js=Path(__file__).with_name("accessibility.js").read_text(encoding="utf-8"),
        css=Path(__file__).with_name("style.css").read_text(encoding="utf-8"),
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        show_error=False,
        run_history=False,
        enable_monitoring=False,
        mcp_server=False,
        footer_links=[],
        quiet=True,
    )
