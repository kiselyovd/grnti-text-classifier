"""Upload trained artifacts to HuggingFace Hub.

Uploads the XLM-RoBERTa main checkpoint (``artifacts/main/hf/``) and
optionally the ruBERT baseline (``artifacts/baseline/hf/``) to HF Hub.
Renders a Jinja2 model card from ``docs/model_card.md.j2`` and writes it
as ``README.md`` inside the main artifact directory before uploading.

Supports ``--dry-run`` to render and preview the model card locally without
touching HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file; return empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def _widget_text(widget_path: Path) -> str:
    """Extract the ``text`` field from the widget sample payload JSON."""
    if not widget_path.exists():
        return "Введите текст научной статьи."
    data = _load_json(widget_path)
    text: str = data.get("text", "Введите текст научной статьи.")
    return text


def _format_metrics_table(summary: dict[str, Any]) -> str:
    """Render a Markdown comparison table from ``metrics_summary.json``.

    Expected keys: main_top1, main_top5, main_macro_f1, main_weighted_f1,
    baseline_top1, baseline_top5, baseline_macro_f1, baseline_weighted_f1,
    main_model, baseline_model.
    """
    if not summary:
        return "TBD"

    main_model = summary.get("main_model", "XLM-RoBERTa-base (main)")
    baseline_model = summary.get("baseline_model", "ruBERT-base-cased (baseline)")

    main_top1 = summary.get("main_top1", "—")
    main_top5 = summary.get("main_top5", "—")
    main_macro = summary.get("main_macro_f1", "—")
    main_weighted = summary.get("main_weighted_f1", "—")

    baseline_top1 = summary.get("baseline_top1", "—")
    baseline_top5 = summary.get("baseline_top5", "—")
    baseline_macro = summary.get("baseline_macro_f1", "—")
    baseline_weighted = summary.get("baseline_weighted_f1", "—")

    header = "| Model | Top-1 | Top-5 | Macro F1 | Weighted F1 |"
    sep = "|-------|------:|------:|---------:|------------:|"
    row_main = f"| {main_model} | {main_top1} | {main_top5} | {main_macro} | {main_weighted} |"
    row_base = (
        f"| {baseline_model} | {baseline_top1} | {baseline_top5}"
        f" | {baseline_macro} | {baseline_weighted} |"
    )

    return "\n".join([header, sep, row_main, row_base])


def render_card(template_path: Path, context: dict[str, Any]) -> str:
    """Render the Jinja2 model-card template and return the rendered string.

    ``autoescape`` is False because the template produces Markdown + YAML
    frontmatter, not HTML; HTML-escaping would mangle Markdown table pipes
    and YAML braces.
    """
    env = Environment(  # nosec B701 - renders Markdown/YAML, not HTML
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    tpl = env.get_template(template_path.name)
    rendered: str = tpl.render(**context)
    return rendered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish grnti-text-classifier artifacts to HuggingFace Hub.",
    )
    parser.add_argument("--repo-id", default="kiselyovd/grnti-text-classifier")
    parser.add_argument("--main-dir", default="artifacts/main/hf")
    parser.add_argument("--baseline-dir", default="artifacts/baseline/hf")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--summary", default="reports/metrics_summary.json")
    parser.add_argument("--widget", default="data/widget/sample_payload.json")
    parser.add_argument("--template", default="docs/model_card.md.j2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render model card locally and print preview; skip upload.",
    )
    args = parser.parse_args()

    main_dir = Path(args.main_dir)
    baseline_dir = Path(args.baseline_dir)
    template_path = Path(args.template)

    # Load inputs
    main_metrics: dict[str, Any] = _load_json(Path(args.metrics))
    summary: dict[str, Any] = _load_json(Path(args.summary))

    if not main_metrics:
        print(
            f"[warn] {args.metrics} not found or empty — metrics will show placeholder values.",
            file=sys.stderr,
        )
        main_metrics = {
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "num_classes": 28,
            "n": 0,
        }

    context: dict[str, Any] = {
        "repo_id": args.repo_id,
        "widget_text": _widget_text(Path(args.widget)),
        "main_metrics": main_metrics,
        "test_size": main_metrics.get("n", summary.get("test_size", "TBD")),
        "num_classes": main_metrics.get("num_classes", summary.get("num_classes", 28)),
        "metrics_table": _format_metrics_table(summary),
    }

    rendered = render_card(template_path, context)

    if args.dry_run:
        lines = rendered.splitlines()
        print("[dry-run] First 80 lines of rendered model card:")
        print("---")
        for line in lines[:80]:
            print(line)
        print("---")
        print(f"[dry-run] Total lines: {len(lines)}")
        print("[dry-run] No upload performed.")
        return

    # Live upload
    if not main_dir.exists():
        raise SystemExit(f"Main artifact dir not found: {main_dir}")

    # Write README.md into main_dir before upload
    readme_path = main_dir / "README.md"
    readme_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote model card to {readme_path}")

    hf_api = HfApi()
    hf_api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)

    hf_api.upload_folder(
        folder_path=str(main_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="publish main XLM-R checkpoint + card",
    )
    print(f"Uploaded main checkpoint from {main_dir}")

    # Optional baseline upload
    if baseline_dir.exists() and any(baseline_dir.iterdir()):
        hf_api.upload_folder(
            folder_path=str(baseline_dir),
            path_in_repo="baseline",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="publish baseline ruBERT checkpoint",
        )
        print(f"Uploaded baseline checkpoint from {baseline_dir} -> baseline/")

    print(f"Published to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
