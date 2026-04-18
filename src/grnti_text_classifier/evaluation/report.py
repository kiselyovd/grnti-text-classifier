"""Summary report builder — merges main and baseline metrics into a JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_summary(
    main_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    out_path: str | Path,
) -> dict[str, Any]:
    """Write a flat JSON summary combining main and baseline scoring results.

    Parameters
    ----------
    main_metrics:
        Output of ``compute_metrics`` for the primary model.
    baseline_metrics:
        Output of ``compute_metrics`` for the baseline model.
    out_path:
        Destination path for the JSON file.  Parent dirs are created if needed.

    Returns
    -------
    The summary dict that was written to disk.
    """
    assert main_metrics["n"] == baseline_metrics["n"], (
        f"Test-set sizes differ: {main_metrics['n']} vs {baseline_metrics['n']}"
    )
    assert main_metrics["num_classes"] == baseline_metrics["num_classes"], (
        "num_classes mismatch between main and baseline"
    )

    def pct(v: float) -> str:
        return f"{v:.1%}"

    summary = {
        "main_model": "FacebookAI/xlm-roberta-base",
        "baseline_model": "DeepPavlov/rubert-base-cased",
        "main_top1": pct(main_metrics["top1_accuracy"]),
        "main_top5": pct(main_metrics["top5_accuracy"]),
        "main_macro_f1": pct(main_metrics["macro_f1"]),
        "main_weighted_f1": pct(main_metrics["weighted_f1"]),
        "baseline_top1": pct(baseline_metrics["top1_accuracy"]),
        "baseline_top5": pct(baseline_metrics["top5_accuracy"]),
        "baseline_macro_f1": pct(baseline_metrics["macro_f1"]),
        "baseline_weighted_f1": pct(baseline_metrics["weighted_f1"]),
        "test_size": main_metrics["n"],
        "num_classes": main_metrics["num_classes"],
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
