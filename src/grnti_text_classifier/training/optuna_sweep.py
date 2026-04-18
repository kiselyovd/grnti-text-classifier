"""Optuna hyper-parameter sweep over train_one for GRNTI classifiers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

from .train import train_one


def run_sweep(
    processed_dir: Path,
    out_dir: Path,
    *,
    model_builder: Callable[..., Any],
    model_name_for_tokenizer: str,
    n_trials: int = 10,
    seed: int = 42,
    trial_epochs: int = 3,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Run an Optuna TPE sweep over learning-rate, weight-decay and warmup-ratio.

    Parameters
    ----------
    processed_dir:
        Pre-processed data directory (parquet splits + label_encoder.json).
    out_dir:
        Root directory; each trial writes to ``out_dir / "trial_<n>"``.
    model_builder:
        Callable ``(num_labels: int) -> PreTrainedModel``.
    model_name_for_tokenizer:
        HuggingFace model name used to build the tokeniser.
    n_trials:
        Number of Optuna trials.
    seed:
        Seed for the TPE sampler (for reproducibility).
    trial_epochs:
        Max epochs per trial (use a small number for speed).
    batch_size:
        Mini-batch size forwarded to ``train_one``.
    num_workers:
        DataLoader workers forwarded to ``train_one``.

    Returns
    -------
    dict
        ``{"best_params": {...}, "best_value": float}``
    """
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=False)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15, log=False)

        trial_dir = out_dir / f"trial_{trial.number}"

        train_one(
            model_builder,
            model_name_for_tokenizer,
            processed_dir,
            trial_dir,
            max_epochs=trial_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            patience=1,
            num_workers=num_workers,
            save_hf=False,
        )

        # Read the best val/macro_f1 from the CSVLogger output
        metrics_csv = trial_dir / "logs" / "version_0" / "metrics.csv"
        import pandas as pd

        df = pd.read_csv(metrics_csv)
        if "val/macro_f1" not in df.columns:
            return 0.0
        best_f1 = float(df["val/macro_f1"].dropna().max())
        return best_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    return {"best_params": study.best_params, "best_value": float(study.best_value)}
