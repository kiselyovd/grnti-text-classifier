"""Lightning Trainer entrypoint for GRNTI text classification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from ..data.datamodule import GRNTIDataModule
from ..models.lightning_module import GRNTIClassifier


def train_one(
    model_builder,
    model_name_for_tokenizer: str,
    processed_dir: Path,
    out_dir: Path,
    *,
    max_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    patience: int = 2,
    seed: int = 42,
    max_length: int = 256,
    num_workers: int = 0,
    save_hf: bool = True,
) -> Path:
    """Train a single GRNTI classifier run.

    Parameters
    ----------
    model_builder:
        Callable ``(num_labels: int) -> PreTrainedModel``.
    model_name_for_tokenizer:
        HuggingFace model name used to load the tokenizer, e.g.
        ``"FacebookAI/xlm-roberta-base"``.
    processed_dir:
        Directory containing ``train.parquet``, ``val.parquet``,
        ``test.parquet``, and ``label_encoder.json``.
    out_dir:
        Root output directory for this run.
    max_epochs:
        Maximum training epochs.
    batch_size:
        Batch size for training and validation.
    lr:
        Peak learning rate for AdamW.
    weight_decay:
        AdamW weight-decay coefficient.
    warmup_ratio:
        Fraction of total steps used for linear warmup.
    patience:
        Early-stopping patience (in validation epochs).
    seed:
        Global random seed.
    max_length:
        Tokeniser max-sequence length.
    num_workers:
        DataLoader worker count.
    save_hf:
        When ``True`` (default) saves the best checkpoint as a HuggingFace
        model directory and returns that path.  When ``False``, skips the HF
        save and returns the raw checkpoint path instead (useful for sweeps).

    Returns
    -------
    Path
        ``out_dir / "hf"`` when *save_hf* is True; otherwise the path to the
        best Lightning checkpoint file.
    """
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)

    # 1. Global seed
    L.seed_everything(seed, workers=True)

    # 2. DataModule
    dm = GRNTIDataModule(
        processed_dir,
        model_name_for_tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        seed=seed,
    )
    dm.setup()

    # 3. Inverse-frequency class weights
    label_enc = json.loads((processed_dir / "label_encoder.json").read_text(encoding="utf-8"))
    num_classes: int = int(label_enc["num_classes"])

    import pandas as pd
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    freq = np.bincount(train_df["label_idx"].to_numpy(), minlength=num_classes).astype(np.float64)
    weights = 1.0 / np.clip(freq, 1, None)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # 4. Build model + Lightning module
    inner = model_builder(num_labels=num_classes)
    steps_per_epoch = len(dm.train_dataloader())
    total_steps = steps_per_epoch * max_epochs

    lit = GRNTIClassifier(
        inner,
        class_weights=class_weights,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        total_steps=total_steps,
        num_classes=num_classes,
    )

    # 5. Hardware
    precision = "bf16-mixed" if torch.cuda.is_available() else 32
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # 6. Callbacks + logger
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "ckpt",
        filename="{epoch:02d}-{val_macro_f1:.4f}",
        monitor="val/macro_f1",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    es_cb = EarlyStopping(monitor="val/macro_f1", mode="max", patience=patience)
    logger = CSVLogger(save_dir=str(out_dir), name="logs")

    # 7. Trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[ckpt_cb, es_cb],
        logger=logger,
        deterministic="warn",
        enable_progress_bar=True,
        log_every_n_steps=max(1, steps_per_epoch // 10),
    )
    trainer.fit(lit, datamodule=dm)

    # 8. Optionally reload best checkpoint and export to HF format
    if not save_hf:
        return Path(ckpt_cb.best_model_path)

    best = GRNTIClassifier.load_from_checkpoint(
        ckpt_cb.best_model_path,
        model=model_builder(num_labels=num_classes),
        class_weights=None,
    )

    hf_dir = out_dir / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    best.model.save_pretrained(hf_dir)
    dm._tok.save_pretrained(hf_dir)
    return hf_dir
