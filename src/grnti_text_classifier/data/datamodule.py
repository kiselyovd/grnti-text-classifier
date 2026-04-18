"""Lightning DataModule wrapping GRNTIDataset for HuggingFace tokenizers."""
from __future__ import annotations

from pathlib import Path

import lightning
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .grnti import ENCODED_COL, TEXT_COL


class GRNTIDataset(Dataset):
    """Map-style dataset over a processed GRNTI DataFrame."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256) -> None:
        self.texts: list[str] = df[TEXT_COL].tolist()
        self.labels = df[ENCODED_COL].to_numpy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class GRNTIDataModule(lightning.LightningDataModule):
    """LightningDataModule that loads train/val/test parquet splits."""

    def __init__(
        self,
        processed_dir: str | Path,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 256,
        num_workers: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.seed = seed
        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

    def setup(self, stage: str | None = None) -> None:
        self.train_df = pd.read_parquet(self.processed_dir / "train.parquet")
        self.val_df = pd.read_parquet(self.processed_dir / "val.parquet")
        self.test_df = pd.read_parquet(self.processed_dir / "test.parquet")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            GRNTIDataset(self.train_df, self._tok, self.max_length),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            GRNTIDataset(self.val_df, self._tok, self.max_length),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            GRNTIDataset(self.test_df, self._tok, self.max_length),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
