"""Lightning wrapper for GRNTI sequence classification models."""

from __future__ import annotations

import lightning
import torch
import torch.nn.functional as F  # noqa: N812
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from transformers import PreTrainedModel, get_linear_schedule_with_warmup


class GRNTIClassifier(lightning.LightningModule):
    """Lightning module wrapping any HuggingFace sequence-classification model."""

    def __init__(
        self,
        model: PreTrainedModel,
        class_weights: torch.Tensor | None = None,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int = 1000,
        num_classes: int = 28,
    ) -> None:
        super().__init__()
        self.model = model
        self.class_weights = class_weights
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        _top5 = min(5, num_classes)
        self.val_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.val_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=_top5, average="micro")
        self.test_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro")
        self.test_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=_top5, average="micro")

    def _step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Lazy device-move for class_weights so constructor stays device-agnostic.
        if self.class_weights is not None and self.class_weights.device != batch["labels"].device:
            self.class_weights = self.class_weights.to(batch["labels"].device)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = F.cross_entropy(out.logits, batch["labels"], weight=self.class_weights)
        preds = out.logits.argmax(-1)
        return loss, out.logits, preds

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _logits, preds = self._step(batch)
        self.train_f1(preds, batch["labels"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/macro_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, logits, preds = self._step(batch)
        self.val_f1(preds, batch["labels"])
        self.val_top1(logits, batch["labels"])
        self.val_top5(logits, batch["labels"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/macro_f1", self.val_f1, prog_bar=True)
        self.log("val/top1_acc", self.val_top1, prog_bar=True)
        self.log("val/top5_acc", self.val_top5, prog_bar=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, logits, preds = self._step(batch)
        self.test_f1(preds, batch["labels"])
        self.test_top1(logits, batch["labels"])
        self.test_top5(logits, batch["labels"])
        self.log("test/loss", loss)
        self.log("test/macro_f1", self.test_f1)
        self.log("test/top1_acc", self.test_top1)
        self.log("test/top5_acc", self.test_top5)

    def configure_optimizers(self) -> dict[str, object]:  # type: ignore[override]
        opt = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,  # type: ignore[attr-defined]
            weight_decay=self.hparams.weight_decay,  # type: ignore[attr-defined]
        )
        total_steps: int = self.hparams.total_steps  # type: ignore[attr-defined]
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)  # type: ignore[attr-defined]
        sch = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}
