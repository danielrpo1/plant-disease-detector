"""
LightningModule: EfficientNet-B0 + cabeza de 38 clases.

Flujo de transfer learning (resumido):
  1. Cargamos pesos ImageNet en el backbone.
  2. Épocas 0..freeze_epochs-1: backbone congelado → solo aprende la cabeza Linear.
  3. Después: descongelamos todo y fine-tune con learning rate más bajo en el backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class PlantDiseaseModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 38,
        learning_rate: float = 1e-3,
        backbone_lr: float = 1e-4,
        dropout: float = 0.3,
        freeze_epochs: int = 2,
        # En 1 día: 2 épocas congelado + 8 fine-tune suele bastar con subset
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.backbone_lr = backbone_lr
        self.dropout = dropout
        self.freeze_epochs = freeze_epochs

        # --- Backbone pretrained (feature extractor) ---
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1280 para B0

        # Quitamos el clasificador original de ImageNet (1000 clases)
        self.backbone.classifier = nn.Identity()

        # --- Cabeza nueva para nuestras 38 enfermedades ---
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # Métricas: accuracy en train y valid (multiclass, 38 clases)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

        # Empezamos con backbone congelado (época 0)
        self._set_backbone_requires_grad(False)

    def _set_backbone_requires_grad(self, trainable: bool) -> None:
        """Congela o descongela parámetros del EfficientNet."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: batch de imágenes [B, 3, 224, 224] → logits [B, num_classes]"""
        features = self.backbone(x)
        return self.head(features)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        if stage == "train":
            self.train_acc(preds, labels)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.val_acc(preds, labels)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_train_epoch_start(self) -> None:
        """
        Al inicio de cada época decidimos si el backbone entrena o no.
        current_epoch empieza en 0.
        """
        if self.current_epoch < self.freeze_epochs:
            self._set_backbone_requires_grad(False)
        else:
            self._set_backbone_requires_grad(True)

    def configure_optimizers(self):
        """
        Dos grupos de parámetros:
          - Cabeza: LR más alto (aprende rápido al inicio)
          - Backbone: LR bajo en fine-tune (no destruir features de ImageNet)
        """
        head_params = list(self.head.parameters())
        backbone_params = list(self.backbone.parameters())

        param_groups = [
            {"params": head_params, "lr": self.learning_rate},
            {"params": backbone_params, "lr": self.backbone_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
            },
        }

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Inferencia: probabilidades con softmax."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return F.softmax(logits, dim=1)
