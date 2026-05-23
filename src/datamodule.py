"""
LightningDataModule: carga imágenes del dataset Plant Village (train/valid por carpetas).

Responsabilidades:
  - Encontrar carpetas train/ y valid/ (o train/ + test/ renombrado)
  - Aplicar augmentations solo en entrenamiento
  - Normalizar como ImageNet (requerido por EfficientNet pretrained)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Estadísticas ImageNet: el backbone EfficientNet fue entrenado con estos valores.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PlantDiseaseDataModule(pl.LightningDataModule):
    """
    DataModule de PyTorch Lightning.

    Lightning llama en orden:
      setup() → train_dataloader() / val_dataloader()
    Así el mismo código sirve en notebook, CLI y tests.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        # Para prueba rápida en 1 día: limita imágenes por clase (None = todas)
        max_samples_per_class: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_samples_per_class = max_samples_per_class

        # Se rellenan en setup()
        self.train_dataset: ImageFolder | None = None
        self.val_dataset: ImageFolder | None = None
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

    def _resolve_splits(self) -> tuple[Path, Path]:
        """
        El dataset de Kaggle suele traer:
          New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/
            train/  valid/

        Aceptamos también valid/ llamado 'validation' o usar test/ como valid.
        """
        candidates_train = ["train", "Train"]
        candidates_val = ["valid", "validation", "val", "test"]

        train_dir = None
        for name in candidates_train:
            p = self.data_dir / name
            if p.is_dir():
                train_dir = p
                break

        val_dir = None
        for name in candidates_val:
            p = self.data_dir / name
            if p.is_dir():
                val_dir = p
                break

        if train_dir is None or val_dir is None:
            raise FileNotFoundError(
                f"No encontré train/ y valid/ dentro de {self.data_dir}. "
                "Ajusta data_dir a la carpeta que contiene esas subcarpetas."
            )
        return train_dir, val_dir

    def _build_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        """
        Train: augmentations ligeras (hojas similares, mucha variación de luz en campo).
        Valid: solo resize + normalize (evaluación justa).
        """
        train_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return train_tf, val_tf

    def _maybe_subsample(self, dataset: ImageFolder) -> ImageFolder:
        """
        Modo '1 día': entrenar con N fotos por clase (~38*N imágenes) para terminar en ~30-60 min.
        """
        if self.max_samples_per_class is None:
            return dataset

        from collections import defaultdict

        by_class: dict[int, list] = defaultdict(list)
        for i, (_, label) in enumerate(dataset.samples):
            by_class[label].append(i)

        keep_indices: list[int] = []
        for label, indices in by_class.items():
            keep_indices.extend(indices[: self.max_samples_per_class])

        dataset.samples = [dataset.samples[i] for i in keep_indices]
        dataset.targets = [dataset.targets[i] for i in keep_indices]
        return dataset

    def setup(self, stage: str | None = None) -> None:
        """Lightning invoca esto antes de los dataloaders."""
        train_dir, val_dir = self._resolve_splits()
        train_tf, val_tf = self._build_transforms()

        self.train_dataset = ImageFolder(root=str(train_dir), transform=train_tf)
        self.val_dataset = ImageFolder(root=str(val_dir), transform=val_tf)

        # Misma orden de clases en train y valid (ImageFolder ordena alfabéticamente)
        self.class_to_idx = self.train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.train_dataset = self._maybe_subsample(self.train_dataset)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        # num_workers=0 evita tensores en CPU con GPU en algunos entornos (Lightning Studio)
        nw = 0 if self.num_workers > 0 else 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def save_class_mapping(self, path: str | Path) -> None:
        """Guarda índices para ONNX/Lambda (mismo orden que la capa Linear)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "class_to_idx": self.class_to_idx,
            "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
