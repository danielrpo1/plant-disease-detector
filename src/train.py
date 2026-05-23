"""
Script de entrenamiento — ejecutar desde la raíz del repo:

  python -m src.train --data_dir /ruta/dataset --fast

Opciones:
  --fast     → 50 img/clase, 10 épocas (modo 1 día en Kaggle)
  --full     → dataset completo, 15 épocas (varias horas)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import PlantDiseaseDataModule
from src.model import PlantDiseaseModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar clasificador de enfermedades en plantas")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Carpeta que contiene subcarpetas train/ y valid/",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Entrenamiento rápido para demo (subset + pocas épocas)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="plant-disease-eafit")
    parser.add_argument("--wandb_off", action="store_true", help="Desactiva W&B (solo TensorBoard local)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_dotenv() -> None:
    """Carga .env local (LIGHTNING_*, WANDB_API_KEY) sin subirlo a git."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass


def main() -> None:
    _load_dotenv()
    args = parse_args()
    pl.seed_everything(args.seed)

    # --- Hiperparámetros según modo ---
    if args.fast:
        max_per_class = 50
        epochs = args.epochs or 10
        freeze_epochs = 2
    else:
        max_per_class = None
        epochs = args.epochs or 15
        freeze_epochs = 3

    # --- Datos ---
    dm = PlantDiseaseDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples_per_class=max_per_class,
    )
    dm.setup()
    num_classes = len(dm.class_to_idx)

    mapping_path = Path(args.checkpoint_dir) / "class_mapping.json"
    dm.save_class_mapping(mapping_path)
    print(f"Clases: {num_classes} | Mapeo guardado en {mapping_path}")

    # --- Modelo ---
    model = PlantDiseaseModule(
        num_classes=num_classes,
        freeze_epochs=freeze_epochs,
    )

    # --- Callbacks: guardar mejor modelo por val_acc ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="efficientnet-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )
    early_stop = EarlyStopping(monitor="val_acc", patience=4, mode="max")

    # --- Logger ---
    logger: pl.loggers.Logger
    if args.wandb_off:
        logger = pl.loggers.TensorBoardLogger("logs", name="plant_disease")
    else:
        logger = WandbLogger(project=args.wandb_project, log_model=False)

    # --- Trainer Lightning ---
    import torch

    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision=32,  # 16-mixed puede fallar con backbone congelado en algunos drivers
        callbacks=[checkpoint_cb, early_stop],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)

    best = checkpoint_cb.best_model_path
    print(f"\n✓ Entrenamiento terminado. Mejor checkpoint: {best}")
    print(f"  Siguiente paso: python -m src.export_onnx --checkpoint {best} --class_mapping {mapping_path}")


if __name__ == "__main__":
    main()
