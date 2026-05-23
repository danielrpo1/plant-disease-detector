# Notebook Kaggle en 15 minutos

1. **New Notebook** → GPU T4 x2
2. **Add Data** → `vipoooool/new-plant-diseases-dataset`
3. **Add Data** → sube un zip con la carpeta `src/` de este repo (o clona desde GitHub)

## Celda 1 — Instalar

```python
!pip install -q pytorch-lightning torchmetrics wandb onnx onnxruntime
```

## Celda 2 — Encontrar ruta del dataset

```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    if "train" in dirs and "valid" in dirs:
        DATA_DIR = root
        print("DATA_DIR =", DATA_DIR)
        break
```

## Celda 3 — Entrenar (modo rápido)

```python
import sys
sys.path.insert(0, "/kaggle/working")  # si copiaste src aquí
# o sys.path.insert(0, "/kaggle/input/tu-dataset-src")

!cd /kaggle/working && python -m src.train --data_dir "{DATA_DIR}" --fast --wandb_off
```

Quita `--wandb_off` si haces `wandb login` y quieres gráficas para el informe.

## Celda 4 — Export ONNX

```python
import glob
ckpt = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
!python -m src.export_onnx --checkpoint {ckpt} --class_mapping checkpoints/class_mapping.json
```

## Celda 5 — Descargar

En Kaggle: **Output** → descarga `checkpoints/` y `artifacts/`.

## EDA (1 celda extra para el informe)

```python
from collections import Counter
from pathlib import Path
train = Path(DATA_DIR) / "train"
counts = {c.name: len(list(c.glob("*"))) for c in train.iterdir() if c.is_dir()}
print("Clases:", len(counts), "| Total train:", sum(counts.values()))
# Gráfico opcional con matplotlib
```
