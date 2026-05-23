# Plan express — 1 día (MVP entregable)

**Meta de hoy:** modelo entrenado (aunque sea en subset) + ONNX + demo web. AWS es bonus si sobra tiempo.

## Lo que NO hagas hoy

- CNN baseline (notebook 02) — omitir
- Dataset completo 87k con 20 épocas — tardaría horas
- SageMaker
- CI/CD perfecto

## Lo que SÍ debes entregar

1. Código Lightning comentado (`src/`)
2. Checkpoint + curvas (W&B o capturas de TensorBoard)
3. `model.onnx` + prueba local
4. README con pipeline
5. Web que llame API **o** demo local con `scripts/local_api.py`

---

## Cronograma (8–10 h)

| Hora | Tarea |
|------|--------|
| 0:00–0:30 | Kaggle: nuevo notebook GPU, add dataset `vipoooool/new-plant-diseases-dataset`, subir carpeta `src/` |
| 0:30–1:00 | EDA mínimo: contar clases, 3 imágenes por clase (screenshot para informe) |
| 1:00–3:00 | **Entrenar `--fast`** en Kaggle (ver comandos abajo) |
| 3:00–3:30 | Export ONNX + test inferencia local |
| 3:30–5:00 | Lambda Docker **O** `local_api.py` si AWS se atasca |
| 5:00–6:00 | Webapp + GitHub Pages |
| 6:00–7:00 | README + capturas W&B + ensayo demo |

---

## Opción A — Lightning AI Studio (tus keys ya en `.env`)

1. En terminal del Mac:
   ```bash
   cd plant-disease-detector
   source scripts/lightning_login.sh
   ```
2. Abre [lightning.ai](https://lightning.ai) → **New Studio** → GPU (L4 o T4).
3. Sube/clona este repo en el Studio (`git clone` o arrastra carpeta `src/`).
4. Sube el dataset (o descárgalo dentro del Studio desde Kaggle).
5. En terminal del Studio:
   ```bash
   pip install -r requirements.txt
   python -m src.train --data_dir /path/train-valid --fast
   python -m src.export_onnx --checkpoint checkpoints/*.ckpt --class_mapping checkpoints/class_mapping.json
   ```
   Las credenciales `LIGHTNING_*` ya vienen inyectadas dentro del Studio.

Ver `notebooks/LIGHTNING_STUDIO.md`.

---

## Opción B — Kaggle (si Lightning falla)

## Comandos Kaggle (copiar en celda)

```python
!pip install -q pytorch-lightning wandb onnx onnxruntime

import os
os.chdir("/kaggle/working")
# Subir src como dataset de Kaggle o clonar tu repo
DATA = "/kaggle/input/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
# Ajusta DATA si la ruta difiere — lista con !ls /kaggle/input/

!python -m src.train --data_dir "$DATA" --fast --wandb_off
# Si tienes W&B: wandb login y quita --wandb_off

!python -m src.export_onnx --checkpoint checkpoints/efficientnet-*.ckpt --class_mapping checkpoints/class_mapping.json
```

Descarga `checkpoints/` y `artifacts/` a tu Mac.

---

## Plan B (sin AWS hoy)

```bash
cd plant-disease-detector
pip install -r requirements.txt
python scripts/local_api.py --onnx artifacts/model.onnx --meta artifacts/model.meta.json
# En webapp/config.js: API_URL = "http://localhost:8000/predict"
```

En la memoria escribes: *"Lambda listo en código; demo con API local por límite de tiempo."*

---

## Plan mínimo AWS (si hay 2 h libres)

1. `docker build` en `lambda/` (copiar `model.onnx` y `model.meta.json` antes)
2. Push a ECR, crear Lambda desde imagen
3. API Gateway POST + CORS
4. Pegar URL en `webapp/config.js`

Ver `infra/deploy.sh`.
