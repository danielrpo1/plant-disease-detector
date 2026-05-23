# Qué sigue después del EDA

Ya sabes **cuántas imágenes hay**, **cuántas clases** y que el dataset está **desbalanceado (~36×)**.  
El siguiente bloque es el **modelo + despliegue**.

```
EDA ✅  →  ENTRENAR  →  ONNX  →  API  →  WEB (Pages) ✅
```

---

## Paso 2 — Entrenar el modelo (1–3 h en GPU)

### Datos en el Studio

```bash
cd plant-disease-detector
python scripts/download_dataset_hf.py --out data/plantvillage
# o sube el zip de Kaggle y apunta a esa carpeta
```

### Entrenamiento rápido (entregable hoy)

```bash
python -m src.train \
  --data_dir data/plantvillage \
  --fast \
  --wandb_off
```

- `--fast` = 50 fotos/clase, 10 épocas (~1–2 h).
- Sin W&B: logs en `lightning_logs/` o TensorBoard.
- Con W&B: quita `--wandb_off` y haz `wandb login`.

**Salida:** `checkpoints/efficientnet-*.ckpt` y `checkpoints/class_mapping.json`.

### Qué capturar para el informe

- Curva **loss** y **val_acc** por época.
- Mejor **val_acc** y ejemplo de predicción correcta/incorrecta.

---

## Paso 3 — Exportar a ONNX (5 min)

```bash
CKPT=$(ls -t checkpoints/*.ckpt | head -1)
python -m src.export_onnx \
  --checkpoint "$CKPT" \
  --class_mapping checkpoints/class_mapping.json \
  --output artifacts/model.onnx
```

**Salida:** `artifacts/model.onnx` + `artifacts/model.meta.json`.

Prueba local opcional:

```bash
pip install onnxruntime
# sube una imagen de prueba y corre inferencia (ver README)
```

---

## Paso 4 — API de predicción

### Opción A — Demo rápida (sin AWS)

En el Studio o tu Mac:

```bash
python scripts/local_api.py \
  --onnx artifacts/model.onnx \
  --meta artifacts/model.meta.json
```

### Opción B — AWS (entregable MLOps)

1. Copia `model.onnx` y `model.meta.json` a `lambda/`.
2. `docker build` + push a ECR + Lambda container.
3. API Gateway POST `/predict` con CORS.
4. Copia la URL pública.

---

## Paso 5 — Conectar GitHub Pages

La web ya se publica sola en:

**https://danielrpo1.github.io/plant-disease-detector/**

Edita `webapp/config.js` en el repo:

```javascript
window.API_URL = "https://TU-API.execute-api.us-east-1.amazonaws.com/prod/predict";
```

Haz commit y push → en ~1 min la web llama a tu API.

> Sin API configurada, la UI se ve pero al analizar dirá que falta `API_URL`.

---

## Paso 6 — Cierre académico

- [ ] README + enlace al repo y a Pages.
- [ ] Capturas del EDA (ya en `notebooks/eda_outputs/`).
- [ ] Capturas de entrenamiento (W&B o TensorBoard).
- [ ] Screenshot de la web con una predicción real.
- [ ] Párrafo: por qué EfficientNet vs CNN baseline (opcional).
- [ ] Limitaciones: desbalance, dominio (fondo blanco vs campo).

---

## Orden si tienes poco tiempo

1. `train --fast`  
2. `export_onnx`  
3. `local_api.py` + `config.js` → `http://127.0.0.1:8000/predict` (demo en vivo)  
4. AWS solo si sobra tiempo  
5. Informe con EDA + arquitectura + screenshot web
