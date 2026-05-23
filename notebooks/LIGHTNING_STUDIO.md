# Entrenar en Lightning AI Studio (hoy)

## 1. Login local (ya configurado)

```bash
cd plant-disease-detector
chmod +x scripts/lightning_login.sh
./scripts/lightning_login.sh
```

Debe decir: `logged in as "ddrestrepo"`.

Credenciales en `.env` (no se suben a Git).

## 2. Crear Studio en el navegador

1. https://lightning.ai → **Studios** → **New Studio**
2. Máquina: **GPU** (la más barata disponible: L4 / T4)
3. Cuando abra el IDE, terminal:

```bash
git clone <TU_REPO_GITHUB> plant-disease
cd plant-disease
pip install -r requirements.txt
```

## 3. Dataset

**Opción rápida:** subir zip del dataset a `/teamspace/studios/.../data/` desde el panel de archivos.

**Opción Kaggle dentro del Studio:**

```bash
pip install kaggle
# Configura ~/.kaggle/kaggle.json con tu token de Kaggle
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip -q *.zip
```

Encuentra la carpeta con `train/` y `valid/`:

```bash
find . -type d -name train 2>/dev/null | head -3
```

## 4. Entrenar (modo 1 día)

**Recomendado para el informe:** abre y ejecuta `notebooks/02_entrenamiento.ipynb` (tabla y gráficas de épocas, loss y accuracy).

Alternativa por terminal:

```bash
export DATA_DIR="/ruta/que/contenga/train-y-valid"
python -m src.train --data_dir "$DATA_DIR" --fast --wandb_off
```

## 5. Export y descarga

```bash
CKPT=$(ls -t checkpoints/*.ckpt | head -1)
python -m src.export_onnx --checkpoint "$CKPT" --class_mapping checkpoints/class_mapping.json
```

Descarga `artifacts/` y `checkpoints/` desde el UI del Studio.

## 6. W&B (opcional, 2 min)

```bash
wandb login
python -m src.train --data_dir "$DATA_DIR" --fast
```

## CLI desde Mac (copiar archivos al Studio)

Cuando tengas un Studio corriendo, anota teamspace `ddrestrepo/nombre-studio`:

```bash
source scripts/lightning_login.sh
lightning cp -r src ddrestrepo/MI-STUDIO:/content/plant-disease/src
```

Si `lightning studio list` falla por menú interactivo, usa solo la web UI.
