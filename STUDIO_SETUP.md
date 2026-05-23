# Si ves "No such file requirements.txt" en Lightning Studio

## Qué pasó

Estabas en la carpeta **equivocada** del Studio (vacía). El código vive en:

`plant-disease-detector/` (dentro del Studio, no en la raíz `/`).

## Pasos en el terminal del Studio

```bash
cd ~/plant-disease-detector
# o a veces:
cd /teamspace/studios/this_studio/plant-disease-detector

ls
# Debes ver: requirements.txt   src/   README.md
```

Si `ls` no muestra eso, sube el zip desde tu Mac:

1. Archivo: `proyecto integrador/plant-disease-detector.zip`
2. En Lightning: panel **Files** → arrastra el zip → clic derecho → **Unzip**
3. Luego:

```bash
cd plant-disease-detector
pip install -r requirements.txt
```

## Entrenar (cuando tengas el dataset)

```bash
# Encuentra train/ y valid/
find /teamspace -type d -name train 2>/dev/null | head -5

export DATA_DIR="/ruta/que/muestre/find"
python -m src.train --data_dir "$DATA_DIR" --fast --wandb_off
```

`RUTA_CON_train_y_valid` era solo un ejemplo — debes poner la ruta real.
