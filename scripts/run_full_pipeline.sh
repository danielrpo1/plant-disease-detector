#!/usr/bin/env bash
# Pipeline completo en Lightning Studio (log: pipeline.log)
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date) Inicio pipeline ==="
pip install -q -r requirements.txt

echo "=== Descarga dataset (50 img/clase para rapidez) ==="
python scripts/download_dataset_hf.py --out data/plantvillage --max-per-class 50

echo "=== Entrenamiento --fast ==="
python -m src.train --data_dir data/plantvillage --fast --wandb_off

echo "=== Export ONNX ==="
CKPT=$(ls -t checkpoints/*.ckpt | head -1)
python -m src.export_onnx \
  --checkpoint "$CKPT" \
  --class_mapping checkpoints/class_mapping.json \
  --output artifacts/model.onnx

echo "DONE" > pipeline_status.txt
echo "=== $(date) Pipeline terminado ==="
ls -la artifacts/
