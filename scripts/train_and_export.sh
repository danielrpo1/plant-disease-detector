#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a pipeline.log) 2>&1
echo "=== $(date) Train+export en GPU ==="
pip install -q -r requirements.txt
python -m src.train --data_dir data/plantvillage --fast --wandb_off
CKPT=$(ls -t checkpoints/*.ckpt | head -1)
python -m src.export_onnx --checkpoint "$CKPT" --class_mapping checkpoints/class_mapping.json --output artifacts/model.onnx
echo "DONE" > pipeline_status.txt
ls -la artifacts/
