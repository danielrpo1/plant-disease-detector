#!/usr/bin/env bash
# Ejecuta 02_entrenamiento.ipynb en Lightning Studio (background).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MARKER="notebooks/.training_done"
rm -f "$MARKER"

pip install -q -r requirements.txt jupyter nbconvert ipykernel

cd notebooks
echo "INICIO $(date -Iseconds)" > training_run.log
jupyter nbconvert --to notebook --execute 02_entrenamiento.ipynb \
  --ExecutePreprocessor.timeout=7200 \
  --output 02_entrenamiento_executed.ipynb >> training_run.log 2>&1
echo "FIN $(date -Iseconds)" >> training_run.log
touch "$MARKER"
