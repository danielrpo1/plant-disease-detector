#!/usr/bin/env python3
"""Ejecuta entrenamiento en el Studio remoto (requiere dataset en DATA_DIR)."""
import os
import sys

from pathlib import Path

_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

STUDIO = "cognitive-bronze-moof"
TEAMSPACE = "neural-network-development-project"
USER = "ddrestrepo"
DATA_DIR = os.environ.get("DATA_DIR", "")


def main() -> None:
    from lightning_sdk import Studio

    studio = Studio(name=STUDIO, teamspace=TEAMSPACE, user=USER, create_ok=False)

    if not DATA_DIR:
        print("Buscando dataset (carpetas train/)...")
        find_cmd = "find /teamspace /home/zeus -maxdepth 8 -type d -name train 2>/dev/null | head -3"
        out = studio.run(find_cmd).strip()
        if not out:
            print("No hay dataset. Sube el zip de Kaggle al Studio o define DATA_DIR.")
            sys.exit(1)
        # Padre de train/
        first = out.split("\n")[0]
        data_dir = first.rsplit("/train", 1)[0]
        print(f"Usando DATA_DIR={data_dir}")
    else:
        data_dir = DATA_DIR

    cmd = (
        f"cd plant-disease-detector && "
        f"pip install -q -r requirements.txt && "
        f"python -m src.train --data_dir '{data_dir}' --fast --wandb_off"
    )
    print("Iniciando entrenamiento (puede tardar ~1-2 h)...")
    print(studio.run(cmd))


if __name__ == "__main__":
    main()
