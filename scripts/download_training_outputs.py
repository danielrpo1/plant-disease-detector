#!/usr/bin/env python3
"""Descarga notebook ejecutado y salidas de entrenamiento desde Lightning Studio."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

REMOTE = "plant-disease-detector"
FILES = [
    "notebooks/02_entrenamiento_executed.ipynb",
    "notebooks/training_outputs/curvas_entrenamiento.png",
    "notebooks/training_outputs/historial_entrenamiento.csv",
    "notebooks/training_outputs/historial_entrenamiento.json",
]


def main() -> None:
    from lightning_sdk import Studio

    studio = Studio(
        name=os.environ.get("LIGHTNING_STUDIO", "cognitive-bronze-moof"),
        teamspace=os.environ.get("LIGHTNING_TEAMSPACE", "neural-network-development-project"),
        user=os.environ.get("LIGHTNING_USER", "ddrestrepo"),
        create_ok=False,
    )
    for rel in FILES:
        remote = f"{REMOTE}/{rel}"
        local = ROOT / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        studio.download_file(remote, str(local))
        print("✓", local)


if __name__ == "__main__":
    main()
