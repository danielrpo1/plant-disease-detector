#!/usr/bin/env python3
"""Sube solo lo necesario para ejecutar 02_entrenamiento.ipynb."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for line in (ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []:
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

USER = os.environ.get("LIGHTNING_USER", "ddrestrepo")
TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE", "neural-network-development-project")
STUDIO_NAME = os.environ.get("LIGHTNING_STUDIO", "cognitive-bronze-moof")
REMOTE = "plant-disease-detector"

PATHS = [
    "requirements.txt",
    "scripts/run_training_nb.sh",
    "scripts/download_dataset_hf.py",
    "notebooks/02_entrenamiento.ipynb",
]
PATHS += [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")]


def main() -> None:
    from lightning_sdk import Studio

    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USER, create_ok=False)
    print(f"Studio {STUDIO_NAME} — subiendo {len(PATHS)} archivos")
    for rel in sorted(set(PATHS)):
        local = ROOT / rel
        if not local.is_file():
            continue
        remote = f"{REMOTE}/{Path(rel).as_posix()}"
        studio.upload_file(str(local), remote_path=remote, progress_bar=False)
        print("OK", rel)
    print(studio.run(f"ls -la {REMOTE}/notebooks/02_entrenamiento.ipynb {REMOTE}/src/train.py"))


if __name__ == "__main__":
    main()
