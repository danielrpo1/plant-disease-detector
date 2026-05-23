#!/usr/bin/env python3
"""Sube el repo y ejecuta notebooks/02_entrenamiento.ipynb en Lightning Studio."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for env_file in (ROOT / ".env",):
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

USER = os.environ.get("LIGHTNING_USER", "ddrestrepo")
TEAMSPACE = os.environ.get("LIGHTNING_TEAMSPACE", "neural-network-development-project")
STUDIO_NAME = os.environ.get("LIGHTNING_STUDIO", "cognitive-bronze-moof")
REMOTE = "plant-disease-detector"
MARKER = f"{REMOTE}/notebooks/.training_done"


def main() -> None:
    print("1) Subiendo código al Studio...")
    subprocess.run([sys.executable, str(ROOT / "scripts/upload_to_lightning.py")], check=True)

    from lightning_sdk import Studio

    studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USER, create_ok=False)
    print(f"Studio: {STUDIO_NAME} | status={studio.status}")

    sh_local = ROOT / "scripts/run_training_nb.sh"
    studio.upload_file(
        str(sh_local),
        remote_path=f"{REMOTE}/scripts/run_training_nb.sh",
        progress_bar=False,
    )

    print("2) Lanzando entrenamiento en GPU (puede tardar ~30–90 min)...")
    studio.run(
        f"chmod +x {REMOTE}/scripts/run_training_nb.sh\n"
        f"rm -f {MARKER}\n"
        f"nohup bash {REMOTE}/scripts/run_training_nb.sh > /tmp/train_nb.log 2>&1 &\n"
        "sleep 3\nhead -15 /tmp/train_nb.log 2>/dev/null || true"
    )

    print("3) Esperando finalización (poll cada 90s)...")
    for i in range(120):  # hasta 3 h
        time.sleep(90)
        done = studio.run(f"test -f {MARKER} && echo DONE || echo RUNNING").strip()
        tail = studio.run(f"tail -8 {REMOTE}/notebooks/training_run.log 2>/dev/null || tail -5 /tmp/train_nb.log")
        print(f"[{(i+1)*90//60} min] {done}")
        print(tail)
        if done == "DONE":
            print(studio.run(f"ls -lh {REMOTE}/notebooks/training_outputs/"))
            print("\n✓ Notebook ejecutado en Lightning.")
            return
        if "Error" in tail and "Traceback" in studio.run(f"grep -c Traceback {REMOTE}/notebooks/training_run.log 2>/dev/null || echo 0"):
            print("Falló. Log completo en Studio.")
            sys.exit(1)

    print("Timeout. Revisa training_run.log en el Studio.")
    sys.exit(1)


if __name__ == "__main__":
    main()
