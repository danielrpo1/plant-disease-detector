#!/usr/bin/env python3
"""Sube el proyecto al Studio de Lightning (sin .env)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# Credenciales desde .env si existe
env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

USER = "ddrestrepo"
TEAMSPACE = "neural-network-development-project"
REMOTE_DIR = "plant-disease-detector"
SKIP = {".env", ".env.bak", ".git", "__pycache__", ".DS_Store", "checkpoints", "wandb", "artifacts"}


def should_upload(path: Path) -> bool:
    parts = path.parts
    if any(p in SKIP for p in parts):
        return False
    if path.suffix in {".pyc", ".ckpt", ".onnx"}:
        return False
    return True


def main() -> None:
    from lightning_sdk import Studio
    from lightning_sdk.teamspace import Teamspace

    print(f"Conectando: user={USER}, teamspace={TEAMSPACE}")
    ts = Teamspace(name=TEAMSPACE, user=USER)
    studios = ts.studios
    if not studios:
        print("No hay studios. Creando uno...")
        studio = Studio(name="plant-disease-eafit", teamspace=TEAMSPACE, user=USER)
    else:
        # Usar el primero encendido o el primero de la lista
        studio = None
        for s in studios:
            print(f"  Studio: {s.name} — status={s.status}")
            if str(s.status).lower() in ("running", "on", "ready"):
                studio = s
                break
        if studio is None:
            studio = studios[0]
        print(f"Usando studio: {studio.name}")

    # Subir archivos uno a uno (evita .env)
    files = [p for p in ROOT.rglob("*") if p.is_file() and should_upload(p.relative_to(ROOT))]
    print(f"Subiendo {len(files)} archivos a {REMOTE_DIR}/ ...")

    for local in sorted(files):
        rel = local.relative_to(ROOT)
        remote = f"{REMOTE_DIR}/{rel.as_posix()}"
        try:
            studio.upload_file(str(local), remote_path=remote, progress_bar=False)
            print(f"  OK {rel}")
        except Exception as e:
            print(f"  FAIL {rel}: {e}")

  # Verificar
    out = studio.run(f"ls -la {REMOTE_DIR} && ls {REMOTE_DIR}/src")
    print("\n--- En el Studio ---")
    print(out)


if __name__ == "__main__":
    main()
