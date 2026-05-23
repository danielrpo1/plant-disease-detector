"""Ejecuta el EDA y guarda figuras (para Studio sin Jupyter)."""
from pathlib import Path

# Ejecutar notebook lógica vía nbconvert si existe, si no importar módulo eda
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
nb = ROOT / "notebooks" / "01_EDA.ipynb"
if nb.exists():
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb),
        ],
        cwd=str(ROOT),
    )
else:
    print("Falta notebooks/01_EDA.ipynb")
