#!/usr/bin/env python3
"""
Genera el sitio de GitHub Pages:
  /           → README renderizado (home del proyecto)
  /app/       → demo ONNX (webapp/)
  /assets/eda/ → figuras del EDA referenciadas en el README
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"
README = ROOT / "README.md"
REPO = "https://github.com/danielrpo1/plant-disease-detector"
BLOB = f"{REPO}/blob/main"

HTML_SHELL = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Plant Disease Detector — EAFIT</title>
  <meta name="description" content="Clasificación de enfermedades en hojas con Deep Learning. Proyecto integrador EAFIT." />
  <link rel="stylesheet" href="pages.css" />
</head>
<body>
  <header class="site-header">
    <div class="wrap">
      <strong>Plant Disease Detector</strong>
      <nav>
        <a href="./">Documentación</a>
        <a class="btn-demo" href="app/">Abrir demo</a>
        <a href="{repo}">Código en GitHub</a>
      </nav>
    </div>
  </header>
  <main class="readme wrap">
{body}
  </main>
  <footer class="site-footer wrap">
    Proyecto integrador · Maestría en Ciencia de Datos · EAFIT
  </footer>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    document.querySelectorAll("pre code.language-mermaid").forEach((el) => {{
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = el.textContent;
      el.parentElement.replaceWith(div);
    }});
    mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
  </script>
</body>
</html>
"""


def _preprocess(md: str) -> str:
    md = md.replace("notebooks/eda_outputs/", "assets/eda/")
    # Enlaces relativos al repo → GitHub (evita 404 en Pages)
    for prefix in (
        "notebooks/",
        "src/",
        "scripts/",
        "webapp/",
        "infra/",
        "lambda/",
        ".github/",
    ):
        md = md.replace(f"]({prefix}", f"]({BLOB}/{prefix}")
    return md


def build() -> None:
    try:
        import markdown
    except ImportError:
        print("Instala markdown: pip install markdown", file=sys.stderr)
        sys.exit(1)

    if not README.is_file():
        print(f"No se encontró {README}", file=sys.stderr)
        sys.exit(1)

    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir()

    raw = README.read_text(encoding="utf-8")
    body = markdown.markdown(
        _preprocess(raw),
        extensions=["extra", "tables", "fenced_code", "nl2br"],
    )
    (SITE / "index.html").write_text(
        HTML_SHELL.format(body=body, repo=REPO),
        encoding="utf-8",
    )
    shutil.copy(ROOT / "site" / "pages.css", SITE / "pages.css")
    (SITE / ".nojekyll").touch()

    eda_src = ROOT / "notebooks" / "eda_outputs"
    if eda_src.is_dir():
        shutil.copytree(eda_src, SITE / "assets" / "eda")

    shutil.copytree(ROOT / "webapp", SITE / "app")

    print(f"✓ Sitio generado en {SITE}")
    print("  /       → README (home)")
    print("  /app/   → demo ONNX")


if __name__ == "__main__":
    build()
