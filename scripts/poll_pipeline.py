#!/usr/bin/env python3
"""Consulta estado del pipeline en Lightning Studio."""
import os
import sys
import time
from pathlib import Path

for line in (Path(__file__).parents[1] / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from lightning_sdk import Studio

studio = Studio(
    name="cognitive-bronze-moof",
    teamspace="neural-network-development-project",
    user="ddrestrepo",
    create_ok=False,
)

for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 1):
    out = studio.run(
        "cd plant-disease-detector && "
        "(test -f pipeline_status.txt && echo STATUS=DONE || echo STATUS=RUNNING) && "
        "tail -8 pipeline.log 2>/dev/null"
    )
    print(out)
    if "STATUS=DONE" in out:
        break
    time.sleep(60)
