#!/usr/bin/env python3
"""Lanza pipeline en Lightning Studio en background."""
import os
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

# Subir script
local = Path(__file__).parents[1] / "scripts" / "run_full_pipeline.sh"
studio.upload_file(str(local), remote_path="plant-disease-detector/scripts/run_full_pipeline.sh")

cmd = """
cd plant-disease-detector
chmod +x scripts/run_full_pipeline.sh
rm -f pipeline.log pipeline_status.txt
nohup bash scripts/run_full_pipeline.sh > /dev/null 2>&1 &
echo STARTED
sleep 2
head -20 pipeline.log 2>/dev/null || echo 'log iniciando...'
"""
print(studio.run(cmd))
