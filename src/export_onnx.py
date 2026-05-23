"""
Exporta el checkpoint Lightning a ONNX para AWS Lambda.

Uso:
  python -m src.export_onnx \\
    --checkpoint checkpoints/efficientnet-epoch=09-val_acc=0.9500.ckpt \\
    --class_mapping checkpoints/class_mapping.json \\
    --output artifacts/model.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from src.model import PlantDiseaseModule


class ExportableModel(nn.Module):
    """Backbone + head en un solo nn.Module para ONNX (sin Lightning)."""

    def __init__(self, backbone: nn.Module, head: nn.Sequential) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_from_checkpoint(ckpt_path: str, num_classes: int) -> ExportableModel:
    """Carga pesos desde .ckpt de Lightning (siempre en CPU para ONNX)."""
    lit = PlantDiseaseModule.load_from_checkpoint(
        ckpt_path, num_classes=num_classes, map_location="cpu"
    )
    exportable = ExportableModel(lit.backbone, lit.head)
    exportable.cpu().eval()
    return exportable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--class_mapping", required=True)
    p.add_argument("--output", default="artifacts/model.onnx")
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mapping = json.loads(Path(args.class_mapping).read_text(encoding="utf-8"))
    num_classes = len(mapping["class_to_idx"])

    model = load_from_checkpoint(args.checkpoint, num_classes)
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # Copiar mapeo junto al modelo para Lambda
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(Path(args.class_mapping).read_text(encoding="utf-8"), encoding="utf-8")

    print(f"ONNX guardado: {out_path}")
    print(f"Mapeo copiado: {meta_path}")


if __name__ == "__main__":
    main()
