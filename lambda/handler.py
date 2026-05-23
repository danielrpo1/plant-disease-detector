"""
AWS Lambda: recibe imagen base64, devuelve top-3 predicciones.

Variables de entorno:
  MODEL_PATH   → ruta al .onnx (default /var/task/model.onnx)
  META_PATH    → class_mapping JSON
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Carga perezosa (reutilizada entre invocaciones warm)
_SESSION: ort.InferenceSession | None = None
_IDX_TO_CLASS: dict[int, str] = {}
_DISPLAY_ES: dict[str, str] = {}


def _load_display_names() -> dict[str, str]:
    try:
        from labels import DISPLAY_NAMES_ES  # type: ignore

        return DISPLAY_NAMES_ES
    except ImportError:
        return {}


def _get_session() -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        model_path = os.environ.get("MODEL_PATH", "/var/task/model.onnx")
        _SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _SESSION


def _load_meta() -> None:
    global _IDX_TO_CLASS, _DISPLAY_ES
    if _IDX_TO_CLASS:
        return
    meta_path = os.environ.get("META_PATH", "/var/task/model.meta.json")
    data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    _IDX_TO_CLASS = {int(k): v for k, v in data["idx_to_class"].items()}
    _DISPLAY_ES = _load_display_names()


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Mismas reglas que datamodule: resize 224, normalize ImageNet."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis, ...].astype(np.float32)


def _display(class_id: str) -> str:
    return _DISPLAY_ES.get(class_id, class_id.replace("___", " - ").replace("_", " "))


def predict(image_bytes: bytes, top_k: int = 3) -> dict:
    _load_meta()
    session = _get_session()
    inp = _preprocess(image_bytes)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: inp})[0][0]

    # Softmax estable
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    top_idx = np.argsort(probs)[::-1][:top_k]
    predictions = []
    for i in top_idx:
        cls = _IDX_TO_CLASS[int(i)]
        conf = float(probs[i])
        predictions.append(
            {"class": cls, "confidence": round(conf, 4), "display_name": _display(cls)}
        )

    best = predictions[0]
    return {
        "predictions": predictions,
        "top_prediction": best["class"],
        "confidence": best["confidence"],
    }


def lambda_handler(event, context):
    """
    API Gateway suele enviar body JSON:
      { "image": "<base64>" }
    """
    try:
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)

        b64 = body.get("image") or body.get("image_base64")
        if not b64:
            return _response(400, {"error": "Falta campo 'image' en base64"})

        image_bytes = base64.b64decode(b64)
        result = predict(image_bytes)
        return _response(200, result)
    except Exception as e:
        return _response(500, {"error": str(e)})


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        },
        "body": json.dumps(body),
    }
