"""
API local para demo sin AWS (Plan B del día 1).

  pip install flask
  python scripts/local_api.py --onnx artifacts/model.onnx --meta artifacts/model.meta.json

Webapp: en config.js → window.API_URL = "http://127.0.0.1:8000/predict"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Raíz del repo en PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lambda"))

from flask import Flask, jsonify, request
from handler import predict

app = Flask(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import os

    os.environ["MODEL_PATH"] = str(Path(args.onnx).resolve())
    os.environ["META_PATH"] = str(Path(args.meta).resolve())

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict_route():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(force=True)
        import base64

        image_bytes = base64.b64decode(body["image"])
        return jsonify(predict(image_bytes))

    print(f"API local http://127.0.0.1:{args.port}/predict")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
