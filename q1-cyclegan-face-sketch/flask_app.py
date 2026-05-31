"""
Flask UI for CycleGAN face<->sketch translation.

Endpoints:
    GET  /                 - upload form + webcam capture widget
    POST /translate        - accepts a file or dataURL, returns base64 result
    GET  /status           - reports weight-load status

Auto-detects whether the input image is a sketch or a real face and routes
through the correct generator. Users can also override via the UI.

Run:
    cd Q1_CycleGAN
    python flask_app.py --weights weights/generators.pt
"""
import argparse
import base64
import io
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from inference import detect_sketch, load_generators, translate


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

_STATE = {"G_AB": None, "G_BA": None, "device": None, "weights": None, "err": None}


def _init_model(weights_path):
    try:
        G_AB, G_BA, device = load_generators(weights_path)
        _STATE.update(G_AB=G_AB, G_BA=G_BA, device=device,
                      weights=str(weights_path), err=None)
        print(f"[flask_app] Loaded weights from {weights_path} on {device}")
    except Exception as e:
        _STATE.update(err=str(e))
        print(f"[flask_app] WARNING: could not load weights ({e}).")
        print("             UI will still start; upload will return an error until weights are present.")


def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_input(req):
    """Returns a PIL.Image from either a file upload or a base64 dataURL."""
    if "file" in req.files and req.files["file"].filename:
        return Image.open(req.files["file"].stream).convert("RGB")
    # `req.get_json(silent=True)` returns None instead of raising 415 when
    # the Content-Type isn't JSON, so we can fall through to form/data URL.
    body = req.get_json(silent=True) or {}
    data_url = req.form.get("dataurl") or body.get("dataurl")
    if data_url and "," in data_url:
        header, b64 = data_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    raise ValueError("No image provided.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "loaded": _STATE["G_AB"] is not None,
        "weights": _STATE["weights"],
        "device": str(_STATE["device"]) if _STATE["device"] else None,
        "error": _STATE["err"],
    })


@app.route("/translate", methods=["POST"])
def translate_route():
    if _STATE["G_AB"] is None:
        return jsonify({"error": f"Model not loaded: {_STATE['err']}"}), 503
    try:
        img = _parse_input(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    override = request.form.get("mode") or (request.get_json(silent=True) or {}).get("mode")
    force = None
    if override in ("face2sketch", "sketch2face"):
        force = override

    out, label = translate(img, _STATE["G_AB"], _STATE["G_BA"], _STATE["device"], force=force)
    return jsonify({
        "result": _pil_to_b64(out),
        "mode": label,
        "detected_sketch": bool(detect_sketch(img)),  # numpy bool -> py bool for JSON
    })


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str,
                   default=str(Path(__file__).parent / "weights" / "generators.pt"))
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _init_model(args.weights)
    app.run(host=args.host, port=args.port, debug=args.debug)
