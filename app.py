"""QRGuard web app: photograph a QR poster from a phone and check it for tampering.

    python app.py            # then open http://localhost:8000 (or http://<laptop-ip>:8000 on a phone)
"""

import argparse
import base64
import time

import cv2
from flask import Flask, jsonify, render_template, request

from qrguard import config
from qrguard.data import list_images
from qrguard.detector import QRGuard, annotate_photo, heatmap
from qrguard.preprocess import decode_image_bytes, load_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
guard: QRGuard | None = None
SAMPLE_DIR = config.DATA_DIR / "synthetic" / "test"


def b64(img, size=None) -> str:
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def respond(img):
    t0 = time.perf_counter()
    res = guard.analyze(img)
    ms = (time.perf_counter() - t0) * 1000
    body = {**res.summary(), "latency_ms": round(ms, 1), "photo": b64(annotate_photo(img, res, 720))}
    if res.patch is not None:
        body.update(input=b64(res.patch, 256), recon=b64(res.recon, 256),
                    heat=b64(heatmap(res.error, res.patch, vmax=res.heat_vmax), 256))
    return jsonify(body)


def samples():
    out = []
    for sub, label in [("genuine", "Genuine"), ("tampered", "Tampered")]:
        files = list_images(SAMPLE_DIR / sub)
        step = max(1, len(files) // 3)
        out += [{"path": f"{sub}/{p.name}", "label": f"{label}: {p.stem}"} for p in files[::step][:3]]
    return out


@app.get("/")
def index():
    return render_template("index.html", threshold=guard.threshold, samples=samples())


@app.post("/analyze")
def analyze():
    f = request.files.get("photo")
    if f is None:
        return jsonify(error="no photo uploaded"), 400
    try:
        img = decode_image_bytes(f.read())
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return respond(img)


@app.get("/sample")
def sample():
    rel = request.args.get("path", "")
    path = (SAMPLE_DIR / rel).resolve()
    if SAMPLE_DIR.resolve() not in path.parents or not path.is_file():
        return jsonify(error="unknown sample"), 404
    return respond(load_image(path))


def main():
    global guard
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)  # macOS reserves 5000 for AirPlay
    args = ap.parse_args()
    guard = QRGuard(args.checkpoint)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
