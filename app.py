"""QRGuard web app: photograph a QR poster from a phone and check it for tampering.

    python app.py            # then open http://localhost:8000 (or http://<laptop-ip>:8000 on a phone)

Pick which poster you are checking, or enrol a new one from a photo of its genuine state.
"""

import argparse
import base64
import json
import re
import threading
import time
import uuid
from collections import OrderedDict

import cv2
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from qrguard import config
from qrguard.agent import investigate
from qrguard.data import list_images
from qrguard.detector import QRGuard, annotate_photo, available_posters, heatmap
from qrguard.evidence import build_evidence
from qrguard.model import save_checkpoint
from qrguard.preprocess import decode_image_bytes, find_qr, load_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
SAMPLE_DIR = config.DATA_DIR / "synthetic" / "test"
DEMO = "demo-canteen"
POSTER_DIR = config.CHECKPOINT_DIR / "posters"

_guards: dict[str, QRGuard] = {}
_jobs: dict[str, dict] = {}
_enroll_lock = threading.Lock()
_evidence: OrderedDict[str, dict] = OrderedDict()     # last analysed results, for the Investigator panel


def remember(ev: dict) -> str:
    eid = uuid.uuid4().hex[:12]
    _evidence[eid] = ev
    while len(_evidence) > 50:
        _evidence.popitem(last=False)
    return eid


def get_guard(name: str) -> QRGuard:
    posters = available_posters()
    if name not in posters:
        raise KeyError(name)
    if name not in _guards:
        _guards[name] = QRGuard(posters[name])
    return _guards[name]


def b64(img, size=None) -> str:
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def respond(img, poster: str):
    try:
        guard = get_guard(poster)
    except KeyError:
        return jsonify(error=f"unknown poster '{poster}'"), 404
    t0 = time.perf_counter()
    res = guard.analyze(img)
    ms = (time.perf_counter() - t0) * 1000
    body = {**res.summary(), "poster": poster, "latency_ms": round(ms, 1),
            "photo": b64(annotate_photo(img, res, 720)), "evidence_id": remember(build_evidence(res, poster))}
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
    posters = list(available_posters())
    default = request.args.get("poster") or (DEMO if DEMO in posters else (posters[0] if posters else ""))
    return render_template("index.html", posters=posters, selected=default,
                           samples=samples() if DEMO in posters else [])


@app.get("/posters")
def posters_list():
    return jsonify(posters=list(available_posters()))


@app.post("/analyze")
def analyze():
    f = request.files.get("photo")
    if f is None:
        return jsonify(error="no photo uploaded"), 400
    try:
        img = decode_image_bytes(f.read())
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return respond(img, request.form.get("poster", DEMO))


@app.get("/sample")
def sample():
    rel = request.args.get("path", "")
    path = (SAMPLE_DIR / rel).resolve()
    if SAMPLE_DIR.resolve() not in path.parents or not path.is_file():
        return jsonify(error="unknown sample"), 404
    return respond(load_image(path), DEMO)


@app.post("/explain")
def explain():
    """The Investigator agent explains a result that /analyze or /sample just produced (it cannot change the verdict)."""
    ev = _evidence.get((request.get_json(silent=True) or {}).get("evidence_id", ""))
    if ev is None:
        return jsonify(error="unknown or expired result - analyse the photo again"), 404
    return jsonify(investigate(ev))


# ----------------------------------------------------------------- results page
def _load(*parts):
    p = config.RESULTS_DIR.joinpath(*parts)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


@app.get("/results")
def results_page():
    data = {"main": _load("metrics.json"), "fresh": _load("fresh_test", "metrics.json"),
            "web": _load("web", "metrics.json"), "baseline": _load("baseline_supervised.json"),
            "robust": _load("robustness.json"), "robust_web": _load("robustness_web010.json"),
            "agent": _load("agent_eval.json"),
            "has": {n: (config.RESULTS_DIR / n).is_file() for n in
                    ("confusion_matrix.png", "roc_curve.png", "score_histogram.png", "examples.png",
                     "attacks_gallery.png", "baseline_vs_autoencoder.png", "web/summary.png", "web/examples.png")}}
    return render_template("results.html", d=data)


@app.get("/results/files/<path:name>")
def results_file(name):
    if not name.lower().endswith((".png", ".json")):
        abort(404)
    return send_from_directory(config.RESULTS_DIR, name)


# ----------------------------------------------------------------- enrolment (background job)
def _run_enrollment(job_id: str, name: str, photos):
    from qrguard.enrollment import enroll
    job = _jobs[job_id]
    with _enroll_lock:                     # one training run at a time
        job["state"] = "training"
        try:
            t0 = time.time()
            model, thresholds, stats = enroll(photos)
            save_checkpoint(POSTER_DIR / f"{name}.pt", model, thresholds, {**stats, "name": name})
            _guards.pop(name, None)
            job.update(state="done", seconds=round(time.time() - t0), payloads=stats["payloads"])
        except Exception as e:             # surfaced to the page
            job.update(state="error", error=str(e))


@app.post("/enroll")
def enroll_poster():
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", request.form.get("name", "")).strip("_")[:40]
    files = request.files.getlist("photos")
    if not name:
        return jsonify(error="give the poster a name"), 400
    if name == DEMO:
        return jsonify(error="that name is reserved"), 400
    if not files:
        return jsonify(error="upload at least one photo of the genuine poster"), 400
    try:
        photos = [decode_image_bytes(f.read()) for f in files]
    except ValueError as e:
        return jsonify(error=str(e)), 400
    if not any(find_qr(p)[0] is not None for p in photos):
        return jsonify(error="no QR code found in the photo(s); retake closer and straighter"), 400
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"state": "queued", "name": name}
    threading.Thread(target=_run_enrollment, args=(job_id, name, photos), daemon=True).start()
    return jsonify(job=job_id, name=name)


@app.get("/enroll/<job_id>")
def enroll_status(job_id):
    job = _jobs.get(job_id)
    return (jsonify(job), 200) if job else (jsonify(error="unknown job"), 404)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)  # macOS reserves 5000 for AirPlay
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
