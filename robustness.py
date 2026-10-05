"""Robustness of the verdicts to ordinary photo problems, and what the photo-quality gate changes.

Takes genuine test photos, degrades them the way real phone photos get degraded (darker, brighter, blurry, heavy
JPEG, noise, rotation, low resolution) and reports, per condition, how the full pipeline answers:

    genuine    correct
    tampered   a FALSE ALARM on a genuine photo
    unverified "retake the photo" (the quality gate caught it)
    no_qr      the code could not be found

It also reports the other side of the trade-off: how many TAMPERED photos the gate turns into "retake" (clean and
degraded), and how many are wrongly accepted as genuine. Writes results/robustness.json.

    python robustness.py [--checkpoint checkpoints/qrguard_ae.pt] [--data data/synthetic] [--n 60]
"""

import argparse
import json
import random
from collections import Counter

import cv2
import numpy as np

from qrguard import config, quality, synth
from qrguard.data import list_images
from qrguard.detector import QRGuard
from qrguard.preprocess import find_qr, load_image


def _jpeg(q):
    return lambda im: cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _noise(sd):
    return lambda im: np.clip(im + np.random.default_rng(0).normal(0, sd, im.shape), 0, 255).astype(np.uint8)


def _rotate(a):
    def f(im):
        h, w = im.shape[:2]
        return cv2.warpAffine(im, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return f


CONDITIONS = {
    "clean": lambda im: im,
    "darker x0.6": lambda im: np.clip(im * 0.6, 0, 255).astype(np.uint8),
    "brighter x1.4": lambda im: np.clip(im * 1.4, 0, 255).astype(np.uint8),
    "blur sigma 1.5": lambda im: cv2.GaussianBlur(im, (0, 0), 1.5),
    "blur sigma 2.5": lambda im: cv2.GaussianBlur(im, (0, 0), 2.5),
    "blur sigma 4": lambda im: cv2.GaussianBlur(im, (0, 0), 4),
    "JPEG quality 20": _jpeg(20),
    "noise sd 16": _noise(16),
    "rotated 12 deg": _rotate(12),
    "resized to 40%": lambda im: cv2.resize(im, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA),
}
TAMPER_CONDITIONS = ("clean", "darker x0.6", "brighter x1.4", "blur sigma 2.5", "JPEG quality 20")


def status_for(res, ref, margin):
    """Re-derive the status for a different quality margin (the detector uses quality.MARGIN)."""
    if res.status == "no_qr":
        return "no_qr"
    if res.payload_match is False:
        return "tampered"
    if res.score <= res.threshold:
        return "genuine"
    return "unverified" if quality.check(res.quality, ref, margin) else "tampered"


def run(guard, items, fn):
    """items: image paths or BGR arrays."""
    return [guard.analyze(fn(load_image(x) if not isinstance(x, np.ndarray) else x)) for x in items]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--data", default=str(config.DATA_DIR / "synthetic"))
    ap.add_argument("--n", type=int, default=60, help="photos per condition")
    ap.add_argument("--real-photo", default=None,
                    help="a real photo of an enrolled poster: simulated genuine/tampered shots of it are used "
                         "instead of the synthetic test folders (use with --checkpoint checkpoints/posters/NAME.pt)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "robustness.json"))
    args = ap.parse_args()

    guard = QRGuard(args.checkpoint)
    if not guard.quality_ref:
        raise SystemExit("this checkpoint has no photo-quality reference; run: python make_meta.py")
    if args.real_photo:
        photo = load_image(args.real_photo)
        corners, _ = find_qr(photo)
        if corners is None:
            raise SystemExit("no QR found in --real-photo")
        poster, box, _ = synth.rectify_real_photo(photo, corners)
        rng = random.Random(123)
        shot = lambda src: synth.simulate_photo(src, rng, out_size=640, qr_box=box)
        gen = [shot(poster) for _ in range(args.n)]
        tam = [shot(synth.tamper(poster, synth.TAMPER_TYPES[i % 4], rng, box)) for i in range(args.n)]
    else:
        data = config.Path(args.data)
        gen = list_images(data / "test" / "genuine")
        tam = list_images(data / "test" / "tampered")
        gen = gen[:: max(1, len(gen) // args.n)][: args.n]
        tam = tam[:: max(1, len(tam) // args.n)][: args.n]
    margins = [0.0, 0.1, quality.MARGIN, 0.5, 1.0]

    report = {"checkpoint": str(args.checkpoint), "n_per_condition": args.n, "margin_used": quality.MARGIN,
              "genuine": {}, "tampered": {}, "margin_sweep": {}}
    sweep = {m: {"genuine_false_alarm_caught": 0, "genuine_false_alarm_total": 0, "tampered_downgraded": 0,
                 "tampered_total": 0} for m in margins}

    print(f"{'GENUINE photos, degraded':26s} {'found':>5s} {'genuine':>8s} {'FALSE ALARM':>12s} {'retake':>7s}   (false alarms without the gate)")
    for name, fn in CONDITIONS.items():
        results = run(guard, gen, fn)
        c = Counter(r.status for r in results)
        found = len(results) - c["no_qr"]
        no_gate_fa = c["tampered"] + c["unverified"]
        report["genuine"][name] = {"found": found, "no_qr": c["no_qr"], "genuine": c["genuine"], "tampered": c["tampered"],
                                   "unverified": c["unverified"], "false_alarm_without_gate": no_gate_fa}
        for m in margins:
            for r in results:
                if r.status != "no_qr" and r.score > r.threshold:
                    sweep[m]["genuine_false_alarm_total"] += 1
                    sweep[m]["genuine_false_alarm_caught"] += status_for(r, guard.quality_ref, m) == "unverified"
        pct = lambda k: f"{100 * k / max(found, 1):5.1f}%"
        print(f"{name:26s} {found:5d} {pct(c['genuine']):>8s} {pct(c['tampered']):>12s} {pct(c['unverified']):>7s}   {pct(no_gate_fa)}")

    print(f"\n{'TAMPERED photos':26s} {'found':>5s} {'TAMPERED':>8s} {'MISSED':>12s} {'retake':>7s}")
    for name in TAMPER_CONDITIONS:
        results = run(guard, tam, CONDITIONS[name])
        c = Counter(r.status for r in results)
        found = len(results) - c["no_qr"]
        report["tampered"][name] = {"found": found, "no_qr": c["no_qr"], "tampered": c["tampered"],
                                    "missed_as_genuine": c["genuine"], "unverified": c["unverified"]}
        for m in margins:
            for r in results:
                if r.status != "no_qr" and r.score > r.threshold:
                    sweep[m]["tampered_total"] += 1
                    sweep[m]["tampered_downgraded"] += status_for(r, guard.quality_ref, m) == "unverified"
        pct = lambda k: f"{100 * k / max(found, 1):5.1f}%"
        print(f"{name:26s} {found:5d} {pct(c['tampered']):>8s} {pct(c['genuine']):>12s} {pct(c['unverified']):>7s}")

    print("\nmargin sweep (share of flagged photos sent to 'retake'):")
    for m in margins:
        s = sweep[m]
        fa = s["genuine_false_alarm_caught"] / max(s["genuine_false_alarm_total"], 1)
        td = s["tampered_downgraded"] / max(s["tampered_total"], 1)
        report["margin_sweep"][str(m)] = {**s, "false_alarms_caught": round(fa, 3), "tampered_downgraded": round(td, 3)}
        tag = "  <- used" if m == quality.MARGIN else ""
        print(f"  margin {m:4.2f}: false alarms caught {100 * fa:5.1f}%   tampered photos sent to retake {100 * td:5.1f}%{tag}")

    out = config.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
