"""Check one or more poster photos for QR tampering.

    python detect.py photo.jpg [more.jpg ...] [--save-dir results/detections]

Exit code is 1 if any photo is flagged as tampered (or has no QR), else 0.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from qrguard import config
from qrguard.detector import QRGuard, annotate_photo, result_panel
from qrguard.preprocess import load_image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--threshold", type=float, default=None, help="override the calibrated threshold")
    ap.add_argument("--save-dir", default=str(config.RESULTS_DIR / "detections"))
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    args = ap.parse_args()

    guard = QRGuard(args.checkpoint, threshold=args.threshold)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    flagged = False
    out = []
    for path in args.images:
        img = load_image(path)
        res = guard.analyze(img)
        flagged |= res.status != "genuine"
        summary = {"image": path, **res.summary()}
        out.append(summary)

        photo = annotate_photo(img, res)
        panel = result_panel(res, tile=photo.shape[1] // 3)
        canvas = np.vstack([photo[:, :panel.shape[1]], panel]) if res.patch is not None else photo
        dst = save_dir / f"{Path(path).stem}_qrguard.jpg"
        cv2.imwrite(str(dst), canvas)

        if not args.json:
            if res.status == "no_qr":
                print(f"[NO QR ] {path}: no QR code found -- retake the photo closer / straighter")
            else:
                tag = "TAMPER" if res.status == "tampered" else "OK    "
                print(f"[{tag}] {path}: score {res.score:.5f} vs threshold {res.threshold:.5f} "
                      f"({res.ratio:.1f}x)  payload={res.payload or '<not decoded>'}  -> {dst}")
    if args.json:
        print(json.dumps(out, indent=2))
    raise SystemExit(1 if flagged else 0)


if __name__ == "__main__":
    main()
