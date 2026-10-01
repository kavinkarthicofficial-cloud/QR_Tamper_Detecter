"""Re-calibrate the decision threshold of an existing checkpoint from a folder of genuine photos.

Use this when deploying at a new poster/location: take ~30-50 photos of the genuine
poster with different phones and lighting, put them in a folder, and run
    python calibrate.py --genuine path/to/photos
No retraining and no tampered photos are needed.
"""

import argparse

import torch

from qrguard import config
from qrguard.calibration import calibrate
from qrguard.data import load_folder
from qrguard.detector import QRGuard


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genuine", default=str(config.DATA_DIR / "synthetic" / "val" / "genuine"))
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--k-sigma", type=float, default=config.K_SIGMA)
    args = ap.parse_args()

    guard = QRGuard(args.checkpoint)
    X, _, failed = load_folder(config.Path(args.genuine))
    if len(X) < 10:
        raise SystemExit(f"Only {len(X)} usable photos (no QR in {len(failed)}); need at least 10")
    thresholds, stats = calibrate(guard.model, X, guard.device, args.k_sigma)

    ckpt = guard.ckpt
    old = ckpt.get("thresholds", {})
    ckpt["thresholds"] = thresholds
    ckpt["threshold"] = thresholds[ckpt["score_mode"]]
    ckpt["stats"] = {**ckpt.get("stats", {}), **stats, "calibrated_on": args.genuine}
    torch.save(ckpt, args.checkpoint)
    print(f"{len(X)} genuine photos (no QR found in {len(failed)})")
    for mode, thr in thresholds.items():
        print(f"[{mode:5s}] threshold {old.get(mode, float('nan')):.6f} -> {thr:.6f}")
    print(f"saved to {args.checkpoint} (default mode: {ckpt['score_mode']})")


if __name__ == "__main__":
    main()
