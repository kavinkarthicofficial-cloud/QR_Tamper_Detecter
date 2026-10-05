"""Write the per-poster reference data (photo-quality range + enrolled QR text) next to a checkpoint.

Posters enrolled with enroll.py / the web app already store this inside their checkpoint. The demo model was trained
before it existed, so this script records it in a small "<checkpoint>.meta.json" that the detector reads:

    python make_meta.py                      # demo model, from the synthetic training photos
    python make_meta.py --genuine DIR --payload "https://example.org/x" --checkpoint checkpoints/mine.pt
"""

import argparse
import json
from pathlib import Path

from qrguard import config, quality, synth
from qrguard.data import load_folder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--genuine", default=str(config.DATA_DIR / "synthetic" / "train" / "genuine"),
                    help="folder of genuine photos of the poster (the ones the model was trained on)")
    ap.add_argument("--payload", action="append", default=None,
                    help="the genuine QR text (repeatable); default: the synthetic demo poster's UPI link")
    args = ap.parse_args()

    X, _, failed = load_folder(Path(args.genuine))
    if len(X) < 20:
        raise SystemExit(f"need at least 20 usable genuine photos, found {len(X)} (no QR in {len(failed)})")
    meta = {"quality": quality.reference_ranges(X), "payloads": args.payload or [synth.GENUINE_PAYLOAD],
            "n_photos": int(len(X))}
    out = Path(args.checkpoint).with_suffix(".meta.json")
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for m, (lo, hi) in meta["quality"].items():
        print(f"  {m:9s} {lo:8.4f} .. {hi:8.4f}")


if __name__ == "__main__":
    main()
