"""Enrol your own QR poster so QRGuard can check it for tampering.

    python enroll.py my_poster.jpg --name canteen          # 1+ photos of the GENUINE poster
    python detect.py --poster canteen new_photo.jpg         # later: check any photo of it

Give photos of the poster in its known-good state (one is enough; a few photos from
different angles/lighting make it more robust). Takes 1–3 minutes.
"""

import argparse
import re
import time

from qrguard import config
from qrguard.enrollment import enroll
from qrguard.model import save_checkpoint
from qrguard.preprocess import load_image

POSTER_DIR = config.CHECKPOINT_DIR / "posters"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="photo(s) of the genuine, untampered poster")
    ap.add_argument("--name", required=True, help="short name for this poster, e.g. canteen")
    args = ap.parse_args()

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.name).strip("_")
    if not name:
        raise SystemExit("--name must contain letters or digits")
    t0 = time.time()
    print(f"Enrolling '{name}' from {len(args.photos)} photo(s); this takes 1–3 minutes...")
    model, thresholds, stats = enroll([load_image(p) for p in args.photos])
    out = POSTER_DIR / f"{name}.pt"
    save_checkpoint(out, model, thresholds, {**stats, "photos": args.photos, "name": name})

    print(f"Saved {out}  ({stats['n_photos']} usable photo(s), {stats['epochs_run']} epochs, "
          f"{time.time() - t0:.0f}s)")
    if stats["payloads"]:
        print(f"Genuine QR payload: {', '.join(stats['payloads'])}")
    print(f"\nCheck a photo:   python3 detect.py --poster {name} photo.jpg")
    print("Or in the app:   python3 app.py   (pick the poster from the list)")


if __name__ == "__main__":
    main()
