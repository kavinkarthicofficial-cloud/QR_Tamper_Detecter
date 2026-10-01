"""Generate the synthetic QRGuard dataset.

Layout (real photos can use the same layout):
    data/<name>/train/genuine/*.jpg
    data/<name>/val/genuine/*.jpg
    data/<name>/test/genuine/*.jpg
    data/<name>/test/tampered/<tamper_type>_*.jpg

Also writes printable assets for a physical demo:
    data/<name>/poster_genuine.png, data/<name>/sticker_malicious.png
"""

import argparse
import random

import cv2
from tqdm import tqdm

from qrguard import synth
from qrguard.config import DATA_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="synthetic")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test-genuine", type=int, default=200)
    ap.add_argument("--test-tampered", type=int, default=100, help="per tamper type")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = DATA_DIR / args.name
    poster = synth.make_genuine_poster()
    root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "poster_genuine.png"), poster)
    cv2.imwrite(str(root / "sticker_malicious.png"), synth.make_sticker_png(rng))

    jobs = [("train/genuine", None, args.train), ("val/genuine", None, args.val),
            ("test/genuine", None, args.test_genuine)]
    jobs += [("test/tampered", kind, args.test_tampered) for kind in synth.TAMPER_TYPES]

    for sub, kind, n in jobs:
        out = root / sub
        out.mkdir(parents=True, exist_ok=True)
        prefix = kind or "genuine"
        for i in tqdm(range(n), desc=f"{sub} [{prefix}]"):
            src = poster if kind is None else synth.tamper(poster, kind, rng)
            photo = synth.simulate_photo(src, rng)
            cv2.imwrite(str(out / f"{prefix}_{i:04d}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"Dataset written to {root}")


if __name__ == "__main__":
    main()
