"""Draw one figure that shows how the tampered inputs are made and how QRGuard sees them.

Rows: the genuine poster, then each of the four sticker attacks. Columns: the simulated phone photo (QR outlined),
the aligned input patch, the autoencoder's reconstruction, and the error heatmap. Writes results/attacks_gallery.png.

    python make_gallery.py [--checkpoint checkpoints/qrguard_ae.pt] [--seed 3]
"""

import argparse
import random

import cv2
import numpy as np

from qrguard import config, synth
from qrguard.detector import QRGuard, annotate_photo, result_panel

DESCRIPTION = {
    None: "genuine poster",
    "aligned_overlay": "aligned_overlay: same-size sticker exactly over the QR",
    "loose_overlay": "loose_overlay: hand-placed, larger, rotated, off-white paper, shadow",
    "branded_sticker": "branded_sticker: large printed sticker with its own coloured frame",
    "partial_patch": "partial_patch: small patch rewriting part of the QR modules",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "attacks_gallery.png"))
    args = ap.parse_args()

    guard = QRGuard(args.checkpoint)
    poster = synth.make_genuine_poster()
    rng = random.Random(args.seed)
    tile = 200
    rows = []
    for kind in (None, *synth.TAMPER_TYPES):
        src = poster if kind is None else synth.tamper(poster, kind, rng)
        photo = synth.simulate_photo(src, rng)
        res = guard.analyze(photo)
        left = cv2.resize(annotate_photo(photo, res, 400), (tile, tile), interpolation=cv2.INTER_AREA)
        panel = result_panel(res, tile=tile)
        body = np.hstack([left, panel])
        bar = np.full((30, body.shape[1], 3), 255, np.uint8)
        colour = {"genuine": (40, 140, 40), "tampered": (40, 40, 200)}.get(res.status, (0, 120, 220))
        ratio = "" if res.score is None else f"  ({res.ratio:.1f}x threshold)"
        cv2.putText(bar, f"{DESCRIPTION[kind]}  ->  {res.status.upper()}{ratio}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, body]))
        print(f"{kind or 'genuine':16s} -> {res.status:10s} {ratio}")
    head = np.full((26, rows[0].shape[1], 3), 245, np.uint8)
    for x, t in zip(range(0, 4 * tile, tile), ["phone photo", "aligned input", "reconstruction", "error heatmap"]):
        cv2.putText(head, t, (x + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    out = config.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), np.vstack([head, *rows]))
    print(f"written {out}")


if __name__ == "__main__":
    main()
