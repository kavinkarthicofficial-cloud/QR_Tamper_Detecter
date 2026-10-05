"""Photo-quality gate: is this photo taken under conditions the poster's model was trained on?

The autoencoder only knows the capture conditions of the enrolment shots. A genuine photo that is much blurrier,
darker or more over-exposed than anything seen at enrolment can reconstruct badly and look "tampered". We measure
four cheap statistics on the aligned 128x128 patch and compare them with the range seen in the poster's own
enrolment photos. Nothing here ever turns a verdict into GENUINE: the detector only uses it to downgrade a
TAMPERED verdict to "unable to verify, retake the photo".
"""

from __future__ import annotations

import cv2
import numpy as np

METRICS = ("lum", "contrast", "sharp", "hi")
MARGIN = 0.25            # how far outside the enrolment range (as a fraction of that range) still counts as normal

LABELS = {
    "lum_low": "too dark",
    "lum_high": "too bright",
    "contrast_low": "low contrast (washed out or glare)",
    "sharp_low": "blurry",
    "hi_high": "over-exposed (large blown-out areas)",
}


def patch_metrics(x: np.ndarray) -> dict:
    """x: 3xHxW float32 RGB in [0, 1] (the autoencoder input) -> quality statistics."""
    g = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).astype(np.float32)
    contrast = float(g.std())
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    return {"lum": float(g.mean()), "contrast": contrast,
            "sharp": lap / (contrast * contrast + 1e-6),       # edge energy relative to contrast
            "hi": float((g > 0.98).mean())}                     # fraction of blown-out pixels


def reference_ranges(X: np.ndarray, lo_q: float = 0.5, hi_q: float = 99.5) -> dict:
    """Range of each statistic over the genuine enrolment/training patches X (N x 3 x H x W)."""
    rows = [patch_metrics(x) for x in X]
    return {m: [float(np.percentile([r[m] for r in rows], lo_q)),
                float(np.percentile([r[m] for r in rows], hi_q))] for m in METRICS}


def check(metrics: dict, ref: dict | None, margin: float = MARGIN) -> list[str]:
    """Quality problems of one photo relative to the poster's enrolment range (empty list = fine)."""
    if not ref:
        return []
    issues = []
    for m, bad_low, bad_high in (("lum", "lum_low", "lum_high"), ("contrast", "contrast_low", None),
                                 ("sharp", "sharp_low", None), ("hi", None, "hi_high")):
        if m not in ref:
            continue
        lo, hi = ref[m]
        pad = margin * max(hi - lo, 1e-6)
        if bad_low and metrics[m] < lo - pad:
            issues.append(bad_low)
        if bad_high and metrics[m] > hi + pad:
            issues.append(bad_high)
    return issues


def describe(issues: list[str]) -> str:
    return ", ".join(LABELS[i] for i in issues)
