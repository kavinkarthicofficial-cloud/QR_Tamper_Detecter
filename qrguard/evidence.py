"""Structured evidence about one analysed photo: the facts the Investigator agent explains.

Everything here is computed deterministically from the detector's output. The agent never changes the verdict;
it only turns this record into a plain-language explanation.
"""

from __future__ import annotations

import difflib
import re
import urllib.parse

import cv2
import numpy as np

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc"}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def safe_text(s: str, n: int = 160) -> str:
    """Untrusted QR text: strip control characters and truncate before it goes anywhere near a prompt."""
    return _CTRL.sub(" ", s or "")[:n]


# --------------------------------------------------------------------------- payload analysis

def _split(payload: str):
    """(kind, host_or_payee) of a decoded QR payload."""
    p = (payload or "").strip()
    try:                                  # QR text is attacker-controlled: malformed input must never raise
        if p.lower().startswith("upi://"):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(p).query)
            return "upi", (q.get("pa", [""])[0]).lower()
        u = urllib.parse.urlparse(p if "://" in p else "//" + p)
        return "url", (u.hostname or "").lower()
    except ValueError:
        return "url", ""


def _registered(host: str) -> str:
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def url_features(payload: str, expected: list[str] | None) -> dict:
    """Cheap, explainable properties of the decoded payload (and how it differs from the enrolled one)."""
    if not payload:
        return {"decoded": False}
    kind, ident = _split(payload)
    f = {"decoded": True, "kind": kind, "identity": safe_text(ident, 80),
         "https": payload.lower().startswith("https://") if kind == "url" else None,
         "shortener": kind == "url" and _registered(ident) in SHORTENERS,
         "punycode": "xn--" in ident,
         "ip_host": bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", ident))}
    if expected:
        exp = [_split(e) for e in expected]
        f["same_as_enrolled"] = payload in expected
        best = max((difflib.SequenceMatcher(None, ident, e[1]).ratio() for e in exp if e[0] == kind), default=0.0)
        f["similarity_to_enrolled"] = round(best, 2)
        f["look_alike"] = (not f["same_as_enrolled"]) and best >= 0.8
    return f


# --------------------------------------------------------------------------- where is the error?

_ROWS, _COLS = ("upper", "middle", "lower"), ("left", "centre", "right")


def hot_region(error: np.ndarray, threshold: float, window: int = 16, margin: float = 0.25) -> dict:
    """Where the window-averaged reconstruction error exceeds the local threshold.

    `error` is the HxW pixel error map of the 128x128 patch (QR in the middle, `margin` of poster around it).
    """
    h, w = error.shape
    pooled = cv2.boxFilter(error.astype(np.float32), -1, (window, window), normalize=True)
    off = window // 2
    pooled = pooled[off:h - off + 1, off:w - off + 1]                 # valid windows only
    hot = pooled > threshold
    peak = float(pooled.max())
    out = {"peak_ratio": round(peak / threshold, 2), "hot_fraction": round(float(hot.mean()), 3)}
    if not hot.any():
        return {**out, "location": None, "in_qr_core": None}
    ys, xs = np.nonzero(hot)
    cy, cx = ys.mean() + off, xs.mean() + off
    core0 = w * margin / (1 + 2 * margin)
    core1 = w - core0
    r, c = min(int(cy / h * 3), 2), min(int(cx / w * 3), 2)
    loc = "centre" if (r, c) == (1, 1) else f"{_ROWS[r]} {_COLS[c]}"
    return {**out, "location": loc, "in_qr_core": bool(core0 <= cy <= core1 and core0 <= cx <= core1)}


# --------------------------------------------------------------------------- the record

def build_evidence(res, poster: str = "") -> dict:
    """Evidence dict for one detector `Result` (JSON-serialisable)."""
    ev = {
        "poster": safe_text(poster, 60),
        "verdict": res.status,                         # genuine | tampered | unverified | no_qr  (decided by the detector)
        "visual": None if res.score is None else {"score": round(res.score, 5), "threshold": round(res.threshold, 5),
                                                 "ratio": round(res.ratio, 2)},
        "decoded_payload": safe_text(res.payload),
        "payload_match": res.payload_match,            # True / False / None (unknown)
        "url": url_features(res.payload, res.expected_payloads),
        "quality": {"issues": list(res.quality_issues), "metrics": res.quality},
        "hot_region": res.hot,
        "reasons": list(res.reasons),
    }
    return ev
