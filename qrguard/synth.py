"""Synthetic data: a genuine QR poster, physical sticker tampering, and phone-capture simulation.

Real photographs of a real poster can replace this data at any time (same folder
layout, see README). The simulator exists so the whole pipeline can be trained and
evaluated reproducibly without a photo shoot.

Tampering is applied to the flat poster *before* the simulated photograph, which is
how it happens physically: the attacker sticks paper on the poster, then the
victim photographs it under arbitrary lighting / angle.
"""

from __future__ import annotations

import random
import string

import cv2
import numpy as np

GENUINE_PAYLOAD = "upi://pay?pa=amritacanteen@okaxis&pn=Amrita%20Canteen&cu=INR"
POSTER_W, POSTER_H = 600, 850
QR_BOX = (150, 260, 450, 560)      # x0, y0, x1, y1 of the white QR card on the poster

TAMPER_TYPES = ("aligned_overlay", "loose_overlay", "branded_sticker", "partial_patch")


# --------------------------------------------------------------------------- QR + poster

def make_qr(payload: str, size: int, quiet_modules: int = 2) -> np.ndarray:
    """Render `payload` as a size x size grayscale QR with a small quiet zone."""
    raw = cv2.QRCodeEncoder.create().encode(payload)
    ys, xs = np.where(raw == 0)
    modules = raw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    modules = cv2.copyMakeBorder(modules, *[quiet_modules] * 4, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(modules, (size, size), interpolation=cv2.INTER_NEAREST)


def make_genuine_poster(payload: str = GENUINE_PAYLOAD) -> np.ndarray:
    """A canteen 'Scan to Pay' poster (BGR uint8, POSTER_H x POSTER_W)."""
    p = np.zeros((POSTER_H, POSTER_W, 3), np.uint8)
    # vertical gradient background
    t = np.linspace(0, 1, POSTER_H)[:, None]
    for c, (a, b) in enumerate([(120, 60), (40, 20), (160, 110)]):     # BGR maroon -> dark
        p[..., c] = (a * (1 - t) + b * t).astype(np.uint8)
    # decorative shapes
    cv2.circle(p, (540, 80), 120, (60, 30, 140), -1, cv2.LINE_AA)
    cv2.circle(p, (40, 800), 160, (90, 40, 120), -1, cv2.LINE_AA)
    cv2.rectangle(p, (0, 0), (POSTER_W, 150), (40, 20, 110), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(p, "AMRITA CANTEEN", (60, 70), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(p, "Scan to Pay  |  UPI accepted", (95, 120), font, 0.9, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(p, "SCAN HERE", (205, 235), font, 1.1, (0, 215, 255), 2, cv2.LINE_AA)
    # QR card with coloured frame
    x0, y0, x1, y1 = QR_BOX
    cv2.rectangle(p, (x0 - 14, y0 - 14), (x1 + 14, y1 + 14), (0, 180, 255), -1)
    cv2.rectangle(p, (x0, y0), (x1, y1), (255, 255, 255), -1)
    qr = make_qr(payload, x1 - x0 - 20)
    p[y0 + 10:y1 - 10, x0 + 10:x1 - 10] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    cv2.putText(p, "amritacanteen@okaxis", (150, 610), font, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(p, "GPay  PhonePe  Paytm  BHIM", (110, 680), font, 0.8, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(p, "Do not pay to any other QR", (125, 760), font, 0.7, (180, 180, 255), 1, cv2.LINE_AA)
    return p


def lookalike_payload(rng: random.Random) -> str:
    """A malicious UPI/URL payload resembling the genuine one."""
    choices = [
        "upi://pay?pa=amritacanteen@okaxls&pn=Amrita%20Canteen&cu=INR",
        "upi://pay?pa=amrita.canteen@ybl&pn=Amrita%20Canteen&cu=INR",
        "upi://pay?pa=amritacanteen1@paytm&pn=Amrita%20Canteen&cu=INR",
        "https://amrita-canteen-pay.in/upi",
        "https://bit.ly/" + "".join(rng.choices(string.ascii_letters + string.digits, k=7)),
    ]
    p = rng.choice(choices)
    if rng.random() < 0.5:
        p += "&tn=" + "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10)))
    return p


# --------------------------------------------------------------------------- tampering

def _paste_rotated(dst: np.ndarray, sticker: np.ndarray, center, angle: float, shadow: bool,
                   rng: random.Random):
    """Alpha-composite a rectangular sticker onto dst at `center`, rotated, with optional drop shadow."""
    h, w = sticker.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += (center[0] - w / 2, center[1] - h / 2)
    H, W = dst.shape[:2]
    warped = cv2.warpAffine(sticker, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (W, H), flags=cv2.INTER_LINEAR)
    out = dst.astype(np.float32)
    if shadow:
        off = rng.randint(2, 6)
        sm = np.roll(np.roll(mask, off, 0), off, 1).astype(np.float32)
        sm = cv2.GaussianBlur(sm, (0, 0), rng.uniform(2, 5)) / 255.0 * rng.uniform(0.25, 0.5)
        out *= (1 - sm[..., None])
    a = mask.astype(np.float32)[..., None] / 255.0
    out = out * (1 - a) + warped.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _paper(h, w, rng: random.Random) -> np.ndarray:
    """Slightly tinted sticker paper (attacker's printer/paper never matches exactly)."""
    tint = np.array([rng.uniform(225, 255), rng.uniform(228, 255), rng.uniform(232, 255)])
    return np.tile(tint, (h, w, 1)).astype(np.uint8)


def tamper(poster: np.ndarray, kind: str, rng: random.Random, qr_box=QR_BOX) -> np.ndarray:
    """Return a copy of `poster` with a malicious QR sticker applied.

    `qr_box` is the (x0, y0, x1, y1) square covering the genuine QR and its quiet zone.
    """
    x0, y0, x1, y1 = qr_box
    card = x1 - x0
    pad = round(card / 30)                                  # quiet-zone inset (10 px on the 300 px card)
    payload = lookalike_payload(rng)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    if kind == "aligned_overlay":
        # Hardest case: same size, same place, same white paper -- only the modules differ.
        sticker = np.full((card, card, 3), 255, np.uint8)
        qr = make_qr(payload, card - 2 * pad)
        sticker[pad:card - pad, pad:card - pad] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        return _paste_rotated(poster, sticker, (cx, cy), 0.0, shadow=False, rng=rng)

    if kind == "loose_overlay":
        # Hand-placed sticker: a bit bigger, offset, rotated, off-white paper, casts a shadow.
        s = int(card * rng.uniform(1.0, 1.15))
        sticker = _paper(s, s, rng)
        qsz = int(s * rng.uniform(0.82, 0.92))
        o = (s - qsz) // 2
        sticker[o:o + qsz, o:o + qsz] = cv2.cvtColor(make_qr(payload, qsz), cv2.COLOR_GRAY2BGR)
        sticker = np.minimum(sticker, _paper(s, s, rng))      # tint the white modules too
        c = (cx + rng.uniform(-8, 8), cy + rng.uniform(-8, 8))
        return _paste_rotated(poster, sticker, c, rng.uniform(-5, 5), shadow=True, rng=rng)

    if kind == "branded_sticker":
        # Large printed sticker with its own colourful frame covering the card and its border.
        s = int(card * rng.uniform(1.12, 1.3))
        colour = tuple(int(v) for v in rng.choice([(200, 120, 0), (40, 160, 40), (30, 30, 200), (0, 120, 220)]))
        sticker = np.full((s, s, 3), colour, np.uint8)
        b = int(s * 0.07)
        sticker[b:s - b, b:s - b] = 255
        qsz = s - 2 * b - 16
        sticker[b + 8:b + 8 + qsz, b + 8:b + 8 + qsz] = cv2.cvtColor(make_qr(payload, qsz), cv2.COLOR_GRAY2BGR)
        c = (cx + rng.uniform(-6, 6), cy + rng.uniform(-6, 6))
        return _paste_rotated(poster, sticker, c, rng.uniform(-3, 3), shadow=rng.random() < 0.7, rng=rng)

    if kind == "partial_patch":
        # Subtle module-modification attack: a small printed patch rewrites part of the code
        # (away from the three finder patterns) instead of covering the whole QR.
        qx0, qy0, qsz = x0 + pad, y0 + pad, card - 2 * pad
        fake = make_qr(payload, qsz)
        frac = rng.uniform(0.22, 0.38)                      # patch side as a fraction of the QR
        side = int(qsz * frac)
        lo, hi = int(qsz * 0.3), qsz - side - int(qsz * 0.04)
        px, py = rng.randint(lo, hi), rng.randint(lo, hi)  # centre / lower-right: no finder pattern there
        patch = cv2.cvtColor(fake[py:py + side, px:px + side], cv2.COLOR_GRAY2BGR)
        patch = np.minimum(patch, _paper(side, side, rng))
        c = (qx0 + px + side / 2 + rng.uniform(-2, 2), qy0 + py + side / 2 + rng.uniform(-2, 2))
        return _paste_rotated(poster, patch, c, rng.uniform(-2, 2), shadow=rng.random() < 0.5, rng=rng)

    raise ValueError(f"unknown tamper kind: {kind}")


def make_sticker_png(rng: random.Random) -> np.ndarray:
    """A printable malicious sticker (for physical demos)."""
    x0, _, x1, _ = QR_BOX
    card = x1 - x0
    sticker = np.full((card, card, 3), 255, np.uint8)
    sticker[10:-10, 10:-10] = cv2.cvtColor(make_qr(lookalike_payload(rng), card - 20), cv2.COLOR_GRAY2BGR)
    return sticker


# --------------------------------------------------------------------------- phone capture

def _background(size: int, rng: random.Random) -> np.ndarray:
    base = np.array([rng.uniform(60, 220) for _ in range(3)], np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 15), (size // 8, size // 8, 3))
    tex = cv2.resize(noise.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    return np.clip(base + tex, 0, 255).astype(np.uint8)


def simulate_photo(poster: np.ndarray, rng: random.Random, out_size: int = 900, qr_box=QR_BOX) -> np.ndarray:
    """Photograph `poster` on a wall: perspective, framing, lighting, colour cast, blur, noise, JPEG."""
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    scene = _background(out_size, rng)
    ph, pw = poster.shape[:2]

    # Framing: the photo is centred on the QR region and shows part of the poster around it.
    x0, y0, x1, y1 = qr_box
    qr_side_px = out_size * rng.uniform(0.32, 0.55)            # QR card size in the photo
    scale = qr_side_px / (x1 - x0)
    qcx, qcy = (x0 + x1) / 2, (y0 + y1) / 2
    ccx = out_size / 2 + rng.uniform(-0.1, 0.1) * out_size
    ccy = out_size / 2 + rng.uniform(-0.1, 0.1) * out_size
    ang = np.deg2rad(rng.uniform(-20, 20))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    dst = (src - [qcx, qcy]) * scale @ R.T + [ccx, ccy]
    # perspective (camera not perpendicular to the poster)
    persp = rng.uniform(0.0, 0.12) * qr_side_px
    dst = dst + nrng.uniform(-persp, persp, dst.shape)
    Hm = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(poster, Hm, (out_size, out_size), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), Hm, (out_size, out_size))
    a = (mask.astype(np.float32) / 255.0)[..., None]
    img = scene.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a

    # Lighting: global exposure, linear gradient, vignette, colour temperature.
    yy, xx = np.mgrid[0:out_size, 0:out_size].astype(np.float32) / out_size
    g = rng.uniform(-0.25, 0.25)
    gdir = rng.uniform(0, 2 * np.pi)
    light = 1 + g * ((xx - 0.5) * np.cos(gdir) + (yy - 0.5) * np.sin(gdir))
    r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
    light *= 1 - rng.uniform(0, 0.35) * r2
    light *= rng.uniform(0.65, 1.15)
    cast = np.array([rng.uniform(0.88, 1.12), rng.uniform(0.94, 1.06), rng.uniform(0.88, 1.12)])
    img = img * light[..., None] * cast
    # contrast
    img = (img - 128) * rng.uniform(0.8, 1.15) + 128
    # occasional soft shadow band (a hand / phone shadow)
    if rng.random() < 0.25:
        band = np.clip(1 - np.abs((xx * np.cos(gdir) + yy * np.sin(gdir)) - rng.uniform(0.2, 0.8)) / 0.15, 0, 1)
        band = cv2.GaussianBlur(band, (0, 0), 25)
        img *= 1 - rng.uniform(0.1, 0.3) * band[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Optics / sensor
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.4))
    if rng.random() < 0.15:
        k = rng.choice([5, 7, 9])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1), (k, k))
        kern /= max(kern.sum(), 1e-6)
        img = cv2.filter2D(img, -1, kern)
    img = np.clip(img.astype(np.float32) + nrng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 95)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- real posters

def rectify_real_photo(photo: np.ndarray, corners: np.ndarray, qr_side: int = 300, context: float = 0.55,
                       fill: bool = False):
    """Turn a real photo into a flat 'poster': the QR becomes an axis-aligned qr_side square
    centred in a canvas with `context` x qr_side of real surroundings on each side.

    Where the canvas extends past the photo, the default mirrors the photo. With fill=True it
    uses the photo's median border colour instead: a tightly cropped QR (e.g. a screenshot)
    would otherwise be surrounded by mirrored copies of itself.

    Returns (poster, qr_box, H) where qr_box covers the QR plus its quiet zone and H maps
    photo pixels -> poster pixels.
    """
    m = int(round(qr_side * context))
    size = qr_side + 2 * m
    dst = np.array([[m, m], [m + qr_side, m], [m + qr_side, m + qr_side], [m, m + qr_side]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    if fill:
        ring = np.concatenate([photo[:2].reshape(-1, 3), photo[-2:].reshape(-1, 3),
                               photo[:, :2].reshape(-1, 3), photo[:, -2:].reshape(-1, 3)])
        colour = tuple(int(v) for v in np.median(ring, axis=0))
        poster = cv2.warpPerspective(photo, H, (size, size), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=colour)
    else:
        poster = cv2.warpPerspective(photo, H, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    e = int(round(qr_side * 0.07))
    return poster, (m - e, m - e, m + qr_side + e, m + qr_side + e), H


def tamper_real_photo(photo: np.ndarray, poster: np.ndarray, H: np.ndarray, kind: str,
                      rng: random.Random, qr_box) -> np.ndarray:
    """Apply a sticker to the *original* photo, in its true perspective.

    The sticker is placed on the rectified poster, then only the changed pixels are
    warped back into the photo, so the rest of the photo stays untouched.
    """
    t = tamper(poster, kind, rng, qr_box)
    changed = (np.abs(t.astype(np.int16) - poster).max(axis=2) > 0).astype(np.uint8) * 255
    changed = cv2.dilate(changed, np.ones((3, 3), np.uint8))
    h, w = photo.shape[:2]
    Hinv = np.linalg.inv(H)
    back = cv2.warpPerspective(t, Hinv, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(changed, Hinv, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
    return (photo * (1 - mask) + back * mask).astype(np.uint8)
