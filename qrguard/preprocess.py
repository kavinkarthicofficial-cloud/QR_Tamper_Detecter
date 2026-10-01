"""Step 2 of the pipeline: locate the QR via its finder patterns, deskew, crop, resize.

The QR detector returns the four code corners ordered relative to the finder
patterns (top-left, top-right, bottom-right, bottom-left), so the perspective
warp also undoes in-plane rotation. A margin of surrounding poster is kept so the
autoencoder can see sticker edges, shadows and paper-colour changes around the code.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import IMG_SIZE, MAX_DETECT_SIDE, QR_MARGIN

_detectors = [cv2.QRCodeDetector()]
if hasattr(cv2, "QRCodeDetectorAruco"):
    _detectors.append(cv2.QRCodeDetectorAruco())


def load_image(path) -> np.ndarray:
    """Read an image as BGR uint8, raising a clear error if it cannot be read."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def decode_image_bytes(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded file is not a readable image")
    return img


def _valid_quad(pts: np.ndarray, min_area: float, max_side_ratio: float = 1.35) -> bool:
    if pts.shape != (4, 2) or not cv2.isContourConvex(pts) or cv2.contourArea(pts) <= min_area:
        return False
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return max(sides) / max(min(sides), 1e-6) <= max_side_ratio


def find_qr(bgr: np.ndarray):
    """Locate the QR. Returns (corners TL,TR,BR,BL in `bgr` pixels, decoded payload or "") or (None, "").

    Corners from a successful decode are preferred: decoding validates the finder-pattern
    geometry. Plain detection (no decode) is the fallback, filtered by a shape sanity check.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, MAX_DETECT_SIDE / max(h, w))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    candidates = (gray, cv2.equalizeHist(gray))
    min_area = 0.002 * gray.shape[0] * gray.shape[1]

    for det in _detectors:
        for img in candidates:
            try:
                text, pts, _ = det.detectAndDecode(img)
            except cv2.error:
                continue
            if text and pts is not None:
                pts = pts.reshape(-1, 2).astype(np.float32)
                if _valid_quad(pts, min_area):
                    return pts / scale, text

    for det in reversed(_detectors):          # ArUco-based detector gives tighter quads
        for img in candidates:
            try:
                ok, pts = det.detect(img)
            except cv2.error:
                continue
            if ok and pts is not None:
                pts = pts.reshape(-1, 2).astype(np.float32)
                if _valid_quad(pts, min_area):
                    return pts / scale, ""
    return None, ""


def find_qr_corners(bgr: np.ndarray) -> np.ndarray | None:
    return find_qr(bgr)[0]


def warp_qr_region(bgr: np.ndarray, corners: np.ndarray, size: int = IMG_SIZE,
                   margin: float = QR_MARGIN) -> np.ndarray:
    """Perspective-warp the QR (plus `margin` of context) to a size x size BGR patch."""
    # Warp at 4x and downsample with INTER_AREA to avoid aliasing the QR modules.
    big = size * 4
    q = big / (1 + 2 * margin)
    m = (big - q) / 2
    dst = np.array([[m, m], [m + q, m], [m + q, m + q], [m, m + q]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, H, (big, big), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(warped, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor_array(patch_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> RGB float32 3xHxW in [0, 1]."""
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def preprocess(bgr: np.ndarray):
    """Full preprocessing.

    Returns (array 3xSxS float32, patch_bgr, corners, payload); the first three are None
    when no QR code is found.
    """
    corners, payload = find_qr(bgr)
    if corners is None:
        return None, None, None, ""
    patch = warp_qr_region(bgr, corners)
    return to_tensor_array(patch), patch, corners, payload
