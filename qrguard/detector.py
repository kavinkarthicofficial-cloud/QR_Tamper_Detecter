"""End-to-end inference: photo -> preprocess -> autoencoder -> error map -> verdict."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .config import CHECKPOINT_DIR, DEFAULT_CHECKPOINT
from .model import anomaly_score, error_map, load_checkpoint, pick_device
from .preprocess import preprocess

HEAT_SCALE = 12.0   # a pixel error of HEAT_SCALE x the global-MSE threshold is drawn at full intensity


@dataclass
class Result:
    status: str                      # "genuine" | "tampered" | "no_qr"
    score: float | None = None
    threshold: float | None = None
    payload: str = ""
    patch: np.ndarray | None = None  # BGR uint8, model input
    recon: np.ndarray | None = None  # BGR uint8, reconstruction
    error: np.ndarray | None = None  # float32 HxW pixel-wise MSE
    corners: np.ndarray | None = None
    heat_vmax: float | None = None   # colour scale of the error heatmap

    @property
    def ratio(self) -> float | None:
        return None if self.score is None else self.score / self.threshold

    def summary(self) -> dict:
        return {"status": self.status, "score": self.score, "threshold": self.threshold,
                "score_over_threshold": self.ratio, "payload": self.payload}


def _to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    rgb = (t.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class QRGuard:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device: torch.device | None = None,
                 threshold: float | None = None, score_mode: str | None = None):
        self.device = device or pick_device()
        self.model, self.ckpt = load_checkpoint(checkpoint, self.device)
        thresholds = self.ckpt.get("thresholds", {"mean": self.ckpt["threshold"]})
        self.thresholds = thresholds
        self.score_mode = score_mode or self.ckpt.get("score_mode", "mean")
        self.threshold = float(threshold if threshold is not None else thresholds[self.score_mode])
        # Heatmap colour scale: a fixed multiple of the global-MSE threshold, so genuine stays dark.
        self.heat_vmax = thresholds.get("mean", self.threshold) * HEAT_SCALE

    @torch.no_grad()
    def reconstruct(self, X: np.ndarray, batch_size: int = 64, mode: str | None = None):
        """X: Nx3xHxW float32 -> (scores N, recon Nx3xHxW, error maps NxHxW) as numpy."""
        mode = mode or self.score_mode
        scores, recons, errs = [], [], []
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).to(self.device)
            r = self.model(x)
            scores.append(anomaly_score(x, r, mode=mode).cpu())
            recons.append(r.cpu())
            errs.append(error_map(x, r).cpu())
        if not scores:
            return np.zeros(0), np.zeros((0, *X.shape[1:])), np.zeros((0, *X.shape[2:]))
        return torch.cat(scores).numpy(), torch.cat(recons).numpy(), torch.cat(errs).numpy()

    def analyze(self, bgr: np.ndarray) -> Result:
        arr, patch, corners, payload = preprocess(bgr)
        if arr is None:
            return Result(status="no_qr", threshold=self.threshold)
        scores, recons, errs = self.reconstruct(arr[None])
        score = float(scores[0])
        return Result(
            status="tampered" if score > self.threshold else "genuine",
            score=score, threshold=self.threshold, payload=payload, patch=patch,
            recon=_to_bgr_uint8(torch.from_numpy(recons[0])), error=errs[0], corners=corners,
            heat_vmax=self.heat_vmax,
        )


def available_posters() -> dict:
    """Every poster QRGuard can check: name -> checkpoint path.

    "demo-canteen" is the synthetic demo poster; enrol.py adds posters/<name>.pt;
    web_experiment.py saves a few real web posters under web/.
    """
    out = {}
    if DEFAULT_CHECKPOINT.exists():
        out["demo-canteen"] = DEFAULT_CHECKPOINT
    for sub in ("web", "posters"):                # enrolled posters win on a name clash
        for p in sorted((CHECKPOINT_DIR / sub).glob("*.pt")):
            out[p.stem] = p
    return out


# --------------------------------------------------------------------------- visualisation

def heatmap(error: np.ndarray, base_bgr: np.ndarray, vmax: float | None = None) -> np.ndarray:
    """Overlay a smoothed reconstruction-error heatmap on the input patch."""
    e = cv2.GaussianBlur(error.astype(np.float32), (0, 0), 1.5)
    vmax = vmax or max(float(np.percentile(e, 99.5)), 1e-6)
    norm = np.clip(e / vmax, 0, 1)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(base_bgr, 0.45, colour, 0.55, 0)


def annotate_photo(bgr: np.ndarray, res: Result, max_side: int = 900) -> np.ndarray:
    """Draw the detected QR outline + verdict on the original photo."""
    s = min(1.0, max_side / max(bgr.shape[:2]))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    colour = {"genuine": (60, 180, 60), "tampered": (40, 40, 220), "no_qr": (0, 160, 255)}[res.status]
    if res.corners is not None:
        cv2.polylines(img, [(res.corners * s).astype(np.int32)], True, colour, max(2, img.shape[1] // 200),
                      cv2.LINE_AA)
    label = {"genuine": "GENUINE", "tampered": "TAMPERED", "no_qr": "NO QR FOUND"}[res.status]
    if res.score is not None:
        label += f"  score {res.score:.4f} / thr {res.threshold:.4f}"
    fs = img.shape[1] / 900
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), colour, -1)
    cv2.putText(img, label, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def result_panel(res: Result, tile: int = 256) -> np.ndarray:
    """Input | reconstruction | error heatmap, side by side, with captions."""
    if res.patch is None:
        return np.zeros((tile, tile * 3, 3), np.uint8)
    up = lambda im: cv2.resize(im, (tile, tile), interpolation=cv2.INTER_NEAREST)
    # Fixed colour scale tied to the threshold, so genuine photos stay dark and tampering glows.
    tiles = [up(res.patch), up(res.recon), up(heatmap(res.error, res.patch, vmax=res.heat_vmax))]
    for t, cap in zip(tiles, ["input (aligned)", "reconstruction", "error map"]):
        cv2.rectangle(t, (0, tile - 24), (tile, tile), (0, 0, 0), -1)
        cv2.putText(t, cap, (6, tile - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack(tiles)
