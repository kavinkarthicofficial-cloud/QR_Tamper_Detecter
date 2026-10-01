"""Decision-threshold calibration from genuine photos only (no tampered examples needed)."""

from __future__ import annotations

import numpy as np
import torch

from .config import SCORE_MODES
from .model import anomaly_score


@torch.no_grad()
def scores_by_mode(model, X: np.ndarray, device, bs: int = 64) -> dict[str, np.ndarray]:
    """Anomaly scores of every image in X under every score mode."""
    model.eval()
    out = {m: [] for m in SCORE_MODES}
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        r = model(x)
        for m in SCORE_MODES:
            out[m].append(anomaly_score(x, r, mode=m).cpu())
    return {m: torch.cat(v).numpy() for m, v in out.items()}


def log_sigma_threshold(scores: np.ndarray, k_sigma: float):
    """exp(mean + k*std) of log scores.

    Reconstruction errors are positive and right-skewed, so the mean + kσ bound is taken
    in log space (a log-normal tail bound) rather than on the raw errors.
    """
    log_s = np.log(np.maximum(scores, 1e-12))
    threshold = float(np.exp(log_s.mean() + k_sigma * log_s.std()))
    stats = {"val_mean": float(scores.mean()), "val_std": float(scores.std()),
             "val_log_mean": float(log_s.mean()), "val_log_std": float(log_s.std()),
             "val_p99": float(np.percentile(scores, 99)), "val_max": float(scores.max())}
    return threshold, stats


def calibrate(model, X_val: np.ndarray, device, k_sigma: float):
    """Returns (thresholds {mode: thr}, stats) computed on genuine validation patches."""
    thresholds, stats = {}, {"k_sigma": k_sigma, "n_val": int(len(X_val)),
                             "threshold_rule": "log-space mean + kσ of genuine validation scores"}
    for mode, s in scores_by_mode(model, X_val, device).items():
        thresholds[mode], stats[mode] = log_sigma_threshold(s, k_sigma)
    return thresholds, stats
