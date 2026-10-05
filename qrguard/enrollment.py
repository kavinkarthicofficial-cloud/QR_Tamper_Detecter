"""Enrol a new poster from photos of its genuine state, so QRGuard can check it later.

QRGuard is one-class: it has to learn what *this* poster looks like before it can say
whether a later photo of it has been tampered with. Enrolment needs only genuine photos
(one is enough): each photo is rectified around its QR and expanded into many simulated
phone shots, a fresh autoencoder is trained on them, and the threshold is calibrated on
held-out shots. No tampered example is used.
"""

from __future__ import annotations

import multiprocessing as mp
import random

import cv2
import numpy as np
import torch

from . import config, quality, synth
from .calibration import calibrate
from .model import pick_device
from .preprocess import find_qr, preprocess, to_tensor_array, warp_qr_region
from .training import fit

# ----------------------------------------------------------------- parallel shot generation
_P = {}


def _init(poster, box):
    _P["poster"], _P["box"] = poster, box
    cv2.setNumThreads(1)


def _sample(args):
    kind, seed = args
    rng = random.Random(seed)
    src = _P["poster"] if kind == "genuine" else synth.tamper(_P["poster"], kind, rng, _P["box"])
    arr, *_ = preprocess(synth.simulate_photo(src, rng, out_size=640, qr_box=_P["box"]))
    return arr


def generate(pool, kind, n, seed0):
    """n simulated shots of the pool's poster (genuine or one tamper kind) -> (patches, n_without_qr)."""
    arrs = pool.map(_sample, [(kind, seed0 + i) for i in range(n)], chunksize=8)
    ok = [a for a in arrs if a is not None]
    return (np.stack(ok) if ok else np.zeros((0, 3, config.IMG_SIZE, config.IMG_SIZE), np.float32)), n - len(ok)


def simulated_shots(photo, corners, n, seed, workers=8):
    poster, box, _ = synth.rectify_real_photo(photo, corners)
    with mp.Pool(workers, initializer=_init, initargs=(poster, box)) as pool:
        X, _ = generate(pool, "genuine", n, seed)
    return X


# ----------------------------------------------------------------- enrolment
def enroll(photos: list[np.ndarray], n_train: int = 300, n_val: int = 80, epochs: int = 60,
           min_epochs: int = 40, seed: int = 0, verbose: bool = False):
    """Train a poster-specific model from genuine photos.

    Returns (model, thresholds, stats). The real photos themselves are added to the
    training set; the simulated shots are split evenly across the photos.
    """
    usable = []
    for p in photos:
        corners, payload = find_qr(p)
        if corners is not None:
            usable.append((p, corners, payload))
    if not usable:
        raise ValueError("No QR code found in any enrolment photo -- retake it closer and straighter")

    per_tr, per_va = -(-n_train // len(usable)), -(-n_val // len(usable))
    Xtr, Xva = [], []
    for i, (p, corners, _) in enumerate(usable):
        Xtr.append(simulated_shots(p, corners, per_tr, seed + 1_000_000 * i))
        Xva.append(simulated_shots(p, corners, per_va, seed + 1_000_000 * i + 500_000))
        Xtr.append(to_tensor_array(warp_qr_region(p, corners))[None])     # the real photo itself
    Xtr, Xva = np.concatenate(Xtr), np.concatenate(Xva)

    device = pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, hist = fit(Xtr, Xva, device, epochs=epochs, patience=10, min_epochs=min_epochs, verbose=verbose)
    thresholds, stats = calibrate(model, Xva, device, config.K_SIGMA)
    stats.update({"n_train": int(len(Xtr)), "epochs_run": len(hist["train"]), "n_photos": len(usable),
                  "payloads": sorted({u[2] for u in usable if u[2]}),
                  "quality": quality.reference_ranges(Xtr)})      # capture conditions seen at enrolment
    return model, thresholds, stats
