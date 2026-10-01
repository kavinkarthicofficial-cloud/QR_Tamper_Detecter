"""Fast smoke tests: python -m pytest -q"""

import random

import numpy as np
import pytest
import torch

from qrguard import synth
from qrguard.config import DEFAULT_CHECKPOINT, IMG_SIZE
from qrguard.model import ConvAutoencoder, anomaly_score
from qrguard.preprocess import preprocess


@pytest.fixture(scope="module")
def poster():
    return synth.make_genuine_poster()


def test_autoencoder_shapes():
    model = ConvAutoencoder().eval()
    x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        r = model(x)
    assert r.shape == x.shape
    assert 0 <= r.min() and r.max() <= 1
    for mode in ("mean", "local"):
        assert anomaly_score(x, r, mode=mode).shape == (2,)


def test_local_score_is_sensitive_to_small_patches():
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    r = x.clone()
    r[..., 10:30, 10:30] = 1.0                       # small contiguous error region
    mean, local = anomaly_score(x, r, "mean"), anomaly_score(x, r, "local")
    assert local.item() > 10 * mean.item()


def test_preprocess_finds_qr_in_simulated_photo(poster):
    rng = random.Random(1)
    arr, patch, corners, payload = preprocess(synth.simulate_photo(poster, rng))
    assert arr is not None and arr.shape == (3, IMG_SIZE, IMG_SIZE)
    assert arr.dtype == np.float32 and 0 <= arr.min() and arr.max() <= 1
    assert corners.shape == (4, 2)


def test_preprocess_returns_none_without_qr():
    blank = np.full((400, 400, 3), 128, np.uint8)
    arr, patch, corners, payload = preprocess(blank)
    assert arr is None and corners is None


@pytest.mark.parametrize("kind", synth.TAMPER_TYPES)
def test_tampering_changes_qr_region(poster, kind):
    t = synth.tamper(poster, kind, random.Random(3))
    x0, y0, x1, y1 = synth.QR_BOX
    assert t.shape == poster.shape
    assert np.abs(t[y0:y1, x0:x1].astype(int) - poster[y0:y1, x0:x1]).mean() > 5


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="train the model first")
def test_trained_model_separates_genuine_and_tampered(poster):
    from qrguard.detector import QRGuard
    guard = QRGuard()
    rng = random.Random(2024)
    genuine = guard.analyze(synth.simulate_photo(poster, rng))
    tampered = guard.analyze(synth.simulate_photo(synth.tamper(poster, "loose_overlay", rng), rng))
    assert genuine.status == "genuine"
    assert tampered.status == "tampered"
    assert tampered.score > genuine.score
