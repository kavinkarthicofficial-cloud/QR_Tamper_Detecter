"""Load a folder of poster photos into aligned 128x128 patches (cached as .npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .preprocess import load_image, preprocess

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_folder(folder: Path, use_cache: bool = True):
    """Preprocess every image under `folder`.

    Returns (X float32 Nx3xHxW, names list[str], failed list[str]) where `failed`
    are images in which no QR code could be located.
    """
    folder = Path(folder)
    files = list_images(folder)
    cache = folder / ".qrguard_cache.npz"
    sig = np.array([f"{p.relative_to(folder)}:{p.stat().st_mtime_ns}" for p in files])
    if use_cache and cache.exists():
        c = np.load(cache, allow_pickle=False)
        if c["sig"].shape == sig.shape and (c["sig"] == sig).all():
            return c["X"], list(c["names"]), list(c["failed"])

    xs, names, failed = [], [], []
    for p in tqdm(files, desc=f"preprocess {folder.parent.name}/{folder.name}", leave=False):
        arr, *_ = preprocess(load_image(p))
        rel = str(p.relative_to(folder))
        if arr is None:
            failed.append(rel)
        else:
            xs.append(arr)
            names.append(rel)
    X = np.stack(xs) if xs else np.zeros((0, 3, 1, 1), np.float32)
    if use_cache:
        np.savez(cache, X=X, names=np.array(names), failed=np.array(failed), sig=sig)
    return X, names, failed


class PatchDataset(Dataset):
    """Genuine patches with light augmentation (small shift + photometric jitter).

    The augmentation makes the model tolerant to the residual misalignment and lighting
    differences that genuine photos have, so those do not raise false alarms.
    """

    def __init__(self, X: np.ndarray, augment: bool = False, max_shift: int = 3):
        self.X = torch.from_numpy(X)
        self.augment = augment
        self.max_shift = max_shift

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if not self.augment:
            return x
        s = self.max_shift
        dx, dy = np.random.randint(-s, s + 1, size=2)
        x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(1, 2))
        gain = 1 + 0.08 * (2 * torch.rand(3, 1, 1) - 1)
        bias = 0.04 * (2 * torch.rand(1) - 1)
        return (x * gain + bias).clamp(0, 1)
