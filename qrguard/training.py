"""Autoencoder training loop shared by train.py and the real-web-photo experiment."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .data import PatchDataset
from .model import ConvAutoencoder, anomaly_score


@torch.no_grad()
def val_mse(model, X, device, bs=64):
    """Mean reconstruction MSE over a set of patches (early-stopping criterion)."""
    model.eval()
    tot = 0.0
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        tot += anomaly_score(x, model(x), mode="mean").sum().item()
    return tot / len(X)


def fit(Xtr: np.ndarray, Xva: np.ndarray, device, model: ConvAutoencoder | None = None,
        epochs: int = config.EPOCHS, batch_size: int = config.BATCH_SIZE, lr: float = config.LEARNING_RATE,
        latent_dim: int = config.LATENT_DIM, patience: int = config.PATIENCE, min_epochs: int = 0,
        verbose: bool = True):
    """Train on genuine patches only; returns (best model by val MSE, history).

    Early stopping only applies after `min_epochs`: autoencoders often sit on a plateau
    (blurry, average-looking reconstructions) before they learn fine detail.
    """
    model = model or ConvAutoencoder(latent_dim=latent_dim).to(device)
    loader = DataLoader(PatchDataset(Xtr, augment=True), batch_size=batch_size, shuffle=True,
                        drop_last=len(Xtr) > batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = torch.nn.MSELoss()

    history = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x in loader:
            x = x.to(device)
            loss = loss_fn(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        tr = tot / n
        va = val_mse(model, Xva, device)
        sched.step(va)
        history["train"].append(tr)
        history["val"].append(va)
        flag = ""
        if va < best - 1e-7:
            best, bad, flag = va, 0, " *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"epoch {epoch:3d}  train {tr:.6f}  val {va:.6f}  lr {opt.param_groups[0]['lr']:.1e}{flag}")
        if bad >= patience and epoch >= min_epochs:
            if verbose:
                print("early stopping")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history
