"""Basic convolutional autoencoder (Conv2D encoder, ConvTranspose2D decoder).

A dense bottleneck forces the network to learn a compact description of the
*genuine* poster only, so it cannot simply copy arbitrary input through. A pasted
sticker (different QR modules, paper edge, shadow) is pulled back towards the
genuine appearance on reconstruction, which shows up as high reconstruction error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import IMG_SIZE, LATENT_DIM, LOCAL_WINDOW, SCORE_MODE


def _down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                         nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


def _up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, in_ch: int = 3, img_size: int = IMG_SIZE):
        super().__init__()
        self.feat = img_size // 16                      # 128 -> 8 after four stride-2 convs
        flat = 256 * self.feat * self.feat
        self.encoder = nn.Sequential(
            _down(in_ch, 32), _down(32, 64), _down(64, 128), _down(128, 256),
            nn.Flatten(), nn.Linear(flat, latent_dim),
        )
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim, flat), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            _up(256, 128), _up(128, 64), _up(64, 32),
            nn.ConvTranspose2d(32, in_ch, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z).view(-1, 256, self.feat, self.feat)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def error_map(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Pixel-wise squared error averaged over channels: (N,3,H,W) -> (N,H,W)."""
    return ((x - recon) ** 2).mean(dim=1)


def anomaly_score(x: torch.Tensor, recon: torch.Tensor, mode: str = SCORE_MODE,
                  window: int = LOCAL_WINDOW) -> torch.Tensor:
    """Per-image scalar compared against the threshold.

    mode="mean":  global reconstruction MSE.
    mode="local": worst window-averaged MSE -- a contiguous sticker region is not diluted
                  by the rest of the (well-reconstructed) image.
    """
    e = error_map(x, recon)
    if mode == "mean":
        return e.flatten(1).mean(dim=1)
    if mode == "local":
        return F.avg_pool2d(e.unsqueeze(1), window, stride=1).flatten(1).amax(dim=1)
    raise ValueError(f"unknown score mode: {mode}")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model: ConvAutoencoder, thresholds: dict, stats: dict, score_mode: str = SCORE_MODE):
    """`thresholds` maps score mode -> calibrated threshold; `score_mode` is the default used at inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "latent_dim": model.encoder[-1].out_features,
                "score_mode": score_mode, "thresholds": {k: float(v) for k, v in thresholds.items()},
                "threshold": float(thresholds[score_mode]), "stats": stats}, path)


def load_checkpoint(path, device: torch.device | None = None):
    device = device or pick_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ConvAutoencoder(latent_dim=ckpt.get("latent_dim", LATENT_DIM))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
