"""Supervised baseline: does a classifier generalise to tampering styles it never saw?

Report section 4 argues that a supervised genuine-vs-tampered classifier overfits to the
tampering styles in its training set. This script tests that claim with a
leave-one-attack-out protocol:

  for each tamper type T:
      train a CNN classifier on genuine + every tamper type except T
      test on genuine test photos + tamper type T (never seen)

and compares with the autoencoder, which never sees *any* tampered photo.
The classifier uses the same convolutional encoder as the autoencoder plus a linear
head, so the comparison is about the training paradigm, not the architecture.

Outputs: results/baseline_supervised.json, results/baseline_vs_autoencoder.png
"""

import argparse
import json
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qrguard import config, synth
from qrguard.data import PatchDataset, load_folder
from qrguard.detector import QRGuard
from qrguard.model import _down, pick_device


class TamperClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_down(3, 32), _down(32, 64), _down(64, 128), _down(128, 256),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class Labelled(Dataset):
    def __init__(self, X, y):
        self.ds = PatchDataset(X, augment=True)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.ds[i], self.y[i]


def ensure_tampered_train_pool(root, per_type: int, seed: int):
    """Tampered *training* photos for the classifier (separate seed, disjoint from the test set)."""
    out = root / "baseline_train" / "tampered"
    if out.exists() and len(list(out.glob("*.jpg"))) >= per_type * len(synth.TAMPER_TYPES):
        return out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    poster = synth.make_genuine_poster()
    for kind in synth.TAMPER_TYPES:
        for i in tqdm(range(per_type), desc=f"baseline pool [{kind}]"):
            photo = synth.simulate_photo(synth.tamper(poster, kind, rng), rng)
            cv2.imwrite(str(out / f"{kind}_{i:04d}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def kinds_of(names):
    return np.array([next(k for k in synth.TAMPER_TYPES if n.startswith(k)) for n in names])


@torch.no_grad()
def predict(model, X, device, bs=128):
    model.eval()
    return np.concatenate([torch.sigmoid(model(torch.from_numpy(X[i:i + bs]).to(device))).cpu().numpy()
                           for i in range(0, len(X), bs)])


def train_classifier(X, y, device, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TamperClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([float((y == 0).sum() / max((y == 1).sum(), 1))], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(Labelled(X, y), batch_size=32, shuffle=True, drop_last=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(config.DATA_DIR / "synthetic"))
    ap.add_argument("--per-type", type=int, default=200, help="tampered training photos per type")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    device = pick_device()
    root = config.Path(args.data)
    pool = ensure_tampered_train_pool(root, args.per_type, args.seed)

    Xg_tr, _, _ = load_folder(root / "train" / "genuine")
    Xt_tr, nt_tr, _ = load_folder(pool)
    Xg_te, _, _ = load_folder(root / "test" / "genuine")
    Xt_te, nt_te, _ = load_folder(root / "test" / "tampered")
    k_tr, k_te = kinds_of(nt_tr), kinds_of(nt_te)

    guard = QRGuard()
    ae_g, _, _ = guard.reconstruct(Xg_te)
    ae_t, _, _ = guard.reconstruct(Xt_te)

    results = {"protocol": "leave-one-attack-out", "classifier_threshold": 0.5,
               "autoencoder_threshold": guard.threshold, "autoencoder_score_mode": guard.score_mode,
               "folds": {}}
    for held in list(synth.TAMPER_TYPES) + ["none (all types seen)"]:
        seen = k_tr != held
        X = np.concatenate([Xg_tr, Xt_tr[seen]])
        y = np.r_[np.zeros(len(Xg_tr)), np.ones(seen.sum())]
        model = train_classifier(X, y, device, args.epochs, args.seed)
        pg = predict(model, Xg_te, device)
        test_mask = k_te == held if held in synth.TAMPER_TYPES else np.ones(len(k_te), bool)
        pt = predict(model, Xt_te[test_mask], device)
        at = ae_t[test_mask]
        yy = np.r_[np.zeros(len(pg)), np.ones(len(pt))]
        fold = {
            "trained_on": sorted(set(k_tr[seen])),
            "supervised": {"detection_rate": float((pt > 0.5).mean()), "fpr": float((pg > 0.5).mean()),
                           "auc": float(roc_auc_score(yy, np.r_[pg, pt]))},
            "autoencoder": {"detection_rate": float((at > guard.threshold).mean()),
                            "fpr": float((ae_g > guard.threshold).mean()),
                            "auc": float(roc_auc_score(yy, np.r_[ae_g, at]))},
        }
        results["folds"][held] = fold
        s, a = fold["supervised"], fold["autoencoder"]
        print(f"held-out {held:22s} supervised: det {s['detection_rate']:.3f} FPR {s['fpr']:.3f} AUC {s['auc']:.3f}"
              f" | autoencoder: det {a['detection_rate']:.3f} FPR {a['fpr']:.3f} AUC {a['auc']:.3f}")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "baseline_supervised.json", "w") as f:
        json.dump(results, f, indent=2)

    held = list(synth.TAMPER_TYPES)
    xs = np.arange(len(held))
    plt.figure(figsize=(7.5, 4))
    sup = [results["folds"][h]["supervised"]["detection_rate"] for h in held]
    ae = [results["folds"][h]["autoencoder"]["detection_rate"] for h in held]
    plt.bar(xs - 0.2, sup, 0.4, label="supervised CNN (type held out)", color="#9e9e9e")
    plt.bar(xs + 0.2, ae, 0.4, label="QRGuard autoencoder (no tampered data)", color="#8a1538")
    for x, v in zip(xs - 0.2, sup):
        plt.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for x, v in zip(xs + 0.2, ae):
        plt.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.xticks(xs, held), plt.ylim(0, 1.3), plt.ylabel("detection rate on unseen attack type")
    plt.title("Generalisation to an unseen tampering style"), plt.legend(fontsize=8, loc="upper center", ncol=2)
    plt.tight_layout(), plt.savefig(config.RESULTS_DIR / "baseline_vs_autoencoder.png", dpi=150), plt.close()
    print(f"written results/baseline_supervised.json and results/baseline_vs_autoencoder.png")


if __name__ == "__main__":
    main()
