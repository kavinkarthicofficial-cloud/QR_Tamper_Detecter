"""Train the QRGuard autoencoder on genuine poster photos only, then calibrate the threshold.

Thresholds (one per score mode) are exp(mean + K * std) of log anomaly scores on the
*genuine* validation set -- no tampered example is ever used for training or calibration.
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from qrguard import config
from qrguard.calibration import calibrate
from qrguard.data import load_folder
from qrguard.model import ConvAutoencoder, load_checkpoint, pick_device, save_checkpoint
from qrguard.training import fit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(config.DATA_DIR / "synthetic"))
    ap.add_argument("--out", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--latent-dim", type=int, default=config.LATENT_DIM)
    ap.add_argument("--k-sigma", type=float, default=config.K_SIGMA)
    ap.add_argument("--score-mode", choices=config.SCORE_MODES, default=config.SCORE_MODE)
    ap.add_argument("--init", default=None, help="fine-tune from an existing checkpoint (e.g. on real photos)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    data = config.Path(args.data)

    Xtr, _, ftr = load_folder(data / "train" / "genuine")
    Xva, _, fva = load_folder(data / "val" / "genuine")
    print(f"device={device}  train={len(Xtr)} (no QR: {len(ftr)})  val={len(Xva)} (no QR: {len(fva)})")
    if len(Xtr) == 0 or len(Xva) == 0:
        raise SystemExit("Need genuine images in train/genuine and val/genuine")

    if args.init:
        model, _ = load_checkpoint(args.init, device)
        model.train()
    else:
        model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    out = config.Path(args.out)
    t0 = time.time()
    model, history = fit(Xtr, Xva, device, model=model, epochs=args.epochs, batch_size=args.batch_size,
                         lr=args.lr)

    thresholds, stats = calibrate(model, Xva, device, args.k_sigma)
    stats.update({"n_train": len(Xtr), "epochs_run": len(history["train"]),
                  "train_seconds": time.time() - t0, "data": str(data)})
    save_checkpoint(out, model, thresholds, stats, score_mode=args.score_mode)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(config.RESULTS_DIR / "train_history.json", "w") as f:
        json.dump({"history": history, "stats": stats, "thresholds": thresholds}, f, indent=2)
    plt.figure(figsize=(6, 4))
    plt.plot(history["train"], label="train MSE")
    plt.plot(history["val"], label="val MSE (genuine)")
    plt.yscale("log"), plt.xlabel("epoch"), plt.ylabel("reconstruction MSE"), plt.legend()
    plt.title("QRGuard autoencoder training")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "training_curve.png", dpi=150)

    print(f"\nsaved {out}")
    for mode, thr in thresholds.items():
        tag = " (default)" if mode == args.score_mode else ""
        print(f"[{mode:5s}] val score mean={stats[mode]['val_mean']:.6f} p99={stats[mode]['val_p99']:.6f} -> "
              f"threshold (log-space mean+{args.k_sigma}σ) = {thr:.6f}{tag}")


if __name__ == "__main__":
    main()
