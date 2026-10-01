"""Evaluate QRGuard on held-out genuine and tampered photos.

Outputs in results/: metrics.json, score_histogram.png, roc_curve.png,
confusion_matrix.png, examples.png. Tamper type is read from the filename prefix
(e.g. loose_overlay_0003.jpg); real photos without a known prefix are grouped as "tampered".
Both score modes are reported; the checkpoint's default mode is the headline result.
"""

import argparse
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from qrguard import config
from qrguard.data import load_folder
from qrguard.detector import QRGuard, Result, _to_bgr_uint8, result_panel
from qrguard.synth import TAMPER_TYPES

COLOURS = ["#c62828", "#ef6c00", "#6a1b9a", "#1565c0", "#00838f"]


def tamper_kind(name: str) -> str:
    stem = name.rsplit("/", 1)[-1]
    return next((k for k in TAMPER_TYPES if stem.startswith(k)), "tampered")


def mode_metrics(sg, st, kinds_t, thr):
    y = np.r_[np.zeros(len(sg)), np.ones(len(st))]
    s = np.r_[sg, st]
    pred = (s > thr).astype(int)
    m = {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y, s)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_tpr": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float((sg > thr).mean()),
        "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "mean_score": {"genuine": float(sg.mean()), "tampered": float(st.mean())},
        "per_type": {},
    }
    for k in sorted(set(kinds_t)):
        sk = st[kinds_t == k]
        m["per_type"][k] = {
            "n": int(len(sk)),
            "detection_rate": float((sk > thr).mean()),
            "auc_vs_genuine": float(roc_auc_score(np.r_[np.zeros(len(sg)), np.ones(len(sk))], np.r_[sg, sk])),
            "mean_score": float(sk.mean()),
            "min_score": float(sk.min()),
        }
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(config.DATA_DIR / "synthetic"))
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    ap.add_argument("--out", default=str(config.RESULTS_DIR))
    args = ap.parse_args()

    out = config.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = QRGuard(args.checkpoint)
    data = config.Path(args.data)

    Xg, ng, fg = load_folder(data / "test" / "genuine")
    Xt, nt, ft = load_folder(data / "test" / "tampered")
    kinds_t = np.array([tamper_kind(n) for n in nt])

    by_mode = {}
    for mode in guard.thresholds:
        sg, rg, eg = guard.reconstruct(Xg, mode=mode)
        st, rt, et = guard.reconstruct(Xt, mode=mode)
        by_mode[mode] = (sg, st)
    primary = guard.score_mode
    thr = guard.threshold

    metrics = {
        "score_mode": primary,
        "threshold_rule": guard.ckpt["stats"].get("threshold_rule", "mean + kσ") +
                          f", k={guard.ckpt['stats'].get('k_sigma')}",
        "n_genuine": int(len(Xg)), "n_tampered": int(len(Xt)),
        "no_qr_found": {"genuine": len(fg), "tampered": len(ft)},
        **mode_metrics(*by_mode[primary], kinds_t, thr),
        "comparison_by_score_mode": {m: mode_metrics(*by_mode[m], kinds_t, guard.thresholds[m])
                                     for m in by_mode},
    }
    for m in metrics["comparison_by_score_mode"].values():
        m.pop("confusion_matrix [[TN,FP],[FN,TP]]")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    sg, st = by_mode[primary]
    s = np.r_[sg, st]
    y = np.r_[np.zeros(len(sg)), np.ones(len(st))]

    # --- score histogram (primary mode)
    plt.figure(figsize=(7.5, 4.2))
    bins = np.logspace(np.log10(max(s.min(), 1e-5)), np.log10(s.max()), 50)
    plt.hist(sg, bins=bins, alpha=0.7, label="genuine", color="#2e7d32")
    for k, c in zip(sorted(set(kinds_t)), COLOURS):
        plt.hist(st[kinds_t == k], bins=bins, alpha=0.55, label=f"tampered: {k}", color=c)
    plt.axvline(thr, color="k", ls="--", label=f"threshold {thr:.4f}")
    plt.xscale("log"), plt.xlabel(f"anomaly score ({primary}, log scale)"), plt.ylabel("count")
    plt.legend(fontsize=8), plt.title("Reconstruction error: genuine vs tampered (test set)")
    plt.tight_layout(), plt.savefig(out / "score_histogram.png", dpi=150), plt.close()

    # --- ROC, both modes
    plt.figure(figsize=(4.8, 4.8))
    for mode, (a, b) in by_mode.items():
        fpr, tpr, _ = roc_curve(y, np.r_[a, b])
        auc = metrics["comparison_by_score_mode"][mode]["roc_auc"]
        plt.plot(fpr, tpr, lw=2, label=f"{mode} score (AUC {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("false positive rate"), plt.ylabel("true positive rate"), plt.legend(loc="lower right")
    plt.title("ROC – QRGuard"), plt.tight_layout(), plt.savefig(out / "roc_curve.png", dpi=150), plt.close()

    # --- confusion matrix (primary mode)
    cm = np.array(metrics["confusion_matrix [[TN,FP],[FN,TP]]"])
    plt.figure(figsize=(4, 3.6))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=14,
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["genuine", "tampered"]), plt.yticks([0, 1], ["genuine", "tampered"])
    plt.xlabel("predicted"), plt.ylabel("actual"), plt.title(f"Confusion matrix ({primary} score)")
    plt.tight_layout(), plt.savefig(out / "confusion_matrix.png", dpi=150), plt.close()

    # --- example panels: 2 genuine + 1-2 of each tamper type
    _, rg, eg = guard.reconstruct(Xg[:2])
    rows = []

    def add(x, r, e, score, label):
        res = Result(status="tampered" if score > thr else "genuine", score=float(score), threshold=thr,
                     patch=_to_bgr_uint8(torch.from_numpy(x)), recon=_to_bgr_uint8(torch.from_numpy(r)),
                     error=e, heat_vmax=guard.heat_vmax)
        panel = result_panel(res, tile=200)
        bar = np.full((28, panel.shape[1], 3), 255, np.uint8)
        colour = (40, 40, 200) if res.status == "tampered" else (40, 140, 40)
        cv2.putText(bar, f"{label}  ->  {res.status.upper()}  (score {res.score:.4f})", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, panel]))

    for i in range(len(rg)):
        add(Xg[i], rg[i], eg[i], sg[i], "actual: genuine")
    for k in sorted(set(kinds_t)):
        idx = np.where(kinds_t == k)[0][:2]
        _, rk, ek = guard.reconstruct(Xt[idx])
        for j, i in enumerate(idx):
            add(Xt[i], rk[j], ek[j], st[i], f"actual: {k}")
    if rows:
        cv2.imwrite(str(out / "examples.png"), np.vstack(rows))

    # --- console summary
    print(f"test set: {len(Xg)} genuine, {len(Xt)} tampered (no QR found: {len(fg)} genuine, {len(ft)} tampered)\n")
    hdr = f"{'mode':6s} {'thr':>8s} {'AUC':>7s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'FPR':>7s}"
    print(hdr)
    for mode, m in metrics["comparison_by_score_mode"].items():
        tag = "  <- deployed" if mode == primary else ""
        print(f"{mode:6s} {m['threshold']:8.5f} {m['roc_auc']:7.4f} {m['accuracy']:7.3f} {m['precision']:7.3f} "
              f"{m['recall_tpr']:7.3f} {m['f1']:7.3f} {m['false_positive_rate']:7.3f}{tag}")
    print("\ndetection rate per tamper type:")
    print(f"{'type':18s}" + "".join(f"{m:>10s}" for m in by_mode))
    for k in sorted(set(kinds_t)):
        print(f"{k:18s}" + "".join(f"{metrics['comparison_by_score_mode'][m]['per_type'][k]['detection_rate']:10.3f}"
                                   for m in by_mode))
    print(f"\nconfusion matrix ({primary}) [[TN,FP],[FN,TP]]: {metrics['confusion_matrix [[TN,FP],[FN,TP]]']}")
    print(f"metrics + figures written to {out}")


if __name__ == "__main__":
    main()
