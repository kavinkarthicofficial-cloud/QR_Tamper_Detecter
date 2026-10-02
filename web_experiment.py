"""Train and test QRGuard on real QR photographs downloaded from the web (fetch_web_qr.py).

For every real photo (= one deployed poster), exactly as QRGuard would be deployed:
  1. rectify the photo so the real QR is axis-aligned, keeping real surroundings as context
  2. TRAIN a fresh autoencoder on simulated phone shots of that real poster (genuine only)
  3. CALIBRATE its threshold on separate genuine shots
  4. TEST on
     a) new genuine shots + shots with each of the 4 sticker attacks over the real QR
     b) the untouched ORIGINAL web photo (must pass as genuine)
     c) the original photo with each sticker warped onto it in true perspective

Outputs in results/web/: metrics.json, per_poster.csv, summary.png, examples.png
"""

import argparse
import csv
import json
import multiprocessing as mp
import random
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from qrguard import config, synth
from qrguard.calibration import calibrate
from qrguard.detector import QRGuard, Result, _to_bgr_uint8, result_panel
from qrguard.enrollment import _init, generate
from qrguard.model import anomaly_score, pick_device, save_checkpoint
from qrguard.preprocess import find_qr, preprocess
from qrguard.training import fit

OUT = config.RESULTS_DIR / "web"
RAW = config.DATA_DIR / "web" / "raw"

# ----------------------------------------------------------------- scoring helpers
class _Guard(QRGuard):
    """QRGuard around an in-memory model (no checkpoint file)."""

    def __init__(self, model, thresholds, device, mode):
        self.device, self.model, self.thresholds = device, model, thresholds
        self.score_mode, self.threshold = mode, thresholds[mode]
        self.heat_vmax = thresholds["mean"] * 12.0
        self.ckpt = {"thresholds": thresholds}


@torch.no_grad()
def scores(model, X, device, mode):
    if len(X) == 0:
        return np.zeros(0)
    x = torch.from_numpy(X).to(device)
    return np.concatenate([anomaly_score(x[i:i + 128], model(x[i:i + 128]), mode=mode).cpu().numpy()
                           for i in range(0, len(x), 128)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=80, help="genuine test shots per poster")
    ap.add_argument("--n-tamper", type=int, default=40, help="test shots per attack type per poster")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min-epochs", type=int, default=40,
                    help="no early stopping before this epoch (AEs plateau before learning fine detail)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N posters (0 = all)")
    ap.add_argument("--save-checkpoints", type=int, default=3, help="keep model files for the first N posters")
    args = ap.parse_args()

    sources = json.load(open(RAW / "sources.json"))
    if args.limit:
        sources = sources[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    (config.CHECKPOINT_DIR / "web").mkdir(parents=True, exist_ok=True)
    device = pick_device()
    mode = config.SCORE_MODE
    kinds = list(synth.TAMPER_TYPES)
    per_dir = OUT / "posters"
    per_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for pi, src in enumerate(sources):
        done_file = per_dir / f"{src['file'][:-4]}.json"
        if done_file.exists():
            continue                      # resumable: finished posters are skipped
        t0 = time.time()
        photo = cv2.imread(str(RAW / src["file"]))
        corners, _ = find_qr(photo)
        if corners is None:
            print(f"[{pi + 1}/{len(sources)}] {src['file']}: QR not re-detected, skipped")
            continue
        poster, box, H = synth.rectify_real_photo(photo, corners)
        base = 1_000_000 * (pi + 1)
        with mp.Pool(8, initializer=_init, initargs=(poster, box)) as pool:
            Xtr, _ = generate(pool, "genuine", args.n_train, base)
            Xva, _ = generate(pool, "genuine", args.n_val, base + 100_000)
            Xte, miss_g = generate(pool, "genuine", args.n_test, base + 200_000)
            Xt = {}
            miss_t = 0
            for j, k in enumerate(kinds):
                Xt[k], m_ = generate(pool, k, args.n_tamper, base + 300_000 + 10_000 * j)
                miss_t += m_

        torch.manual_seed(pi)
        np.random.seed(pi)
        model, hist = fit(Xtr, Xva, device, epochs=args.epochs, patience=10, min_epochs=args.min_epochs,
                          verbose=False)
        thresholds, stats = calibrate(model, Xva, device, config.K_SIGMA)
        if pi < args.save_checkpoints:
            save_checkpoint(config.CHECKPOINT_DIR / "web" / f"{src['file'][:-4]}.pt", model, thresholds,
                            {**stats, "source": src}, score_mode=mode)

        row = {"poster": src["file"], "title": src["title"], "license": src["license"],
               "qr_side_px": src["qr_side_px"], "epochs": len(hist["train"]),
               "no_qr_genuine": miss_g, "no_qr_tampered": miss_t}
        ratios = {}                       # score / threshold for every test shot, per mode and class
        for m in config.SCORE_MODES:
            thr = thresholds[m]
            sg = scores(model, Xte, device, m)
            st = {k: scores(model, Xt[k], device, m) for k in kinds}
            all_t = np.concatenate(list(st.values()))
            row[f"{m}_threshold"] = thr
            row[f"{m}_auc"] = roc_auc_score(np.r_[np.zeros(len(sg)), np.ones(len(all_t))], np.r_[sg, all_t])
            row[f"{m}_fpr"] = float((sg > thr).mean())
            row[f"{m}_recall"] = float((all_t > thr).mean())
            ratios[m] = {"genuine": (sg / thr).round(4).tolist(),
                         **{k: (st[k] / thr).round(4).tolist() for k in kinds}}
            for k in kinds:
                row[f"{m}_det_{k}"] = float((st[k] > thr).mean())

        # (b) + (c): the original web photo, untouched and with perspective-correct stickers
        guard = _Guard(model, thresholds, device, mode)
        res_raw = guard.analyze(photo)
        row["raw_genuine_pass"] = res_raw.status == "genuine"
        row["raw_genuine_ratio"] = res_raw.ratio
        rng = random.Random(base + 900_000)
        raw_t = {}
        for k in kinds:
            res = guard.analyze(synth.tamper_real_photo(photo, poster, H, k, rng, box))
            raw_t[k] = res
            row[f"raw_{k}_flagged"] = res.status == "tampered"
        with open(done_file, "w") as f:
            json.dump({"row": row, "ratios": ratios}, f)

        if pi < 6:
            def labelled(res, label):
                p = result_panel(res, tile=160)
                bar = np.full((24, p.shape[1], 3), 255, np.uint8)
                col = (40, 40, 200) if res.status == "tampered" else (40, 140, 40)
                txt = f"{label}: {res.status.upper()}" + (f" ({res.ratio:.1f}x thr)" if res.ratio else "")
                cv2.putText(bar, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                return np.vstack([bar, p])
            thumb = cv2.resize(photo, (160, 160), interpolation=cv2.INTER_AREA)
            left = np.vstack([np.full((24, 160, 3), 255, np.uint8), thumb])
            cv2.imwrite(str(per_dir / f"{src['file'][:-4]}_example.png"),
                        np.hstack([left, labelled(res_raw, "original photo"),
                                   labelled(raw_t["loose_overlay"], "+ sticker"),
                                   labelled(raw_t["partial_patch"], "+ partial patch")]))

        dets = " ".join(f"{k.split('_')[0][:5]} {row[f'{mode}_det_{k}']:.2f}" for k in kinds)
        raw_flags = sum(row[f"raw_{k}_flagged"] for k in kinds)
        print(f"[{pi + 1:2d}/{len(sources)}] {src['file']} AUC {row[f'{mode}_auc']:.4f} FPR {row[f'{mode}_fpr']:.3f} "
              f"det: {dets} | original photo {'PASS' if row['raw_genuine_pass'] else 'FLAGGED'} "
              f"({row['raw_genuine_ratio']:.2f}x), stickered originals flagged {raw_flags}/4 "
              f"[{time.time() - t0:.0f}s]", flush=True)

    aggregate(args, sources, kinds, mode, per_dir, time.time() - t_start)


def aggregate(args, sources, kinds, mode, per_dir, seconds):
    """Combine every finished poster's result file into the summary metrics and figures."""
    done = [json.load(open(per_dir / f"{s['file'][:-4]}.json")) for s in sources
            if (per_dir / f"{s['file'][:-4]}.json").exists()]
    rows = [d["row"] for d in done]
    agg = {m: {c: [np.array(d["ratios"][m][c]) > 1 for d in done] for c in ["genuine", *kinds]}
           for m in config.SCORE_MODES}
    example_rows = [cv2.imread(str(p)) for p in sorted(per_dir.glob("*_example.png"))]
    n = len(rows)
    summary = {"n_posters": n, "score_mode": mode, "minutes_this_run": round(seconds / 60, 1),
               "training": {"epochs": args.epochs, "min_epochs": args.min_epochs, "patience": 10},
               "per_poster_sets": {"train": args.n_train, "val": args.n_val, "test_genuine": args.n_test,
                                   "test_tampered_per_type": args.n_tamper},
               "download_detection_stats": json.load(open(RAW / "detection_stats.json")), "by_mode": {}}
    for m in config.SCORE_MODES:
        g = np.concatenate(agg[m]["genuine"])
        t = np.concatenate([np.concatenate(agg[m][k]) for k in kinds])
        aucs = np.array([r[f"{m}_auc"] for r in rows])
        summary["by_mode"][m] = {
            "simulated_shots": {
                "false_positive_rate": float(g.mean()), "detection_rate": float(t.mean()),
                "accuracy": float((np.sum(~g) + np.sum(t)) / (len(g) + len(t))),
                "auc_mean": float(aucs.mean()), "auc_median": float(np.median(aucs)), "auc_min": float(aucs.min()),
                "posters_auc_ge_0.99": int((aucs >= 0.99).sum()),
                "detection_by_type": {k: float(np.concatenate(agg[m][k]).mean()) for k in kinds},
            }}
    summary["original_photos"] = {
        "genuine_pass_rate": float(np.mean([r["raw_genuine_pass"] for r in rows])),
        "stickered_flag_rate": float(np.mean([r[f"raw_{k}_flagged"] for r in rows for k in kinds])),
        "stickered_flag_rate_by_type": {k: float(np.mean([r[f"raw_{k}_flagged"] for r in rows])) for k in kinds},
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT / "per_poster.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if example_rows:
        cv2.imwrite(str(OUT / "examples.png"), np.vstack(example_rows))

    # summary figure: per-type detection + FPR (both modes), and per-poster AUC
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = kinds + ["false positives"]
    xs = np.arange(len(labels))
    for off, m, c in [(-0.2, "mean", "#9e9e9e"), (0.2, "local", "#8a1538")]:
        s = summary["by_mode"][m]["simulated_shots"]
        vals = [s["detection_by_type"][k] for k in kinds] + [s["false_positive_rate"]]
        ax[0].bar(xs + off, vals, 0.4, label=f"{m} score", color=c)
        for x, v in zip(xs + off, vals):
            ax[0].text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax[0].set_xticks(xs, labels, rotation=15, fontsize=8), ax[0].set_ylim(0, 1.15)
    ax[0].set_title(f"{n} real web posters: detection rate per attack"), ax[0].legend(fontsize=8)
    aucs = sorted(r[f"{mode}_auc"] for r in rows)
    ax[1].bar(range(n), aucs, color="#8a1538")
    ax[1].set_ylim(min(0.9, min(aucs) - 0.01), 1.001), ax[1].set_xlabel("poster (sorted)")
    ax[1].set_ylabel("ROC AUC"), ax[1].set_title(f"Per-poster AUC ({mode} score)")
    plt.tight_layout(), plt.savefig(OUT / "summary.png", dpi=150), plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
