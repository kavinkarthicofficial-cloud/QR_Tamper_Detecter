# QRGuard: Detecting Physical QR Code Tampering Using Autoencoders

23CSE475 Generative AI, Amrita Vishwa Vidyapeetham.
Team: Sanjay A R (CB.SC.U4CSE23052), Praveen S B (CB.SC.U4CSE23136), Kavin Karthic M (CB.SC.U4CSE23161).
Instructor: Dr. Swapna T.R.

QRGuard checks a photo of a QR poster for a malicious sticker pasted over the genuine code
(*quishing*) **before** anyone scans it. A convolutional autoencoder is trained **only on photos
of the genuine poster**. A tampered poster reconstructs poorly, and the reconstruction error
flags it. The model never sees a tampered example during training or threshold calibration.

```
photo ──► find QR finder patterns ──► deskew + crop (128×128, with poster margin)
      ──► convolutional autoencoder ──► pixel-wise error map ──► score > threshold ?  ──► GENUINE / TAMPERED
```

## Results (synthetic test set, held out)

200 genuine photos and 400 tampered photos across 4 attack types. All 600 test photos were
generated independently of the training data, and the QR was located in every one.

| Score | Threshold | ROC AUC | Accuracy | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|---|---|---|
| `mean`: global MSE (as in the report) | 0.00734 | 0.9964 | 0.978 | 0.992 | 0.975 | 0.984 | 1.5% |
| **`local`: worst 16×16 window MSE (deployed)** | 0.03869 | **0.9999** | **0.995** | 0.993 | **1.000** | **0.996** | 1.5% |

Confusion matrix (local score): TN 197, FP 3, FN 0, TP 400.

Detection rate per attack type:

| Attack | What the attacker does | `mean` | `local` |
|---|---|---|---|
| `aligned_overlay` | same-size sticker placed exactly over the QR (hardest full overlay) | 100% | 100% |
| `loose_overlay` | hand-placed sticker: larger, offset, rotated, off-white paper, drop shadow | 100% | 100% |
| `branded_sticker` | large printed sticker with its own coloured frame | 100% | 100% |
| `partial_patch` | small patch rewriting 5–15% of the modules, away from the finder patterns | 90% | 100% |

The global mean dilutes a small patch across the whole image. The `local` score takes the
worst-reconstructed 16×16 window, so a contiguous sticker cannot hide. This is the same idea
as max-of-anomaly-map scoring on MVTec AD. Both scores come from the same model and are
calibrated the same way.

### Why not a supervised classifier? (tests Section 4 of the report)

`baseline_supervised.py` trains a CNN (the same encoder plus a classification head) on genuine
photos and 3 of the 4 attack types, then tests it on the held-out 4th type
(leave-one-attack-out):

| Attack type held out | Supervised CNN detection | Supervised AUC | QRGuard detection | QRGuard AUC |
|---|---|---|---|---|
| aligned_overlay | 99% | 1.000 | 100% | 1.000 |
| loose_overlay | 100% | 1.000 | 100% | 1.000 |
| branded_sticker | 100% | 1.000 | 100% | 1.000 |
| **partial_patch** | **12%** | 0.783 | **100%** | 1.000 |
| none (all 4 types seen) | 91.7% | 0.994 | 100% | 1.000 |

The full-sticker attacks look alike, so the classifier transfers between them. It misses 88%
of the attack style it was not trained on. The autoencoder uses no tampered data at all, so it
has no attack style to overfit to. The supervised model does have a lower false-positive rate
(0% vs 1.5%). That is the trade-off of having labelled attacks, and only for the attacks it
already knows.

### Other measurements

- Training: 60 epochs, about 5 minutes on an Apple-silicon GPU (MPS), 800 genuine photos.
- Model: 5.6 M parameters (22 MB float32).
- Inference: 6 ms per image for the autoencoder on one CPU thread (TorchScript). About 250 ms
  end to end in the web app, which is dominated by QR detection on full-resolution photos.

Figures in `results/`: `score_histogram.png`, `roc_curve.png`, `confusion_matrix.png`,
`examples.png` (input / reconstruction / error heatmap), `training_curve.png`,
`baseline_vs_autoencoder.png`. All numbers are in `results/metrics.json` and
`results/baseline_supervised.json`.

> **Caveat:** these are results on *simulated* phone photos (see "Data" below). They show that
> the method works and how it behaves. They are not a field accuracy figure. The next step is
> to repeat the evaluation on real photographs (see "Using real photos").

## Quick start

```bash
pip install -r requirements.txt
./run_pipeline.sh            # data → train → evaluate → supervised baseline → mobile export (~15 min)
python3 app.py               # web app on http://localhost:8000
```

Tip: Python buffers its output when redirected to a file. To watch progress live, use
`PYTHONUNBUFFERED=1 ./run_pipeline.sh`, or `python3 -u train.py` for a single step.

### Individual steps

| Command | What it does |
|---|---|
| `python3 generate_data.py` | Renders the genuine poster, simulates phone photos, and writes `data/synthetic/` |
| `python3 train.py` | Trains the autoencoder on genuine photos, calibrates thresholds, and saves `checkpoints/qrguard_ae.pt` |
| `python3 evaluate.py` | Test-set metrics and figures in `results/` |
| `python3 baseline_supervised.py` | Leave-one-attack-out supervised CNN comparison |
| `python3 detect.py photo.jpg ...` | Checks photos from the CLI and writes annotated images to `results/detections/` (exit code 1 if anything is flagged) |
| `python3 calibrate.py --genuine DIR` | Re-fits the threshold from a folder of genuine photos (no retraining) |
| `python3 export_mobile.py` | TorchScript + PyTorch Mobile (`.ptl`) model and `qrguard_mobile.json` for on-device use |
| `python3 app.py` | Phone-friendly web app (camera upload, verdict, reconstruction, error heatmap) |
| `python3 -m pytest -q` | Smoke tests |

**Using the web app from a phone:** run `python3 app.py` on a laptop, connect the phone to the
same Wi-Fi, and open `http://<laptop-ip>:8000`. Tapping the upload area opens the rear camera.
The app also has one-click test photos.

## How it works

1. **Preprocess** (`qrguard/preprocess.py`): OpenCV's QR detectors locate the three finder
   patterns. Corners from a successful decode are preferred because decoding validates the
   geometry, and plain detection is the fallback. A perspective warp deskews the QR plus a 25%
   margin of surrounding poster (sticker edges and shadows live there) into a 128×128 RGB
   patch in [0, 1]. The decoded payload is also returned, so the app shows where the code
   actually points.
2. **Autoencoder** (`qrguard/model.py`): four Conv2D blocks (stride 2, 3→256 channels,
   128→8 px), a dense 128-d bottleneck, then four ConvTranspose2D blocks and a sigmoid. The
   dense bottleneck stops the network from simply copying its input. It can only describe
   what it has seen, which is the genuine poster.
3. **Training** (`train.py`): MSE loss, Adam (lr 1e-3, halved on plateau), early stopping on
   validation MSE. Light augmentation (±3 px shift, per-channel gain, brightness) makes the
   model tolerant of the residual misalignment and lighting changes in genuine photos.
4. **Error map and score**: pixel-wise squared error averaged over RGB. The `mean` score is the
   global MSE; the `local` score is the maximum of the error map averaged over 16×16 windows.
5. **Threshold** (`qrguard/calibration.py`): reconstruction errors are positive and right-skewed
   (skewness 3.5 raw vs 1.1 after a log), so the threshold is `exp(mean + 3σ)` of the **log**
   scores on genuine validation photos, which is a log-normal tail bound. No tampered data
   is used.

## Data

`qrguard/synth.py` builds a reproducible dataset:

- **Genuine poster:** an "Amrita Canteen — Scan to Pay" UPI poster
  (`data/synthetic/poster_genuine.png`).
- **Tampering applied to the flat poster before the photo,** as happens physically. The
  malicious payloads are look-alike UPI IDs and URLs (`amritacanteen@okaxls`, `bit.ly/...`).
- **Phone-photo simulation:** random framing and scale, ±20° rotation, perspective, background
  wall texture, exposure, lighting gradient, vignette, colour cast, contrast, occasional shadow
  band, Gaussian and motion blur, sensor noise, and JPEG compression at quality 55–95.
- **Splits:** 800 train, 200 validation, 200 test genuine photos, plus 100 test photos per
  attack type. The supervised baseline gets its own 200-per-type tampered training pool,
  generated with a different seed.

## Using real photos

The code works the same on real photographs. Use this layout:

```
data/real/train/genuine/*.jpg      # 100+ photos of the genuine poster: different phones, angles, lighting
data/real/val/genuine/*.jpg        # 30–50 more genuine photos (threshold calibration)
data/real/test/genuine/*.jpg
data/real/test/tampered/*.jpg      # e.g. print data/synthetic/sticker_malicious.png and stick it on
```

```bash
python3 train.py --data data/real --init checkpoints/qrguard_ae.pt   # fine-tune the synthetic model
python3 evaluate.py --data data/real
```

To deploy at a new location without retraining, run `python3 calibrate.py --genuine <folder>`.
For a physical demo, print `poster_genuine.png`, photograph it, paste the printed
`sticker_malicious.png` over the QR, and photograph it again.

## Project layout

```
qrguard/            config, preprocess, model, data loading, calibration, detector, synth
generate_data.py    train.py    evaluate.py    baseline_supervised.py
detect.py           calibrate.py    export_mobile.py    app.py    templates/index.html
tests/              smoke tests
run_pipeline.sh     reproduces everything
```

## Limitations

- **One model per poster design.** That is the point of one-class training, but each new
  poster needs its own genuine photos. Fine-tuning from an existing checkpoint takes minutes.
- **The evaluation is synthetic.** Real printing, paper texture, glare and camera pipelines
  will widen the genuine error distribution, so the threshold must be re-calibrated on real
  photos.
- **The QR must be found.** If a sticker makes the code undetectable, the app reports
  "No QR code found" instead of a verdict. An unscannable code is itself suspicious on a
  payment poster.
- **Extreme lighting causes the remaining false positives** (very dark photos, or strong
  yellow or blue casts). Real-photo calibration and more lighting variety in training would
  reduce them.
