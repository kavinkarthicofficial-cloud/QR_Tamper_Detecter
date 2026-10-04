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

**Fresh test set.** The score type and threshold rule were chosen during development, while
looking at results from the same generator. To check that this did not inflate the numbers,
600 new test photos were generated with a different seed (2026), which nothing in the project
had seen before, and the same model was evaluated unchanged
(`generate_data.py --name synthetic_fresh --seed 2026 --train 0 --val 0`). Result with the
`local` score: AUC 0.9999, accuracy 99.8%, recall 100%, and 1 false positive out of 200 (0.5%).
With the `mean` score, partial-patch detection was 83%. Figures are in `results/fresh_test/`.

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

## Results on 40 real QR photos from the web

`fetch_web_qr.py` downloaded freely licensed photographs of QR codes in the wild from
Wikimedia Commons: street signs, shop and charity payment codes, posters, museum plaques,
business cards and notices from many countries. Credits are in [ATTRIBUTION.md](ATTRIBUTION.md).
`web_experiment.py` then treats **each real photo as one deployed poster** and runs the full
QRGuard workflow on it:

1. Rectify the photo so the real QR is axis-aligned, keeping the real surroundings as context.
2. **Train a new autoencoder** for that poster on 300 phone shots simulated from the real
   photo (genuine only), and calibrate its threshold on 80 more.
3. **Test** it on 80 new genuine shots and 40 shots per attack, with the stickers placed over
   the real QR. Test it also on the **untouched original web photo**, and on the original
   photo with each sticker warped onto it in its true perspective.

| Test (40 posters, `local` score) | Result |
|---|---|
| ROC AUC per poster | mean **0.9997**, median 1.000, worst 0.9958. All 40 posters ≥ 0.99 |
| Tampered shots detected | **98.1%** (6,350 shots) |
| Genuine shots falsely flagged | **0.48%** (3,125 shots) |
| Accuracy | 98.6% |
| Original web photo accepted as genuine | **39 / 40** (97.5%) |
| Original photo with a perspective-correct sticker flagged | **160 / 160** (100%) |

| Attack | `mean` score | `local` score |
|---|---|---|
| aligned_overlay | 99.5% | 99.1% |
| loose_overlay | 99.7% | 98.4% |
| branded_sticker | 100% | 98.9% |
| partial_patch | **34.8%** | **96.1%** |
| false-positive rate | 0.22% | 0.48% |

On real posters the gap between the two scores is much larger than on the synthetic one.
The global mean catches only about a third of the small patch attacks, while the local score
catches 96%. That confirms the local score is the right choice for deployment.

**QR finder on raw web photos (real data, no simulation):** out of 97 photos examined, the QR
was located in 81% and decoded in 63%. Photos where it failed mostly had small, distant or
heavily angled codes. In the simulated shots, the QR was not found in 2.3% of genuine and
0.8% of tampered shots; those were excluded from the scores above.

**A training bug found and fixed along the way.** In a first run (archived in
`results/web_run1_early_stopping/`), 3 of the first 21 posters had weak detection. Their
models had early-stopped at 18–25 epochs while still blurry, and had 4–6× higher genuine
validation error. Retraining one of them without early stopping showed that validation error
kept falling, and its threshold dropped about 3×. The final run uses the same 60-epoch budget
as the main model and allows early stopping only after epoch 40. That rule depends on
validation data alone. Every poster was then re-run under it.

Figures: `results/web/summary.png` (detection per attack and per-poster AUC) and
`results/web/examples.png` (original photo, stickered, and partial patch, with error maps).
Per-poster numbers are in `results/web/per_poster.csv` and `results/web/log.txt`.

> **What is still simulated:** the photos and QR codes are real. The extra phone shots of each
> poster are simulated, and the stickers are composited digitally rather than printed and
> stuck on. The last step is a physical test: print a poster, take about 50 phone photos,
> paste a printed sticker on it, and photograph it again (see "Using real photos").

## Check your own QR code

QRGuard compares a photo with what **that specific poster** looks like when genuine, so it
cannot judge a QR it has never seen. You enrol a poster first, from one or more photos of its
genuine state. That takes 1–3 minutes, and after it you can check any later photo of the poster:

```bash
python3 enroll.py genuine_poster.jpg --name canteen-upi       # one-time, 1–3 min
python3 detect.py --poster canteen-upi todays_photo.jpg       # GENUINE or TAMPERED
python3 detect.py --list                                      # posters it can check
```

In the web app (`python3 app.py`), choose the poster from the **Poster** list, or use
**Enrol a new poster** to upload genuine photos from your phone. When training finishes, the new
poster is selected automatically.

Example with a real web photo the system had never seen (`data/web/raw/web_010.jpg`, enrolled
in 72 s):

| Photo checked | Verdict | Score / threshold |
|---|---|---|
| the genuine photo | GENUINE | 0.6× |
| same photo with a pasted sticker | TAMPERED | 11× |
| same photo with a small partial patch | TAMPERED | 11.6× |
| a completely different QR code | TAMPERED | 14× |

**Important:** enrol from photos you know are genuine, for example when the poster is first
put up. If the enrolment photo already has a sticker on it, QRGuard learns the sticker as the
genuine poster. A few photos at different angles and in different light make the model more
robust than a single photo.

## Quick start

The repo includes the trained demo model (`checkpoints/qrguard_ae.pt`, the model behind the
synthetic results above) and the 40 real web photos (`data/web/raw/`). After cloning, you can
check and enrol posters straight away:

```bash
pip install -r requirements.txt
python3 app.py               # web app on http://localhost:8000 (no training needed)
python3 enroll.py data/web/raw/web_010.jpg --name bayern-sign     # enrol a real poster (~1-2 min)
python3 detect.py --poster bayern-sign data/web/raw/web_010.jpg   # -> GENUINE
```

To reproduce everything from scratch, including the synthetic dataset (about 760 MB, not in
the repo) and the app's one-click test photos:

```bash
./run_pipeline.sh            # data → train → evaluate → supervised baseline → mobile export (~15 min)
```

On Windows, use `python` instead of `python3`, and run the scripts listed in `run_pipeline.sh`
one at a time.

Tip: Python buffers its output when redirected to a file. To watch progress live, use
`PYTHONUNBUFFERED=1 ./run_pipeline.sh`, or `python3 -u train.py` for a single step.

### Individual steps

| Command | What it does |
|---|---|
| `python3 generate_data.py` | Renders the genuine poster, simulates phone photos, and writes `data/synthetic/` |
| `python3 train.py` | Trains the autoencoder on genuine photos, calibrates thresholds, and saves `checkpoints/qrguard_ae.pt` |
| `python3 evaluate.py` | Test-set metrics and figures in `results/` |
| `python3 baseline_supervised.py` | Leave-one-attack-out supervised CNN comparison |
| `python3 detect.py [--poster NAME] photo.jpg ...` | Checks photos from the CLI and writes annotated images to `results/detections/` (exit code 1 if anything is flagged) |
| `python3 calibrate.py --genuine DIR` | Re-fits the threshold from a folder of genuine photos (no retraining) |
| `python3 export_mobile.py` | TorchScript + PyTorch Mobile (`.ptl`) model and `qrguard_mobile.json` for on-device use |
| `python3 app.py` | Phone-friendly web app (camera upload, verdict, reconstruction, error heatmap) |
| `python3 -m pytest -q` | Smoke tests |
| `python3 enroll.py photo.jpg --name NAME` | Enrols your own poster from genuine photo(s) so it can be checked |
| `python3 fetch_web_qr.py` | Downloads real QR photos from Wikimedia Commons into `data/web/raw/` with attribution |
| `python3 web_experiment.py` | Trains and tests one model per real photo (about 2–4 min per poster; resumable, skips finished posters) |

The web experiment takes about 2 hours for 40 posters. To keep it running after you close the
terminal, start it with `nohup caffeinate -ims python3 web_experiment.py > web.log 2>&1 &`
and keep the laptop lid open.

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
enroll.py           enrol your own poster (qrguard/enrollment.py)
fetch_web_qr.py     web_experiment.py    ATTRIBUTION.md      (real web QR photos)
tests/              smoke tests
run_pipeline.sh     reproduces the synthetic results
```

## Limitations

- **One model per poster design.** That is the point of one-class training, but each new
  poster needs its own genuine photos. Fine-tuning from an existing checkpoint takes minutes.
- **No physical tampering has been tested yet.** The real-photo experiment uses real posters
  and QR codes, but simulated extra shots and digitally composited stickers. Real glare, paper
  texture and different phone cameras will widen the genuine error distribution, so the
  threshold should be re-calibrated on real photos of the deployed poster.
- **The QR must be found.** If a sticker makes the code undetectable, the app reports
  "No QR code found" instead of a verdict. An unscannable code is itself suspicious on a
  payment poster.
- **Extreme lighting causes the remaining false positives** (very dark photos, or strong
  yellow or blue casts). Real-photo calibration and more lighting variety in training would
  reduce them.
