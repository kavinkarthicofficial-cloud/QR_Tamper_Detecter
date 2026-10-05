# QRGuard dataset card

QRGuard is a **one-class** detector: the autoencoder is trained on genuine photos of one poster and never sees a
tampered photo. Tampered photos exist only to **test** it. This card says exactly what the data is, where it comes
from, how the tampered inputs are generated, and what is still simulated.

## 1. Where the data comes from

| Dataset | What it is | Size | How to get it |
|---|---|---|---|
| **Synthetic demo poster** | One generated "Amrita Canteen, Scan to Pay" UPI poster (`qrguard/synth.py`), turned into simulated phone photos | train 800, val 200 genuine; test 200 genuine + 400 tampered (4 attacks x 100) | `python generate_data.py` (about 9 minutes, about 400 MB, not in the repo) |
| **Fresh synthetic test set** | Same generator, different seed (2026), never used for any design decision | 200 genuine + 400 tampered | `python generate_data.py --name synthetic_fresh --seed 2026 --train 0 --val 0` |
| **Real QR photos from the web** | 40 freely licensed photographs of QR codes in the wild (street signs, shop and charity codes, posters, plaques, business cards) from Wikimedia Commons | 40 photos in `data/web/raw/` | Included in the repo; re-download with `python fetch_web_qr.py` |
| **Per-poster simulated shots** | For each of the 40 real photos: simulated phone shots of that photo | per poster: 300 train, 80 val, 80 test genuine, 40 per attack x 4 tampered | `python web_experiment.py` (about 2 hours for all 40) |
| **Supervised-baseline pool** | Tampered training photos for the comparison classifier only (separate seed, disjoint from the test set) | 200 per attack type | made by `baseline_supervised.py` |

The real web photos are credited in [ATTRIBUTION.md](ATTRIBUTION.md): CC0 (6), CC BY (4), CC BY-SA (30). From 97
photos examined, a QR was found in 79 and decoded in 61 (`data/web/raw/detection_stats.json`); the 40 kept are those with a
decodable QR at least 150 px wide.

**Not in the dataset yet:** photos of a physically printed poster with a physically stuck-on sticker. The real photos
are real, but the extra shots of each poster are simulated and the stickers are composited digitally.

## 2. How the tampered inputs are generated

The sticker is applied to the **flat poster first**, then the whole poster is turned into a simulated phone photo.
That matches the real order of events (the attacker sticks paper on the poster, then the victim photographs it).
The malicious payloads are look-alike UPI IDs and links (for example `amritacanteen@okaxls`, `bit.ly/...`).

| Attack (`TAMPER_TYPES`) | What the attacker does | Details in `synth.tamper` |
|---|---|---|
| `aligned_overlay` | Same-size sticker placed exactly over the QR | white paper, no rotation, no shadow: only the QR modules differ (the hardest full overlay) |
| `loose_overlay` | Hand-placed sticker | 1.0-1.15x larger, offset up to 8 px, rotated up to 5 degrees, off-white tinted paper, drop shadow |
| `branded_sticker` | Large printed sticker with its own coloured frame | 1.12-1.3x larger, red / green / blue / orange frame, rotation up to 3 degrees, shadow 70% of the time |
| `partial_patch` | Small patch rewriting 5-15% of the QR modules, away from the three finder patterns | patch 22-38% of the QR side, tinted paper, rotation up to 2 degrees, shadow 50% of the time |

For real photos (`web_experiment.py`) the same stickers are applied on the rectified photo and, for one check, warped
back onto the original photo in its true perspective (`synth.tamper_real_photo`).

## 3. How a phone photo is simulated (`synth.simulate_photo`)

Random framing (QR is 32-55% of the frame) and position, rotation up to 20 degrees, perspective jitter, textured wall
background, exposure x0.65-1.15, lighting gradient, vignette, colour cast, contrast 0.8-1.15, an occasional soft shadow
band (25%), Gaussian blur sigma 0.3-1.4 (70%), occasional motion blur (15%), sensor noise, and JPEG quality 55-95.
Genuine photos get the same treatment, so blur, rotation and darkness are learned as **normal**, not as attacks.

## 4. Splits and leakage

* The autoencoder trains on `train/genuine` only; `val/genuine` is used for early stopping and for the threshold.
  No tampered photo is used for training or for the threshold.
* Every test photo is generated with its own random draw; none is an augmented copy of a training photo.
* The "fresh" test set uses a different seed, and `robustness.py` degrades genuine and tampered **test** photos only.
* For the web experiment each real photo is one poster with its own model; its test shots use different seeds from its
  training shots.

## 5. Preprocessing (identical for every photo)

OpenCV finds the QR finder patterns, a perspective warp straightens the QR plus a 25% margin of surrounding poster (sticker
edges and shadows live there) into a 128x128 RGB patch scaled to [0, 1]. If no QR is found the app says "No QR code found"
instead of guessing.

## 6. Enrolment (how a new poster gets a model)

`enroll.py` / the web app take 1-5 photos of the poster in its **genuine** state, expand each into 300 simulated phone shots
(80 more for the threshold), train a fresh autoencoder (about 1-2 minutes) and record: the threshold, the enrolled QR text,
and the range of photo quality seen at enrolment (used by the quality gate). If the enrolment photo already contains a sticker,
the model learns the sticker as genuine, so enrol from a known-good photo.
