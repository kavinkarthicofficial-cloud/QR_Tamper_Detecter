# QRGuard threat model, scope and related work

## The attack
An attacker prints a QR code that points to a phishing page or a fraudulent payment, and sticks it **over the genuine QR on
a public poster, sign or payment stand** ("QR sticker overlay" / physical quishing). From a distance the poster still looks
legitimate. A normal QR scanner only decodes the code and cannot tell that it is not the one the poster owner put there.

## What QRGuard does
Before the user scans the code they photograph the poster. QRGuard answers one question: **does the QR area still look like
the poster that was enrolled?** An autoencoder trained only on genuine photos of that poster reconstructs the genuine QR well
and a replaced or altered QR badly (see `results/attacks_gallery.png`). Two cheap deterministic checks sit around it:

* **URL cross-check:** the decoded QR text is compared with the text recorded at enrolment; a difference is reported as tampered.
* **Photo-quality gate:** a flagged photo that is blurrier, darker or more over-exposed than the enrolment photos becomes
  "retake the photo" instead of a false alarm. It can never turn a verdict into "genuine".

An **Investigator** agent then explains the result in plain language. It cannot change the verdict.

## Deployment story
The **poster owner** (a college, a shop) enrols the poster once, when it is put up, from known-good photos. Anyone with the
app (students, security staff) can then check a photo of that poster at any time. One model per poster design is by design:
that is what lets a different QR, which is just another valid black-and-white code, be recognised as different.

## Assumptions
* The attacker can physically reach a public poster and cover or alter its QR region.
* The poster was genuine when it was enrolled.
* The user can take one reasonably sharp photo with the whole QR code and a little of the poster around it.

## Out of scope (QRGuard does not claim to solve these)
| Case | Why |
|---|---|
| The poster was malicious from the start, or was enrolled with the sticker already on | There is no genuine reference |
| The official website is hacked, or the link later redirects somewhere bad | A web-security problem, not a physical one |
| An unenrolled poster | QRGuard needs a reference; an unknown QR is reported as different |
| An attacker who replaces the whole poster with a perfect reprint | The physical artefacts QRGuard looks for may be absent |
| Very poor photos | Reported as "retake", not judged |

## Known limits (see the results page for the numbers)
* Tampering in all tests is simulated or digitally composited. A physical printed-sticker test has not been done.
* Strong lighting changes and blur can still cause false alarms; the quality gate removes some of them and sends a small
  share of genuinely tampered photos to "retake" (never to "genuine"). Measured in `results/robustness*.json`.
* An unscannable or covered code is itself a warning sign on a payment poster; the app says so.
* The Investigator needs the free Gemini key and internet for AI text; otherwise a built-in template explains the result.

## Related work and positioning
A literature and market search (search tool restricted to US results; a few targeted queries, 30 September 2026) found the
following. We found **no prior work applying reconstruction-based anomaly detection (an autoencoder trained only on genuine
photos) to physical QR sticker tampering**. Absence of search results is not proof, so this is stated as "we found none".

| Work | What it does | How QRGuard differs |
|---|---|---|
| Trad and Chehab (2025), arXiv:2505.03451 | ML on QR pixel structure to detect quishing (XGBoost, AUC 0.9106); supervised; does not address physical sticker overlays | QRGuard checks the physical poster and uses no tampered training data |
| Alsuhibany (2025), *Sensors* 25(13), 3855 | Tamper-proof QR codes using a digital watermark added when the code is generated | Works only for codes made that way; QRGuard works on codes already deployed |
| Kurniasari, Jatmika and Arifin (2025), ICoBITS | MobileNetV2 classifier for visual authenticity of QRIS payment codes, including counterfeit stickers; 60 images (30 genuine, 17 dummy, 13 tampered), 94.3% test accuracy | Supervised, needs labelled tampered examples; QRGuard needs none |
| QRensic (GitHub prototype, OpenCV AI Competition 2026) | Investigates physical QR overlays with classical OpenCV measurements (reflections, edges, texture) and an LLM agent | Closest in goal. No autoencoder; QRGuard learns what one poster looks like instead of measuring generic surface cues |
| Consumer scanners and enterprise tools (Kaspersky, Norton and others) | Decode the QR and check the URL against threat databases | They do not look at the poster; we did not verify each product's features individually |
| MVTec AD benchmark (Bergmann et al., CVPR 2019) | Reconstruction-error anomaly detection for industrial defects | The same idea, applied to a new problem: one poster, one-class training |

Honest claim: QRGuard is an **application and pipeline** (per-poster one-class autoencoder, local error score, enrolment,
URL cross-check, quality gate, explanation agent) for physical QR tampering. It does not claim a new algorithm.
