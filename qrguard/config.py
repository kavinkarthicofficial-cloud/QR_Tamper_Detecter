"""Shared constants for the QRGuard pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

# Preprocessing
IMG_SIZE = 128            # autoencoder input is IMG_SIZE x IMG_SIZE x 3
QR_MARGIN = 0.25          # poster context kept around the QR, as a fraction of the QR side
MAX_DETECT_SIDE = 1600    # large phone photos are downscaled to this before QR detection

# Model / training
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
PATIENCE = 12             # early-stopping patience (epochs without val improvement)

# Anomaly score: "mean" = global pixel-wise MSE (report baseline);
# "local" = max of the error map averaged over LOCAL_WINDOW x LOCAL_WINDOW windows. Stickers are
# spatially contiguous, so the local score is far more sensitive to small patches.
SCORE_MODES = ("mean", "local")
SCORE_MODE = "local"
LOCAL_WINDOW = 16

# Threshold calibration: threshold = exp(mean + K_SIGMA * std) of log validation (genuine) scores
K_SIGMA = 3.0

DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "qrguard_ae.pt"
