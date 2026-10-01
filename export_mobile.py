"""Export the trained autoencoder for on-device inference (the report's 'on-device' pipeline).

Writes to checkpoints/:
  qrguard_mobile.ptl      TorchScript Lite model for PyTorch Mobile (Android / iOS)
  qrguard_traced.pt       plain TorchScript model (desktop / server)
  qrguard_mobile.json     threshold + preprocessing constants the app needs

The exported model takes a 1x3x128x128 RGB float tensor in [0,1] and returns
(reconstruction, score); the app compares score with the threshold in the JSON.
"""

import argparse
import json

import torch

from qrguard import config
from qrguard.model import anomaly_score, load_checkpoint


class ScoringModel(torch.nn.Module):
    def __init__(self, ae, mode: str):
        super().__init__()
        self.ae = ae
        self.mode = mode

    def forward(self, x):
        r = self.ae(x)
        return r, anomaly_score(x, r, mode=self.mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(config.DEFAULT_CHECKPOINT))
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, torch.device("cpu"))
    mode = ckpt.get("score_mode", "mean")
    wrapped = ScoringModel(model, mode).eval()
    example = torch.rand(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
        ref_r, ref_s = wrapped(example)
        r, s = traced(example)
    assert torch.allclose(ref_s, s, atol=1e-6), "traced model output mismatch"

    out = config.CHECKPOINT_DIR
    traced.save(str(out / "qrguard_traced.pt"))
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimize_for_mobile(traced)._save_for_lite_interpreter(str(out / "qrguard_mobile.ptl"))
        print(f"wrote {out / 'qrguard_mobile.ptl'}")
    except Exception as e:  # mobile optimizer is not available in every PyTorch build
        print(f"skipped .ptl export ({e}); qrguard_traced.pt is still usable")

    meta = {"threshold": ckpt["threshold"], "score_mode": mode, "local_window": config.LOCAL_WINDOW,
            "input_size": config.IMG_SIZE, "channels": "RGB",
            "value_range": [0, 1], "qr_margin": config.QR_MARGIN,
            "preprocess": "detect QR corners (TL,TR,BR,BL); perspective-warp so the QR spans the centre "
                          f"1/(1+2*{config.QR_MARGIN}) of a {config.IMG_SIZE}x{config.IMG_SIZE} patch",
            "decision": "tampered if score > threshold"}
    with open(out / "qrguard_mobile.json", "w") as f:
        json.dump(meta, f, indent=2)
    size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    print(f"wrote {out / 'qrguard_traced.pt'} and qrguard_mobile.json  (~{size_mb:.1f} MB float32 weights)")


if __name__ == "__main__":
    main()
