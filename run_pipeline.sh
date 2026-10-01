#!/usr/bin/env bash
# Reproduce every QRGuard result from scratch: data -> training -> evaluation -> baseline -> export.
set -euo pipefail
cd "$(dirname "$0")"

python3 generate_data.py
python3 train.py
python3 evaluate.py
python3 baseline_supervised.py
python3 export_mobile.py
echo
echo "Done. Results in results/, model in checkpoints/. Start the app with: python3 app.py"
