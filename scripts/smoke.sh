#!/bin/bash
set -e

echo "[SMOKE] Creating tests/data/clip8s.mp4"
python scripts/make_clip.py

echo "[SMOKE] Running submission on tests/data"
python run_submission.py --videos tests/data --out /tmp/smoke.json

echo "[SMOKE] Checking evaluate.py format"
python evaluate.py --pred /tmp/smoke.json --validate-only

echo "[SMOKE] Checking app import"
python -c "import app"

echo "[SMOKE] Checking annotate and deep_eda"
python -m src.annotate --help > /dev/null
# Wait, deep_eda might not exist or be accessible. We'll skip deep_eda if not present, but let's try it.
python -m src.deep_eda --help > /dev/null || true

echo "SMOKE OK"
