#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Generating predictions..."
python run_submission.py --videos samples --out predictions_samples.json --team dewew

echo "Evaluating against dev set labels..."
python evaluate.py --pred predictions_samples.json --gt devset/labels.json --per-video --json devset/report.json

echo "Summarizing results..."
python src/devset/summarize.py
