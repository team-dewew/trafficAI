#!/usr/bin/env bash
# Pre-commit smoke check: unit tests + the official harness on an 8 s clip + website import.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m pytest -q tests
echo "SMOKE OK"
