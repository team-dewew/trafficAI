#!/usr/bin/env bash
# Download model weights before the offline evaluation run (max 5 GB)
set -e

WEIGHTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Downloading weights to $WEIGHTS_DIR..."

# Example:
# curl -L -o "$WEIGHTS_DIR/yolov8x.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt"

echo "Weights download completed."
