#!/usr/bin/env bash
# Download model weights before the offline evaluation run (max 5 GB)
set -e

WEIGHTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Downloading weights to $WEIGHTS_DIR..."

if [ ! -f "$WEIGHTS_DIR/yolo11l.pt" ]; then
    curl -L -o "$WEIGHTS_DIR/yolo11l.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
fi

echo "Weights download completed."
