#!/usr/bin/env bash
# Download model weights before the offline evaluation run (max 5 GB total).
# Sources:
#   yolo11l.pt        Ultralytics official release (AGPL-3.0)
#   yolov8n.pt        Ultralytics official release (AGPL-3.0)
#   yolo11s.pt        Ultralytics official release (AGPL-3.0), used only by the website's CPU demo
#   accident_model.pt Enos-123/accident-evaluator-yolov8x (Hugging Face, weights/epoch90.pt,
#                     renamed; MIT per model card; trained on Roboflow "Accident Evaluator")
set -e

WEIGHTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Downloading weights to $WEIGHTS_DIR..."

if [ ! -f "$WEIGHTS_DIR/yolo11l.pt" ]; then
    curl -fL --retry 3 -o "$WEIGHTS_DIR/yolo11l.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
fi

if [ ! -f "$WEIGHTS_DIR/yolov8n.pt" ]; then
    curl -fL --retry 3 -o "$WEIGHTS_DIR/yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
fi

if [ ! -f "$WEIGHTS_DIR/yolo11s.pt" ]; then
    curl -fL --retry 3 -o "$WEIGHTS_DIR/yolo11s.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
fi

# Secondary anomaly detector (accident / fire / smoke), fine-tuned YOLOv8x.
# Downloaded as epoch90.pt and stored under the name the pipeline expects.
if [ ! -f "$WEIGHTS_DIR/accident_model.pt" ]; then
    curl -fL --retry 3 -o "$WEIGHTS_DIR/accident_model.pt" "https://huggingface.co/Enos-123/accident-evaluator-yolov8x/resolve/main/weights/epoch90.pt"
fi

echo "Verifying checksums..."
cd "$WEIGHTS_DIR" && sha256sum -c SHA256SUMS

echo "Weights download completed."

