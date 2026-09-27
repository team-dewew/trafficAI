#!/usr/bin/env bash
# Download model weights before the offline evaluation run (max 5 GB total).
# Sources:
#   yolo11l.pt        Ultralytics official release (AGPL-3.0)
#   yolov8n.pt        Ultralytics official release (AGPL-3.0)
#   yolo11s.pt        Ultralytics official release (AGPL-3.0), used only by the website's CPU demo
#   accident_model.pt Enos-123/accident-evaluator-yolov8x (Hugging Face, weights/epoch90.pt,
#                     renamed; MIT per model card; trained on Roboflow "Accident Evaluator")
#   InternVL2_5-1B/   OpenGVLab/InternVL2_5-1B (Hugging Face, whole repository at commit 9d423ea; MIT)
set -e

WEIGHTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Downloading weights to $WEIGHTS_DIR..."

fetch() {  # fetch <file> <url>: skip files that are already there
    if [ ! -f "$WEIGHTS_DIR/$1" ]; then
        curl -fL --retry 3 -o "$WEIGHTS_DIR/$1.part" "$2"
        mv "$WEIGHTS_DIR/$1.part" "$WEIGHTS_DIR/$1"
    fi
}

fetch yolo11l.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
fetch yolov8n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
fetch yolo11s.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
# Secondary anomaly detector (accident / fire / smoke), fine-tuned YOLOv8x.
# Downloaded as epoch90.pt and stored under the name the pipeline expects.
fetch accident_model.pt "https://huggingface.co/Enos-123/accident-evaluator-yolov8x/resolve/main/weights/epoch90.pt"

# Vision-language model that verifies accident candidates: OpenGVLab/InternVL2_5-1B (MIT),
# the whole repository (weights, code, tokenizer, configs) pinned to one commit.
VLM_REPO="https://huggingface.co/OpenGVLab/InternVL2_5-1B/resolve/9d423ea1ae9f893897ee3f7493141073f5afcf22"
for f in .gitattributes README.md added_tokens.json config.json configuration.json \
         configuration_intern_vit.py configuration_internvl_chat.py conversation.py \
         examples/image1.jpg examples/image2.jpg examples/red-panda.mp4 generation_config.json \
         merges.txt model.safetensors modeling_intern_vit.py modeling_internvl_chat.py \
         preprocessor_config.json \
         runs/Nov22_02-53-47_HOST-10-140-60-109/events.out.tfevents.1732215525.HOST-10-140-60-109.69855.0 \
         special_tokens_map.json tokenizer_config.json vocab.json; do
    mkdir -p "$WEIGHTS_DIR/InternVL2_5-1B/$(dirname "$f")"
    fetch "InternVL2_5-1B/$f" "$VLM_REPO/$f"
done

echo "Verifying checksums..."
# tr strips carriage returns in case the checksum file was checked out with CRLF line endings
cd "$WEIGHTS_DIR" && tr -d '\r' < SHA256SUMS | sha256sum -c -

echo "Weights download completed."
