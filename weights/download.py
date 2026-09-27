"""Download the model weights (run once, with internet) and verify their checksums.

    python weights/download.py            # everything the submission and the website need

Sources (see README -> Install and run):
    yolo11l.pt        Ultralytics release v8.3.0, AGPL-3.0            (Part A detector)
    yolov8n.pt        Ultralytics release v8.3.0, AGPL-3.0            (Part B detector)
    yolo11s.pt        Ultralytics release v8.3.0, AGPL-3.0            (website CPU demo only)
    accident_model.pt Enos-123/accident-evaluator-yolov8x, weights/epoch90.pt, renamed; MIT
    InternVL2_5-1B/   OpenGVLab/InternVL2_5-1B, the whole repository at commit 9d423ea; MIT
                      (vision-language model that verifies accident candidates)
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent

FILES = {
    "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    "accident_model.pt": "https://huggingface.co/Enos-123/accident-evaluator-yolov8x/resolve/main/weights/epoch90.pt",
}

# OpenGVLab/InternVL2_5-1B, every file of the repository, pinned to one commit so the download never changes
VLM_DIR = "InternVL2_5-1B"
VLM_REPO = "https://huggingface.co/OpenGVLab/InternVL2_5-1B/resolve/9d423ea1ae9f893897ee3f7493141073f5afcf22"
VLM_FILES = [
    ".gitattributes", "README.md", "added_tokens.json", "config.json", "configuration.json",
    "configuration_intern_vit.py", "configuration_internvl_chat.py", "conversation.py",
    "examples/image1.jpg", "examples/image2.jpg", "examples/red-panda.mp4", "generation_config.json",
    "merges.txt", "model.safetensors", "modeling_intern_vit.py", "modeling_internvl_chat.py",
    "preprocessor_config.json",
    "runs/Nov22_02-53-47_HOST-10-140-60-109/events.out.tfevents.1732215525.HOST-10-140-60-109.69855.0",
    "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
]
FILES.update({f"{VLM_DIR}/{f}": f"{VLM_REPO}/{f}" for f in VLM_FILES})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_checksums() -> dict[str, str]:
    sums = {}
    for line in (WEIGHTS_DIR / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sums[parts[1].lstrip("*")] = parts[0]
    return sums


def missing(names=None) -> list[str]:
    """Weight files (of `names`, default all) that are absent."""
    return [n for n in (names or FILES) if not (WEIGHTS_DIR / n).exists()]


def download(names=None, log=print) -> None:
    """Fetch the requested weights (default all) that are not present yet, then verify them."""
    sums = expected_checksums()
    for name in names or FILES:
        path = WEIGHTS_DIR / name
        if path.exists():
            continue
        log(f"Downloading {name} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(FILES[name], tmp)
            if name in sums and _sha256(tmp) != sums[name]:
                raise RuntimeError(f"checksum mismatch for {name}")
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()
        log(f"{name}: OK")


def verify() -> bool:
    ok = True
    for name, expected in expected_checksums().items():
        path = WEIGHTS_DIR / name
        if not path.exists():
            print(f"missing: {name}")
            ok = False
        elif _sha256(path) != expected:
            print(f"checksum mismatch: {name}")
            ok = False
        else:
            print(f"{name}: OK")
    return ok


if __name__ == "__main__":
    download()
    sys.exit(0 if verify() else 1)

