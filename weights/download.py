"""Download the model weights (run once, with internet) and verify their checksums.

    python weights/download.py            # everything the submission and the website need

Sources (see README -> Install and run):
    yolo11l.pt        Ultralytics release v8.3.0, AGPL-3.0            (Part A detector)
    yolov8n.pt        Ultralytics release v8.3.0, AGPL-3.0            (Part B detector)
    yolo11s.pt        Ultralytics release v8.3.0, AGPL-3.0            (website CPU demo only)
    accident_model.pt Enos-123/accident-evaluator-yolov8x, weights/epoch90.pt, renamed; MIT
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
        tmp = path.with_suffix(".part")
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
