"""Assemble the website bundle for the Hugging Face Space into dist/space/.

The Space needs its own README header and CPU requirements, while the repository
root keeps the submission's GPU requirements, so the Space is built as a
separate folder instead of deploying the repository root:

    python scripts/build_space.py              # -> dist/space/
    # then commit dist/space/ to the Space's git repo (large files through git-lfs)

Weights are not bundled: the demo downloads yolo11s.pt (19 MB) on first use.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "space"

COPY = [
    "app.py", "solution.py", "predictions_samples.json", ".streamlit/config.toml",
    "weights/download.py", "weights/download.sh", "weights/SHA256SUMS",
]
TREES = ["src", "assets", "eda_results", "docs", "samples/previews", "samples/demo"]
FROM_SPACE = {"space/README.md": "README.md", "space/requirements.txt": "requirements.txt", "space/packages.txt": "packages.txt"}
GITATTRIBUTES = "\n".join(f"*.{ext} filter=lfs diff=lfs merge=lfs -text" for ext in ("mp4", "jpg", "png")) + "\n"


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for rel in COPY:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    for rel in TREES:
        shutil.copytree(ROOT / rel, OUT / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "raw"))
    for src, dst in FROM_SPACE.items():
        shutil.copy2(ROOT / src, OUT / dst)
    (OUT / ".gitattributes").write_text(GITATTRIBUTES)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"built {OUT} ({size / 1e6:.0f} MB, {sum(1 for f in OUT.rglob('*') if f.is_file())} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
