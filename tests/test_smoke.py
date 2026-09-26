"""End-to-end smoke test through the official harness on an 8 s clip.

The clip is cut from samples/C3905.MP4 by scripts/make_clip.py (samples are not
committed); the test is skipped when neither the clip nor the sample exists.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "tests" / "data" / "clip8s.mp4"


@pytest.fixture(scope="module")
def clip() -> Path:
    if not CLIP.exists():
        if not (ROOT / "samples" / "C3905.MP4").exists():
            pytest.skip("no sample video available")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_clip.py")], check=True, cwd=ROOT)
    return CLIP


def test_harness_runs_clean(clip: Path, tmp_path: Path):
    out = tmp_path / "smoke.json"
    res = subprocess.run(
        [sys.executable, "run_submission.py", "--videos", str(clip), "--out", str(out), "--time-factor", "6"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    data = json.loads(out.read_text())
    log = data["log"][clip.name]
    assert log["errors"] == [], log["errors"]
    risk = data["videos"][clip.name]["risk"]
    assert len(risk) > 200 and all(0.0 <= s <= 1.0 for _, s in risk)

    val = subprocess.run([sys.executable, "evaluate.py", "--pred", str(out), "--validate-only"],
                         cwd=ROOT, capture_output=True, text=True)
    assert val.returncode == 0, val.stdout + val.stderr


def test_website_imports():
    res = subprocess.run([sys.executable, "-c", "import app"], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
