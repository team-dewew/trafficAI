import json
import subprocess
from pathlib import Path

import sys

def test_smoke():
    # 1. Run run_submission.py on tests/data
    out_json = Path("smoke.json")
    if out_json.exists():
        out_json.unlink()
        
    cmd = [sys.executable, "run_submission.py", "--videos", "tests/data", "--out", str(out_json)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"run_submission.py failed:\n{result.stderr}"
    
    # 2. Check JSON
    assert out_json.exists()
    with open(out_json, "r") as f:
        data = json.load(f)
        
    assert "videos" in data
    assert "clip8s.mp4" in data["videos"]
    
    # Check log for errors
    assert "log" in data
    assert "clip8s.mp4" in data["log"]
    assert len(data["log"]["clip8s.mp4"].get("errors", [])) == 0, "Errors found in log"
    
    # Check evaluate.py
    cmd_eval = [sys.executable, "evaluate.py", "--pred", str(out_json), "--validate-only"]
    result_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
    assert result_eval.returncode == 0, f"evaluate.py failed:\n{result_eval.stderr}"

    # Check app import
    cmd_app = [sys.executable, "-c", "import app"]
    result_app = subprocess.run(cmd_app, capture_output=True, text=True)
    assert result_app.returncode == 0, f"app.py import failed:\n{result_app.stderr}"

if __name__ == "__main__":
    test_smoke()
    print("SMOKE OK")
