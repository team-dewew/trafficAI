import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    print("=" * 70, flush=True)
    print("STARTING FULL END-TO-END BENCHMARK OF SOLUTION.PY", flush=True)
    print("=" * 70, flush=True)

    # 1. Record start_time
    start_time = time.time()

    # 2. Execute run_submission.py
    cmd_run = [
        sys.executable,
        "-u",
        "run_submission.py",
        "--videos",
        "samples",
        "--out",
        "predictions_samples.json",
        "--team",
        "dewew",
    ]
    print(f"Executing: {' '.join(cmd_run)}", flush=True)
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    res_run = subprocess.run(cmd_run, env=env)
    if res_run.returncode != 0:
        print(f"Error: run_submission.py failed with exit code {res_run.returncode}", file=sys.stderr, flush=True)
        sys.exit(res_run.returncode)

    # 3. Record end_time and calculate total elapsed time in milliseconds
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000.0

    print("\n" + "=" * 70, flush=True)
    print("VALIDATING PREDICTIONS VIA EVALUATE.PY", flush=True)
    print("=" * 70, flush=True)

    # 4. Execute evaluate.py --pred predictions_samples.json --validate-only
    cmd_eval = [
        sys.executable,
        "-u",
        "evaluate.py",
        "--pred",
        "predictions_samples.json",
        "--validate-only",
    ]
    print(f"Executing: {' '.join(cmd_eval)}", flush=True)
    res_eval = subprocess.run(cmd_eval, capture_output=True, text=True, env=env)
    print(res_eval.stdout, flush=True)
    if res_eval.stderr:
        print(res_eval.stderr, file=sys.stderr, flush=True)
    if res_eval.returncode != 0:
        print(f"Validation failed with code {res_eval.returncode}", file=sys.stderr, flush=True)
        sys.exit(res_eval.returncode)

    print("=" * 70, flush=True)
    print("BENCHMARK SUMMARY & PREDICTIONS SNIPPET", flush=True)
    print("=" * 70, flush=True)

    # 5. Read predictions_samples.json
    pred_path = Path("predictions_samples.json")
    if not pred_path.exists():
        print("predictions_samples.json not found!", file=sys.stderr, flush=True)
        sys.exit(1)

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total Execution Time: {elapsed_ms:.2f} ms ({elapsed_ms / 1000.0:.2f} s)", flush=True)
    print(f"Team: {data.get('team', 'N/A')}", flush=True)
    print("\nGenerated Events Snippet (first 2 events per video):", flush=True)
    videos_dict = data.get("videos", {})
    for vid_name, vid_data in sorted(videos_dict.items()):
        events = vid_data.get("events", [])
        total_events = len(events)
        snippet = events[:2]
        print(f"\n  Video: {vid_name}", flush=True)
        print(f"    Total Events Detected: {total_events}", flush=True)
        print(f"    First 2 Events: {json.dumps(snippet, indent=6)}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
