import csv
import json
import subprocess
import sys
from pathlib import Path

"""Cut an annotated review clip around every predicted event (devset/clips/) and
write devset/review.csv for quick TP/FP marking. Does not replace full labelling."""

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    devset_dir = repo_root / "devset"
    devset_dir.mkdir(exist_ok=True)
    clips_dir = devset_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    preds_file = repo_root / "predictions_samples.json"
    if not preds_file.exists():
        print(f"{preds_file} not found.")
        return

    with open(preds_file, "r") as f:
        preds = json.load(f)

    csv_path = devset_dir / "review.csv"
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video", "idx", "label", "start", "end", "clip_path", "verdict(TP/FP/?)", "true_start", "true_end", "true_label", "notes"])

        for video_name, data in preds.get("videos", {}).items():
            events = data.get("events", [])
            video_path = repo_root / "samples" / video_name
            if not video_path.exists():
                print(f"Video {video_path} not found.")
                continue

            vid_log = preds.get("log", {}).get(video_name, {})
            duration_meta = vid_log.get("duration", 0)

            for idx, event in enumerate(events):
                start, end, label = event
                clip_start = max(0, start - 3)
                clip_end = min(duration_meta or end + 3, end + 3)

                clip_filename = f"{Path(video_name).stem}_{idx:03d}_{label}.mp4"
                clip_out = clips_dir / clip_filename

                print(f"Generating clip {clip_out}...")

                cmd = [
                    sys.executable, "-m", "src.annotate",
                    "--video", str(video_path),
                    "--out", str(clip_out),
                    "--start", str(clip_start),
                    "--end", str(clip_end)
                ]
                # For simplicity, we write out the single event to a temporary file for annotate.py to render just this event banner
                tmp_json = devset_dir / "tmp_event.json"
                with open(tmp_json, "w") as tf:
                    json.dump({"videos": {video_name: {"events": [event]}}}, tf)

                cmd.extend(["--events", str(tmp_json)])

                try:
                    subprocess.run(cmd, check=True, cwd=repo_root)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to annotate clip {clip_out}: {e}")

                writer.writerow([video_name, idx, label, start, end, f"clips/{clip_filename}", "", "", "", "", ""])

if __name__ == "__main__":
    main()
