"""Convert hand labels (one CSV per video: start,end,label,note) into evaluate.py's
ground-truth JSON. Labels may be any of the 14 official ids."""
import csv
import json
from pathlib import Path

import cv2

# Annotators may use any of the 14 official ids (not only the ones we predict).
from src.config import OFFICIAL_CLASSES as CLASSES


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            # Overlap, merge and warn
            print(f"Warning: Overlapping segments merged: {last} and {current}")
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    return merged

def process_csvs(raw_dir: str, video_dir: str, out_json: str):
    raw_path = Path(raw_dir)
    vid_path = Path(video_dir)

    if not raw_path.exists():
        print(f"Directory {raw_dir} does not exist.")
        return

    result = {}

    for csv_file in raw_path.glob("*.csv"):
        video_name = csv_file.stem
        # Try to find corresponding video to get duration and fps
        # We assume video_name has no extension, or matches stem.
        # Let's find first matching video
        possible_vids = list(vid_path.glob(f"{video_name}.*"))
        if not possible_vids:
            print(f"Warning: Video for {csv_file.name} not found in {video_dir}.")
            continue

        vid = possible_vids[0]
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps if fps > 0 else 0.0
        cap.release()

        events_by_class = {}

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    start = float(row["start"])
                    end = float(row["end"])
                    label = row["label"]
                except ValueError:
                    continue

                if label not in CLASSES:
                    print(f"Warning: Invalid class '{label}' in {csv_file.name}")
                    continue
                if start >= end:
                    print(f"Warning: start >= end in {csv_file.name}: {start}-{end}")
                    continue
                if end > duration:
                    end = duration

                if label not in events_by_class:
                    events_by_class[label] = []
                events_by_class[label].append([start, end])

        # Merge overlapping segments
        final_events = []
        for label, intervals in events_by_class.items():
            merged = merge_intervals(intervals)
            for m in merged:
                final_events.append([round(m[0], 3), round(m[1], 3), label])

        result[vid.name] = {
            "duration": round(duration, 3),
            "fps": round(fps, 3),
            "events": final_events
        }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)
    print(f"Saved GT to {out_json}")

if __name__ == "__main__":
    import sys

    # python -m src.devset.csv_to_gt [raw_csv_dir] [video_dir] [out_json]
    args = sys.argv[1:] + ["devset/raw", "samples", "devset/labels.json"][len(sys.argv) - 1:]
    process_csvs(*args[:3])
