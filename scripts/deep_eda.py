#!/usr/bin/env python3
"""
deep_eda.py — one-pass EDA artifact generator for the sample videos.

Runs the detection pipeline once per sample video and, in that single pass,
produces every artifact the website's EDA section needs:

  eda_results/<VID>_counts.csv        object counts over time, per class (1 s buckets)
  eda_results/<VID>_density.csv       vehicles per minute over time
  eda_results/<VID>_heatmap.png       occupancy heatmap (vehicle + pedestrian centroids)
  eda_results/<VID>_trajectories.png  vehicle trajectory trails over the first frame
  eda_results/class_distribution.csv  total detections per class across all feeds
  samples/previews/<VID>_preview.mp4  fully annotated preview (via src/annotate.py)

Usage:
    python scripts/deep_eda.py [--videos samples] [--stride 2]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotate import render_annotated  # noqa: E402

CLASS_NAMES = ["car", "bus", "truck", "motorcycle", "bicycle", "pedestrian"]
COCO_IDS = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle", 1: "bicycle", 0: "pedestrian"}


class EdaSink:
    """Per-frame accumulator fed by render_annotated's frame_sink callback."""

    GRID_W, GRID_H = 96, 54  # heatmap grid cells (16:9)

    def __init__(self, frame_shape: tuple[int, int], fps: float, stride: int):
        self.fps = fps
        self.stride = stride
        self.h, self.w = frame_shape
        self.sec_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.sec_frames: dict[int, int] = defaultdict(int)
        self.class_totals: dict[str, int] = defaultdict(int)
        self.heat = np.zeros((self.GRID_H, self.GRID_W), dtype=np.float64)
        self.trail_canvas = None
        self.track_last: dict[int, tuple[int, int]] = {}

    def __call__(self, frame_idx: int, t_sec: float, frame: np.ndarray, tracked) -> None:
        if self.trail_canvas is None:
            self.trail_canvas = cv2.resize(frame, (1280, 720)).copy()

        sec = int(t_sec)
        self.sec_frames[sec] += 1
        counts_this_frame: dict[str, int] = defaultdict(int)

        if len(tracked) > 0:
            for i in range(len(tracked)):
                cname = COCO_IDS.get(int(tracked.class_id[i]))
                if cname is None:
                    continue
                counts_this_frame[cname] += 1
                self.class_totals[cname] += 1

                x1, y1, x2, y2 = tracked.xyxy[i]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                gx = min(self.GRID_W - 1, max(0, int(cx / self.w * self.GRID_W)))
                gy = min(self.GRID_H - 1, max(0, int(cy / self.h * self.GRID_H)))
                self.heat[gy, gx] += 1.0

                # Trajectory trails (vehicles only keeps the image readable)
                if cname != "pedestrian" and tracked.tracker_id is not None:
                    tid = int(tracked.tracker_id[i])
                    pt = (int(cx / self.w * 1280), int(cy / self.h * 720))
                    prev = self.track_last.get(tid)
                    if prev is not None:
                        cv2.line(self.trail_canvas, prev, pt, (0, 220, 255), 2, cv2.LINE_AA)
                    self.track_last[tid] = pt

        for cname in CLASS_NAMES:
            self.sec_counts[sec][cname] += counts_this_frame.get(cname, 0)

    # ------------------------------------------------------------------ #
    def write_counts_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_sec"] + CLASS_NAMES + ["vehicles_total"])
            for sec in sorted(self.sec_counts):
                row = self.sec_counts[sec]
                n = max(1, self.sec_frames[sec])
                # average objects visible per frame within this 1 s bucket
                vals = [round(row[c] / n, 2) for c in CLASS_NAMES]
                writer.writerow([sec] + vals + [round(sum(vals[:-1]), 2)])

    def write_density_csv(self, path: Path) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_min", "vehicles_per_frame"])
            minutes: dict[int, list[float]] = defaultdict(list)
            for sec, row in self.sec_counts.items():
                n = max(1, self.sec_frames[sec])
                veh = sum(row[c] for c in CLASS_NAMES[:-1]) / n
                minutes[sec // 60].append(veh)
            for m in sorted(minutes):
                writer.writerow([m, round(sum(minutes[m]) / len(minutes[m]), 2)])

    def write_heatmap_png(self, path: Path) -> None:
        heat = cv2.GaussianBlur(self.heat, (5, 5), 0)
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_img = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_img = cv2.resize(heat_img, (1280, 720), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(path), heat_img)

    def write_trajectories_png(self, path: Path) -> None:
        if self.trail_canvas is not None:
            cv2.imwrite(str(path), self.trail_canvas)


def process_video(video_path: Path, out_dir: Path, preview_dir: Path, stride: int) -> dict[str, int]:
    import cv2 as _cv2  # local alias to keep top imports light

    meta = _cv2.VideoCapture(str(video_path))
    fps = float(meta.get(_cv2.CAP_PROP_FPS) or 25.0)
    w = int(meta.get(_cv2.CAP_PROP_FRAME_WIDTH))
    h = int(meta.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    meta.release()

    sink = EdaSink((h, w), fps, stride)
    stem = video_path.stem
    render_annotated(
        str(video_path),
        str(preview_dir / f"{stem}_preview.mp4"),
        events=None,
        width=960,
        stride=stride,
        frame_sink=sink,
        progress_callback=lambda p: print(f"\r  {stem}: {p * 100:5.1f}%", end="", flush=True),
    )
    print()

    sink.write_counts_csv(out_dir / f"{stem}_counts.csv")
    sink.write_density_csv(out_dir / f"{stem}_density.csv")
    sink.write_heatmap_png(out_dir / f"{stem}_heatmap.png")
    sink.write_trajectories_png(out_dir / f"{stem}_trajectories.png")
    return sink.class_totals


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all EDA artifacts + annotated previews.")
    ap.add_argument("--videos", type=Path, default=Path("samples"))
    ap.add_argument("--out", type=Path, default=Path("eda_results"))
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    preview_dir = args.videos / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in args.videos.iterdir() if p.suffix.lower() == ".mp4" and p.is_file()
    )
    if not videos:
        print(f"[ERROR] no videos found in {args.videos}", file=sys.stderr)
        sys.exit(1)

    grand_totals: dict[str, int] = defaultdict(int)
    for video_path in videos:
        print(f"[EDA] {video_path.name}")
        totals = process_video(video_path, args.out, preview_dir, max(1, args.stride))
        for k, v in totals.items():
            grand_totals[k] += v

    with open(args.out / "class_distribution.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "detections"])
        for cname in CLASS_NAMES:
            writer.writerow([cname, grand_totals.get(cname, 0)])

    print(f"\n[SUCCESS] EDA artifacts -> {args.out}")
    print(f"[SUCCESS] Annotated previews -> {preview_dir}")


if __name__ == "__main__":
    main()
