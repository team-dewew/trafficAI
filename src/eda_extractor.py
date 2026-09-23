#!/usr/bin/env python3
"""
eda_extractor.py — Exploratory Data Analysis (EDA) helper script for video metadata and frame extraction.

Functions:
  1. Scans a specified folder (e.g., samples/) for .mp4 video files.
  2. Extracts resolution (Width x Height), FPS, total frame count, duration in seconds.
  3. Saves the first frame of each video as a .jpg in eda_results/frames/.
  4. Saves all extracted metadata into eda_results/metadata.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".MP4", ".avi", ".mov", ".mkv"}


def extract_video_metadata_and_first_frame(
    video_path: Path, frames_dir: Path
) -> dict:
    """Extract metadata and the very first frame for a given video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}", file=sys.stderr)
        return {
            "video_name": video_path.name,
            "width": 0,
            "height": 0,
            "resolution": "0x0",
            "fps": 0.0,
            "total_frames": 0,
            "duration_sec": 0.0,
            "first_frame_path": "",
            "status": "failed_to_open",
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 3) if fps > 0 else 0.0

    # Extract first frame
    ret, frame = cap.read()
    cap.release()

    first_frame_filename = f"{video_path.stem}_first_frame.jpg"
    first_frame_path = frames_dir / first_frame_filename

    if ret and frame is not None:
        cv2.imwrite(str(first_frame_path), frame)
        saved_frame_str = str(first_frame_path)
        status = "success"
    else:
        print(f"[WARNING] Failed to read first frame from: {video_path.name}", file=sys.stderr)
        saved_frame_str = ""
        status = "no_frame_read"

    return {
        "video_name": video_path.name,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "first_frame_path": saved_frame_str,
        "status": status,
    }


def run_eda(video_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Process all videos in video_dir and save results in output_dir."""
    if not video_dir.exists():
        print(f"[ERROR] Video directory '{video_dir}' does not exist!", file=sys.stderr)
        return pd.DataFrame()

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        [p for p in video_dir.iterdir() if p.is_file() and p.suffix in VIDEO_EXTENSIONS]
    )

    if not video_files:
        print(f"[INFO] No video files found in '{video_dir}'.")
        # Ensure metadata.csv can still be generated with proper columns
        empty_df = pd.DataFrame(
            columns=[
                "video_name",
                "width",
                "height",
                "resolution",
                "fps",
                "total_frames",
                "duration_sec",
                "first_frame_path",
                "status",
            ]
        )
        metadata_csv_path = output_dir / "metadata.csv"
        empty_df.to_csv(metadata_csv_path, index=False)
        print(f"[INFO] Created empty metadata template at '{metadata_csv_path}'.")
        return empty_df

    print(f"[INFO] Found {len(video_files)} video(s) in '{video_dir}'. Starting EDA extraction...")

    records = []
    for idx, video_path in enumerate(video_files, 1):
        print(f"  [{idx}/{len(video_files)}] Processing: {video_path.name}")
        record = extract_video_metadata_and_first_frame(video_path, frames_dir)
        records.append(record)
        print(
            f"       Resolution: {record['resolution']} | "
            f"FPS: {record['fps']} | "
            f"Frames: {record['total_frames']} | "
            f"Duration: {record['duration_sec']}s"
        )

    df = pd.DataFrame(records)
    metadata_csv_path = output_dir / "metadata.csv"
    df.to_csv(metadata_csv_path, index=False)
    print(f"\n[SUCCESS] Metadata saved to: {metadata_csv_path}")
    print(f"[SUCCESS] Frames saved to: {frames_dir}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract video metadata and first frames for EDA."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("samples"),
        help="Path to folder containing .mp4 video files (default: samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eda_results"),
        help="Path to folder where EDA results will be stored (default: eda_results)",
    )
    args = parser.parse_args()

    run_eda(args.video_dir, args.output_dir)


if __name__ == "__main__":
    main()
