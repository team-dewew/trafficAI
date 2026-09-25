import cv2
import sys
from pathlib import Path

def make_clip(in_path, out_path, duration_sec):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"Failed to open {in_path}")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    max_frames = int(fps * duration_sec)
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1
        
    cap.release()
    writer.release()
    print(f"Saved {count} frames to {out_path}")

if __name__ == "__main__":
    make_clip("samples/C3905.MP4", "tests/data/clip8s.mp4", 8.0)
