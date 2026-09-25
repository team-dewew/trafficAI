import cv2
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.scene import SCENE_CONFIG

def main():
    video_path = repo_root / "tests/data/clip8s.mp4"
    if not video_path.exists():
        print(f"Video {video_path} not found.")
        return
        
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read frame.")
        return
        
    main_signal = SCENE_CONFIG["main_signal"]
    ped_signal = SCENE_CONFIG["ped_signal"]
    
    # Draw main signal
    cv2.rectangle(frame, (main_signal[0], main_signal[1]), (main_signal[2], main_signal[3]), (0, 0, 255), 2)
    cv2.putText(frame, "Main TL", (main_signal[0], main_signal[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Draw ped signal
    cv2.rectangle(frame, (ped_signal[0], ped_signal[1]), (ped_signal[2], ped_signal[3]), (0, 255, 0), 2)
    cv2.putText(frame, "Ped TL", (ped_signal[0], ped_signal[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out_path = repo_root / "scratch/tl_check.jpg"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
