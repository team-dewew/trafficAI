import hashlib
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent

FILES = {
    "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "accident_model.pt": "https://huggingface.co/Enos-123/accident-evaluator-yolov8x/resolve/main/weights/epoch90.pt"
}

def verify_checksums():
    sums_file = WEIGHTS_DIR / "SHA256SUMS"
    if not sums_file.exists():
        print("SHA256SUMS file not found, skipping verification.")
        return True

    expected_sums = {}
    with open(sums_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                expected_sums[parts[1]] = parts[0]

    success = True
    for filename, expected in expected_sums.items():
        filepath = WEIGHTS_DIR / filename
        if not filepath.exists():
            print(f"Missing file: {filename}")
            success = False
            continue

        print(f"Verifying {filename}...")
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        actual = sha256.hexdigest()
        if actual != expected:
            print(f"Checksum mismatch for {filename}: expected {expected}, got {actual}")
            success = False
        else:
            print(f"{filename}: OK")

    return success

def download_files():
    print(f"Downloading weights to {WEIGHTS_DIR}...")
    for filename, url in FILES.items():
        filepath = WEIGHTS_DIR / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            # For simplicity, no progress bar, but retry mechanism could be added.
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Downloaded {filename}.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if filepath.exists():
                    filepath.unlink()
                raise
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    download_files()
    if verify_checksums():
        print("Weights download and verification completed successfully.")
    else:
        print("Verification failed!")
        exit(1)
