# Traffic AI — WIUT Hackathon 2026 (Computer Vision Track)

An intelligent, high-throughput traffic surveillance system combining State-of-the-Art Object Detection, dynamic AI auto-alignment, calibrated 21-zone geometric spatial reasoning, and deep anomaly detection for real-time event detection (Part A) and causal accident anticipation (Part B).

---

## Installation & Running

### Environment Setup
Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Model Weights Placement
Ensure model weights are located inside the `weights/` directory:
- `weights/yolo11l.pt`: Primary road user detector (YOLO11 Large)
- `weights/accident_model.pt`: Secondary anomaly detector (YOLOv8x Crash/Fire model)
- `weights/yolov8n.pt`: Lightweight Part B causal risk estimator (YOLOv8 Nano)

*(Optional: Use `bash weights/download.sh` to fetch remote weights if needed).*

### Running Submission
To run inference over a folder of test videos using the official harness:

```bash
python run_submission.py --videos samples --out predictions_samples.json
```

### Evaluating Format & Metric
Run the official evaluation script to perform format validation or benchmark against ground truth:

```bash
# Format check only:
python evaluate.py --pred predictions_samples.json --validate-only

# Full evaluation against ground truth:
python evaluate.py --pred predictions_samples.json --gt ground_truth.json --per-video
```

### Launching Team Web Application
To start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

---

## Approach (Rule-based vs Learned)

Our architecture employs a **Hybrid AI Pipeline** balancing deep perceptual understanding with rigid geometric spatial logic to remain within the hackathon's `3.0x` time budget while maximizing accuracy across all 14 traffic event classes:

### 1. Rule-Based Events (21-Zone Geometric Coordinate Map)
The following 10 classes are governed by deterministic spatial-temporal rules evaluated over a calibrated 21-zone polygon layout:
- `jaywalking`: Pedestrian detections entering the roadway corridor outside designated crosswalk polygons.
- `red_light`: Vehicles crossing the primary stop line (`stop_line_red`) while the calibrated signal bbox detects an active red light.
- `failure_to_yield`: Vehicles crossing pedestrian yield lines while pedestrian tracks occupy active crosswalk zones.
- `wrong_way`: Vehicles whose displacement trajectory opposes the designated vector flow of `lane_ltr` or `lane_rtl`.
- `solid_line_crossing`: Maneuvers crossing solid division markings between adjacent travel lanes.
- `stopped_vehicle`: Stationary vehicles on the carriageway for $\ge 10$ seconds outside of signalized queues.
- `illegal_turn`: Turns executed from unauthorized lanes or violating intersection turn boundaries.
- `illegal_u_turn`: Sharp trajectory reversals inside prohibited intersection zones.
- `congestion`: Widespread standstill or crawling vehicular traffic across all lanes simultaneously.
- `road_obstacle`: Stationary debris, dropped cargo, or domestic animals detected on the carriageway for $\ge 1.0$ s.

### 2. Learned Events (Secondary YOLOv8x Anomaly Model)
Complex physical collisions and fire hazards cannot be captured by rigid spatial rules alone. These classes are detected using a specialized secondary deep learning model:
- `accident`: Dynamic detection of vehicle-to-vehicle collisions, rollovers, and structural crashes via `weights/accident_model.pt`.
- `fire_smoke`: Detection of active vehicle combustion, open flames, and dense smoke plumes via `weights/accident_model.pt`.

### 3. Causal Accident Anticipation (Part B - `RiskEstimator`)
Estimates $P(\text{accident starts within 5 s})$ causally frame-by-frame:
- Employs a lightweight YOLOv8n detector with ByteTrack association evaluated every 3 frames.
- Analyzes Time-to-Collision (TTC) proxies from bounding box IoU overlap ($> 0.6$) and centroid proximity ($< 40\text{ px}$ in 640p scale).
- Evaluates sudden pedestrian intrusion into core traffic corridors.
- Smooths output probabilities with an exponential moving average ($\alpha = 0.3$), guaranteeing zero future-frame data leakage.

---

## Datasets & Licenses

- **COCO Dataset (GPL-3.0)**: Used for pre-trained weights in standard YOLO11 road user perception (`pedestrian`, `car`, `bus`, `truck`, `motorcycle`, `bicycle`, and obstacle classes).
- **Car Crash Dataset (CCD) / DoTA (MIT / CC-BY)**: Used for training and fine-tuning the specialized secondary anomaly detection model (`accident_model.pt`).

---

## Determinism

To guarantee 100% reproducible benchmark scores and prevent floating-point or stochastic tracking discrepancies across different runs and hardware, all pseudo-random number generators are strictly fixed to seed `42` at module initialization:

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

---

## Team

- **[Member 1 - Role]**: Lead Computer Vision Engineer (Geometric Reasoning & Tracking Architecture)
- **[Member 2 - Role]**: Deep Learning Engineer (Model Training & Anomaly Detection)
- **[Member 3 - Role]**: DevOps & Full-Stack AI Engineer (Inference Optimization & Streamlit Dashboard)
