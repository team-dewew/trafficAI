# Traffic AI — WIUT Hackathon 2026 (Computer Vision Track)

**Team: dewew**

An intelligent, high-throughput traffic surveillance system combining State-of-the-Art Object Detection, dynamic AI auto-alignment, calibrated 21-zone geometric spatial reasoning, and deep anomaly detection for real-time event detection (Part A) and causal accident anticipation (Part B).

---

## Installation & Running

### Environment Setup
Clone the repository and install the required dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

### Model Weights
All three model weights are fetched with one command (run once, with internet, before evaluation):

```bash
bash weights/download.sh
```

This downloads:
- `weights/yolo11l.pt` (~50 MB): primary road user detector (YOLO11 Large, Ultralytics release)
- `weights/yolov8n.pt` (~6 MB): lightweight Part B causal risk estimator (Ultralytics release)
- `weights/accident_model.pt` (~131 MB): secondary YOLOv8x anomaly detector (Crash/Fire), fetched from Hugging Face [`Enos-123/accident-evaluator-yolov8x`](https://huggingface.co/Enos-123/accident-evaluator-yolov8x) (`weights/epoch90.pt`, renamed)

Total ≈ 190 MB — far under the 5 GB limit.

### Running Submission
To run inference over a folder of test videos using the official harness:

```bash
python run_submission.py --videos samples --out predictions_samples.json --team dewew
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
The following 11 classes are governed by deterministic spatial-temporal rules evaluated over a calibrated 21-zone polygon layout (stop lines, lane corridors, crosswalks, concrete islands, sidewalks, intersection core), combined with ByteTrack trajectory analysis:

- `jaywalking`: Pedestrian detections entering the roadway corridor outside designated crosswalk polygons.
- `red_light`: Vehicles crossing the primary stop line (`stop_line_red`) while the calibrated signal bbox detects an active red light.
- `stop_line`: Vehicles stopping past the stop line on red without entering the intersection.
- `failure_to_yield`: Vehicles crossing pedestrian yield lines while pedestrian tracks occupy active crosswalk zones.
- `wrong_way`: Vehicles whose displacement trajectory opposes the designated vector flow of `lane_ltr` or `lane_rtl`.
- `solid_line_crossing`: Maneuvers crossing solid division markings between adjacent travel lanes.
- `stopped_vehicle`: Stationary vehicles on the carriageway for ≥ 10 seconds outside of signalized queues.
- `illegal_turn`: Turns executed from unauthorized lanes or violating intersection turn boundaries.
- `illegal_u_turn`: Sharp trajectory reversals inside prohibited intersection zones.
- `congestion`: Widespread standstill or crawling vehicular traffic across all lanes simultaneously.
- `road_obstacle`: Stationary debris, dropped cargo, or domestic animals detected on the carriageway for ≥ 1.0 s.
- `near_miss`: Scale-aware pairwise proximity analysis — two vehicles closing to < 0.85× their combined bounding-box diagonal while at least one exhibits hard braking (> 55% speed drop within ~0.5 s), with no bounding-box contact (IoU < 0.03, otherwise it is `accident` territory).

### 2. Learned Events (Secondary YOLOv8x Anomaly Model)
Complex physical collisions and fire hazards cannot be captured by rigid spatial rules alone. These classes are detected using a specialized secondary deep learning model (`weights/accident_model.pt`, evaluated every 5th frame):

- `accident`: Dynamic detection of vehicle-to-vehicle collisions, rollovers, and structural crashes.
- `fire_smoke`: Detection of active vehicle combustion, open flames, and dense smoke plumes.

### 3. Causal Accident Anticipation (Part B — `RiskEstimator`)
Estimates P(accident starts within 5 s) causally frame-by-frame:
- Employs a lightweight YOLOv8n detector with ByteTrack association evaluated every 3rd frame.
- **Scale-aware Time-to-Collision proxy**: vehicle pairs closing to < 0.75–0.95× their combined diagonal with measurable approach speed (> 1.5–2.5 px/frame at 640p scale) and frame-over-frame distance shrinkage.
- **Pedestrian–vehicle conflict**: pedestrians on the carriageway (zone-accurate, reusing the Part A polygons) only raise risk when a genuinely moving vehicle bears down on them — avoiding a constant risk floor in this busy intersection.
- Smooths output probabilities with an exponential moving average (α = 0.45), guaranteeing zero future-frame data leakage. The estimator never opens the video file.

---

## Models & Data Sources

| Component | Source | License |
|---|---|---|
| `yolo11l.pt` (primary detector) | Ultralytics YOLO11 official release, pre-trained on COCO | AGPL-3.0 |
| `yolov8n.pt` (risk estimator) | Ultralytics YOLOv8 official release, pre-trained on COCO | AGPL-3.0 |
| `accident_model.pt` (anomaly detector) | [`Enos-123/accident-evaluator-yolov8x`](https://huggingface.co/Enos-123/accident-evaluator-yolov8x) on Hugging Face (YOLOv8x fine-tuned for crash/fire detection), open weights | AGPL-3.0 (Ultralytics) |
| Tracker | ByteTrack via `supervision` | MIT |

No paid or closed API is called at any stage of inference; everything runs offline from the shipped weights. No external footage was used — the 21-zone scene geometry was calibrated by hand from the official sample videos' first frames (the organizers confirmed no `camera.md` is provided for this task).

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

Models are loaded once per process and memoized (`_load_yolo`), so repeated runs over a folder of videos are both faster and identical.

---

## Repository Layout

```
├── solution.py              # the official interface: CLASSES, detect_events, RiskEstimator
├── run_submission.py        # organizers' harness (unchanged)
├── evaluate.py              # organizers' metric + format check (unchanged)
├── requirements.txt         # pinned dependencies
├── weights/download.sh      # one-command weight fetch (≤ 5 GB)
├── src/
│   ├── annotate.py          # annotated-video renderer (previews & demo clips)
│   └── eda_extractor.py     # sample-video metadata + EDA artifact extraction
├── notebooks/               # EDA experiments
├── samples/previews/        # fully annotated sample videos (rendered by src/annotate.py)
├── eda_results/             # EDA artifacts (metadata, first frames, charts)
├── predictions_samples.json # our output on the sample videos (team: dewew)
└── app.py                   # team website (Streamlit)
```

---

## Engineering Squad (Team dewew)

- **Ollabergan** — Lead Computer Vision & Full-Stack AI Architect ([GitHub](https://github.com/DeWeWO) • [LinkedIn](https://www.linkedin.com/in/dewew/))
- **Seymonbek Ikramov** — Deep Learning & Causal Risk Specialist ([GitHub](https://github.com/Seymonbek) • [LinkedIn](https://www.linkedin.com/in/seymonbek-ikramov-0022b2386/))
- **Soliyev Siroj** — Data Ops & Evaluation Engineer (sample EDA, dev-set annotation, benchmark runs)
