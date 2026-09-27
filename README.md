# Traffic AI — WIUT Hackathon 2026, Computer Vision track

**Team: dewew** · Website: https://trafficai.dewew.dev · Repository: https://github.com/DeWeWO/wiut

The system watches a road-junction CCTV camera and does two things:

- **Part A:** reports traffic events as `[start_sec, end_sec, label]` segments.
- **Part B:** outputs a causal per-frame risk that an accident starts within 5 s.

---

## Install and run

Python 3.10–3.12 with an NVIDIA GPU (a CPU works too, but is slow).

```bash
pip install -r requirements.txt
bash weights/download.sh          # once, with internet (Windows: python weights/download.py)
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

`weights/download.sh` fetches the weights (≈2.1 GB total, limit 5 GB) and verifies them against `weights/SHA256SUMS`:

| file | what | source | licence |
|---|---|---|---|
| `yolo11l.pt` | road-user detector (Part A) | Ultralytics release v8.3.0 (COCO) | AGPL-3.0 |
| `yolov8n.pt` | light detector (Part B) | Ultralytics release v8.3.0 (COCO) | AGPL-3.0 |
| `yolo11s.pt` | website CPU demo only (not used by the submission) | Ultralytics release v8.3.0 (COCO) | AGPL-3.0 |
| `accident_model.pt` | YOLOv8x crash-severity / fire / smoke detector (classes: detected-injury, fire, high / medium / low severity, smoke) | [Enos-123/accident-evaluator-yolov8x](https://huggingface.co/Enos-123/accident-evaluator-yolov8x/tree/main/weights): the file `weights/epoch90.pt`, downloaded as-is and renamed to `accident_model.pt` (checksum in `weights/SHA256SUMS`) | MIT (as declared on the model card) |
| `InternVL2_5-1B/` | vision-language model (1B) that verifies accident and fire/smoke candidates with a yes/no question | [OpenGVLab/InternVL2_5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B): the whole repository (weights, model code, tokenizer, configs) pinned to commit `9d423ea`; every file is checksummed | MIT |

### Datasets

We trained nothing ourselves. The datasets below are the ones behind the pretrained weights we use:

| dataset | used through | licence |
|---|---|---|
| COCO 2017 | `yolo11l.pt`, `yolov8n.pt`, `yolo11s.pt` (Ultralytics pretrained) | CC BY 4.0 (annotations) |
| Roboflow "Accident Evaluator" | `accident_model.pt` (named as its training set on the model card) | not stated on the model card; the card links no dataset page |
| InternVL 2.5 pre-training / fine-tuning data | `InternVL2_5-1B/` | listed on the model card |

Two public datasets were used **only to evaluate** the verifier (nothing was trained or tuned on them except the single
acceptance threshold, and none of their files are in this repository):

| dataset | what we used | licence |
|---|---|---|
| TAD (Traffic Anomaly Dataset), via [wbfwonderful/Vad-R1](https://huggingface.co/datasets/wbfwonderful/Vad-R1) (`Vad-Reasoning-RL/TAD/`) | 20 accident + 19 normal CCTV clips, drawn at random (seed 0) | research use, per the TAD authors; the Vad-R1 card states no licence |
| [sherlockab/accident-detection-from-cctv-footage](https://huggingface.co/datasets/sherlockab/accident-detection-from-cctv-footage) | test split, 47 accident + 53 normal CCTV stills | MIT |

The scene layout was drawn by hand on a frame of the provided sample videos.

After the download, nothing else is fetched: the code loads weights only from
`weights/` and raises an error instead of downloading anything.

A `Dockerfile` (CUDA 12.4, Python 3.11) is also provided. Its header shows the build/run commands.

`requirements.txt` is the submission environment (pinned, CUDA build of torch on Linux). The website has
separate dependencies:

- **Locally:** `pip install -r requirements-web.txt && streamlit run app.py`.
- **Own server (the public website):** `deploy/install.sh` installs it on an Ubuntu server without a GPU, behind nginx with a
  Let's Encrypt certificate (see `deploy/README.md`).
- **Hugging Face Space (alternative):** `python scripts/build_space.py` writes `dist/space/`. That folder holds the site files, with
  `space/README.md` (the Space header) and the CPU `space/requirements.txt` as its root files. The Space is deployed from it, so
  the Space's settings never change the submission's requirements.

The live demo runs the same pipeline in a CPU setting: YOLO11-S at 768 px on every 6th frame, no crash/fire model or verifier, and the risk
curve from the same causal tracks (`src/demo.py`). It accepts clips up to 2 minutes / 300 MB. On 2 vCPUs, a 35 s 720p clip
takes about 35 s and 4K takes about 2× the clip length.

---

## Approach

```
video ─► scene registration (SIFT+RANSAC similarity vs. assets/scene_ref.jpg)
      │        └► scene layout mapped onto this video: zones, stop lines, signal lamps
      ├► signal state from the lamps (debounced)
      ├► YOLO11-L @960, FP16, every 3rd frame ─► car/truck duplicate suppression ─► ByteTrack
      │        └► track state: ground point, speed in body-diagonals / s
      ├► YOLOv8x anomaly model at 1 Hz
      ├► rule engine (src/rules.py) ─► merge / clip / drop blips ─► events
      │        └► accident / fire candidates ─► InternVL2.5-1B yes/no on the last 4 s of frames (src/vlm.py)
Part B: YOLOv8n every 3rd frame ─► ByteTrack ─► time-to-collision on collision courses ─► EMA ─► risk
```

**Learned** (off-the-shelf open weights; we trained nothing): YOLO11-L and YOLOv8n (COCO detectors), the YOLOv8x crash/fire model, and the InternVL2.5-1B vision-language model.
**Rule-based**: scene registration, signal read-out, every event rule, post-processing and the Part B risk score.

Key design points:

- **The camera pose drifts between recordings.** Relative to the reference frame (C3905), C3902 is shifted by (−91, +28) px, and C3896/C3897 are rotated by ~1° and scaled by 0.986. Each video is registered once at start-up, and the transform is applied to all zones. See `docs/scene.md`.
- **The signal is read from its lamps.** A lit lamp is ~5 px tall at 4K, so we measure colour in small windows at the calibrated lamp centres. On all four samples this gives a clean cycle of ~37 s red, ~35 s green and 3–6 s amber/transition (`scripts/signal_timeline.py`).
- **Speeds are scale-free**, measured in body-diagonals per second of the track's ground point. The same threshold then works near and far from the camera, and at any resolution.
- **Part B** scores time-to-collision only for pairs on a real collision course: closest approach < 0.3 of their size, held for 2 updates. Duplicate boxes are merged, far-field objects and far-carriageway pairs are skipped (image-space geometry is too compressed there), and same-direction pairs count only as fast rear-end closings. On the samples, which contain no crashes, the score is ≥ 0.5 in under 0.5 % of frames. An earlier version was ≥ 0.5 in 36–60 % of frames because of duplicate boxes and perspective convergence.
- **Accidents are proposed by the tracks and confirmed by a vision-language model.** A candidate is the first moment two road
  users' boxes touch (ground-point gap < 0.5 of the larger body diagonal) while one of them drives at ≥ 0.5 diag/s and the gap
  shrank by ≥ 0.3 diag over the last second. Two seconds later, both must have slowed below 0.5 diag/s and still be within one
  diagonal of each other: vehicles that drive on or apart are dropped. This cuts ~170 raw candidates per 5-minute sample to ~20.
  Crash-model hits at conf ≥ 0.3 are also candidates. Each candidate is shown to InternVL2.5-1B as 4 frames (from 1 s before
  contact to 2 s after), cropped to a square around the pair, with the question "has an accident or collision happened?".
  p(yes) is read from the Yes/No logits of a single forward pass (no text generation, deterministic, ~1 s per question).
  The event is reported at p ≥ 0.7, from the contact to the moment every involved road user has stopped or left.
  The verifier is capped at 40 questions and 0.25× the video duration per video. It is loaded once when the harness imports
  `solution.py`, and if its weights are missing Part A runs without it.
- **Class policy.** `illegal_turn`, `illegal_u_turn` and `solid_line_crossing` are switched off: we do not have the permitted-manoeuvre map or the solid-line geometry, and a class predicted but absent from the test set costs macro-F1. The rule for each class and the reasoning are in `docs/class_policy.md`. All thresholds are in `src/config.py`.

### How the rules were tuned

We have no hand-labelled dev set yet. Instead, perception output was cached once
(`scripts/replay_rules.py cache ...`) and the rules were replayed in seconds. We then
rendered every candidate event of every class as a frame montage with the
involved tracks highlighted, and inspected it. This exposed the false-positive
patterns that each rule now excludes:

- motorcycle riders taken as pedestrians;
- people waiting at the kerb inside a zebra polygon;
- red-light queues taken as congestion or stopped vehicles;
- parked cars;
- vehicles waiting to turn inside the junction.

---

## Results on the sample videos

`predictions_samples.json` is our output on the four samples, produced by the command above.

| video | duration | events | by class | Part A | Part B | total / duration | risk >= 0.5 |
|---|---|---|---|---|---|---|---|
| C3896.MP4 | 340 s | 14 | failure_to_yield 4, jaywalking 7, red_light 1, stop_line 2 | 166 s | 216 s | 1.12x | 0.00% |
| C3897.MP4 | 318 s | 16 | failure_to_yield 5, jaywalking 8, stop_line 2, stopped_vehicle 1 | 150 s | 203 s | 1.11x | 0.50% |
| C3902.MP4 | 318 s | 24 | failure_to_yield 8, jaywalking 15, stop_line 1 | 159 s | 207 s | 1.15x | 0.35% |
| C3905.MP4 | 128 s | 9 | congestion 1, failure_to_yield 4, jaywalking 2, stop_line 1, stopped_vehicle 1 | 67 s | 82 s | 1.17x | 0.00% |

Run on a laptop RTX 3050 (8 GB) with a 4K H.264 input; the budget is 3× the video duration.

---

## Limitations (stated plainly)

- **No labelled dev set.** F1 has not been measured. `src/devset/csv_to_gt.py` converts per-video CSV labels (`start,end,label,note`) into `evaluate.py` ground truth, and `scripts/eval_dev.sh` runs the whole evaluation. Labelling the four samples is the next step.
- `accident`, `near_miss`, `wrong_way`, `fire_smoke` and `road_obstacle` produced no events on the samples. Their rules are deliberately strict, so recall on the hidden set is unknown.
- The accident verifier was checked on other cameras only (the samples contain no crash). Scanning whole TAD clips with
  1.5 s windows at p ≥ 0.7, it flagged 11 of 20 accident clips and 0 of 19 normal clips (≈1,000 normal windows); on
  single CCTV stills its ROC AUC is 0.76. On the four samples it answered 84 questions, the highest p(yes) was 0.29, and
  no accident was reported. Its recall on the hidden set also depends on the track-based candidates.
- `jaywalking` ignores people within ~1 m of a zebra, island or kerb, which trades recall for precision.
- The three turn/marking classes are off (see Class policy).
- Part B is a heuristic (time-to-collision). It was not calibrated on real crashes because the samples contain none.

---

## Determinism

`src/config.py:seed_everything(42)` seeds `random`, NumPy and PyTorch, and sets cuDNN
to deterministic mode. The pipeline has no other randomness: ByteTrack and the
rules are deterministic given the detections. GPU convolution kernels can still
differ at the floating-point-noise level across GPU models.

A time guard exists: if Part A's frame loop runs slower than 1.6× real time, the detector
stride is doubled for the rest of the video. This never triggered on the test
machine, but on a much slower GPU it could make runs differ.

---

## Repository layout

```
solution.py                 interface for the harness (thin wrapper over src/)
run_submission.py           organizers' harness (unchanged)
evaluate.py                 organizers' metric (unchanged)
src/
  registration.py           per-video scene registration
  scene.py                  hand-calibrated scene layout (reference 4K frame)
  traffic_light.py          lamp read-out + debounced signal state
  perception.py             detector, duplicate suppression, tracker, anomaly model
  vlm.py                    InternVL2.5-1B yes/no verifier for accident / fire candidates
  tracks.py                 per-track kinematics
  rules.py                  one method per event class
  postprocess.py            merge / clip / drop blips
  events.py                 Part A pipeline (+ replay from cached perception)
  risk.py                   Part B estimator
  config.py                 thresholds, enabled classes, seed
  annotate.py               annotated-video renderer (website previews)
  demo.py                   website live demo (CPU setting, clips drawn from stored tracks)
  deep_eda.py, eda_extractor.py   EDA artefacts for the website
  devset/                   labelling helpers (CSV -> ground truth, review clips, report)
scripts/                    replay_rules.py, signal_timeline.py, make_examples.py, build_space.py, eval_dev.sh, smoke.sh, make_clip.py, vlm_probe.py
tests/                      unit tests + end-to-end smoke test through run_submission.py
docs/                       scene.md, class_policy.md
assets/scene_ref.jpg        reference frame for registration
assets/examples/            one or two frames per detected class (website Results)
assets/team/team.json       team page content (roles, contributions, links, previous projects)
samples/previews/           annotated sample videos; samples/demo/ a 35 s 720p clip for the demo
space/                      website CPU requirements + Hugging Face Space header (see build_space.py)
deploy/                     self-hosted website: install/update/check/uninstall scripts (nginx, Let's Encrypt, systemd)
weights/                    download.sh / download.py / SHA256SUMS
app.py                      team website (Streamlit)
```

Development: `pip install -r requirements-dev.txt`, then `bash scripts/smoke.sh`, which runs
`pytest`: unit tests, an 8 s clip through the official harness, every website page, and a click-through of the live demo on the bundled clip.

---

## Open-source code used

- Ultralytics YOLO (AGPL-3.0) for detection.
- Hugging Face Transformers (Apache-2.0), timm (Apache-2.0) and einops (MIT) to run InternVL2.5-1B.
- supervision (MIT) for ByteTrack.
- OpenCV (Apache-2.0) for video I/O, SIFT and geometry.

No external footage was used for tuning the rules (see Datasets for the verifier's evaluation data). The scene layout was drawn by hand on a sample frame. The organizers
confirmed that `camera.md` is not provided.

---

## Team

- **Ollabergan** — computer vision and system architecture, website ([GitHub](https://github.com/DeWeWO) • [LinkedIn](https://www.linkedin.com/in/dewew/))
- **Seymonbek Ikramov** — anomaly model integration, Part B risk estimator ([GitHub](https://github.com/Seymonbek) • [LinkedIn](https://www.linkedin.com/in/seymonbek-ikramov-0022b2386/))
- **Soliyev Siroj** — sample-video EDA, scene annotation notes, evaluation runs
