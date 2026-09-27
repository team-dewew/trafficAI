
# Traffic AI — WIUT Hackathon 2026, Computer Vision track

**Team: dewew** · Website: https://trafficai.dewew.dev · Repository: https://github.com/team-dewew/trafficAI

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

After the download, nothing else is fetched: every model is loaded from `weights/` only. A missing YOLO weight
raises an error that names the download command; if `InternVL2_5-1B/` is missing, Part A runs without the verifier.

A `Dockerfile` (CUDA 12.4, Python 3.11) is also provided. Its header shows the build/run commands.

### Which requirements file

| file | for | torch build |
|---|---|---|
| `requirements.txt` | **the submission** (`run_submission.py` on the GPU machine) | CUDA (Linux) |
| `requirements-web.txt` | the public website on a CPU server; installed by `deploy/install.sh` | CPU (`+cpu` wheels) |
| `requirements-dev.txt` | development on a GPU machine: `requirements.txt` + Streamlit + pytest/ruff | CUDA |

The website is self-hosted on our own Ubuntu server (2 vCPU, no GPU) at https://trafficai.dewew.dev, behind nginx with a
Let's Encrypt certificate. `deploy/install.sh` sets up everything (see `deploy/README.md`); locally, after
`pip install -r requirements-dev.txt`, run `streamlit run app.py`.

The live demo runs the same pipeline in a CPU setting: YOLO11-S at 768 px on every 6th frame, no crash/fire model or verifier, and the risk
curve from the same causal tracks (`src/demo.py`). It accepts clips up to 2 minutes / 800 MB (about 45 s of 4K). On 2 vCPUs, a 35 s 720p clip
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
- **Part B** scores time-to-collision only for pairs on a real collision course (at least one of them moving at ≥ 0.5 diag/s,
  about 10 km/h, since junction crashes are often slow): closest approach < 0.3 of their size, held for 2 updates. Duplicate boxes are merged, far-field objects and far-carriageway pairs are skipped (image-space geometry is too compressed there), and same-direction pairs count only as fast rear-end closings. On the samples, which contain no crashes, the score is ≥ 0.5 in under 1 % of frames (9 alarm runs in 18 minutes); on the two crash clips it passes 0.5 at or before the contact. An earlier version was ≥ 0.5 in 36–60 % of frames because of duplicate boxes and perspective convergence.
- **Accidents are proposed by the tracks and confirmed by a vision-language model.** A candidate is the first moment two road
  users' boxes touch (each grown by 5 % of its diagonal, because a head-on hit leaves two boxes side by side with almost no
  overlap; ground points within 0.9 of the larger diagonal) while one of them drives at ≥ 0.5 diag/s and the gap shrank by
  ≥ 0.3 diag over the last second. 2.5 s later, both must have slowed below 0.5 diag/s and still be within one diagonal of
  each other: vehicles that drive on or apart are dropped. Crash-model hits at conf ≥ 0.3 are also candidates.
  InternVL2.5-1B then sees 4–6 frames cropped to a square of 1.6× the pair's box and answers "has an accident or collision
  happened?"; p(yes) is read from the Yes/No logits of one forward pass (no text generation, deterministic, ~0.7 s).
  A first window (0.5 s before to 2.5 s after the contact) screens: below 0.5 the candidate is dropped. Otherwise four
  windows around the contact are asked and the accident is reported if their mean is ≥ 0.7. One answer of this 1B model
  moves by up to ~0.2 with the exact frames, so a single window was not reliable: on the samples a queue at C3902 reached
  0.79 in one window, while its four-window mean is 0.62. The event runs from the contact until every involved road user
  has stopped or left.
- **Smoke after a crash.** Smoke often starts seconds after the impact, so a verified crash site is re-checked every 1.5 s
  for 20 s (longer while smoke is seen) with "is there fire or smoke?". `fire_smoke` starts half a step before the first
  "yes" and ends after two "no"s in a row or at the end of the video.
- The verifier is capped at 60 questions and 0.25× the video duration per video. It is loaded once when the harness imports
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
| C3896.MP4 | 340 s | 14 | failure_to_yield 4, jaywalking 7, red_light 1, stop_line 2 | 362 s | 260 s | 1.82x | 0.00% |
| C3897.MP4 | 318 s | 16 | failure_to_yield 5, jaywalking 8, stop_line 2, stopped_vehicle 1 | 300 s | 238 s | 1.69x | 0.72% |
| C3902.MP4 | 318 s | 24 | failure_to_yield 8, jaywalking 15, stop_line 1 | 361 s | 241 s | 1.89x | 0.38% |
| C3905.MP4 | 128 s | 9 | congestion 1, failure_to_yield 4, jaywalking 2, stop_line 1, stopped_vehicle 1 | 155 s | 99 s | 1.98x | 0.00% |

Run on a laptop RTX 3050 (8 GB) with a 4K H.264 input; the budget is 3× the video duration. Part A includes the
verifier (14–35 questions, 10–27 s per video). Wall-clock time on this laptop depends on what else runs: 4K decoding is
CPU-bound, and during this run the GPU was only ~40 % busy while other applications used the CPU. The same code took
1.1–1.3× on the idle machine (`docs/PROGRESS.md`); the organizers' machine (T4, 8 cores) is dedicated.

**Crash clips.** The samples contain no crash, so we also ran two 10 s clips of this camera with a crash (720p; they look
generated from a sample frame, and are not in the repository):

| clip | contact / smoke (by eye) | our events | Part B alarm (risk ≥ 0.5) | total |
|---|---|---|---|---|
| crash_video | contact ~3.6 s, smoke from ~6.8 s | accident 3.6–6.1 s, fire_smoke 7.4–10.0 s, red_light 1.9–9.9 s, failure_to_yield 1.1–2.1 s | from 3.4 s | 19.5 s of 30 s |
| crash_video_2 | contact ~3.1 s, smoke from ~4.5 s | accident 3.1–5.6 s, fire_smoke 3.9–10.0 s, red_light 1.1–9.9 s | from 3.1 s | 19.8 s of 30 s |

---

## Limitations (stated plainly)

- **No labelled dev set.** F1 has not been measured. `src/devset/csv_to_gt.py` converts per-video CSV labels (`start,end,label,note`) into `evaluate.py` ground truth, and `scripts/eval_dev.sh` runs the whole evaluation. Labelling the four samples is the next step.
- `accident`, `near_miss`, `wrong_way`, `fire_smoke` and `road_obstacle` produced no events on the samples. Their rules are deliberately strict, so recall on the hidden set is unknown.
- The accident verifier was tuned on very little crash data: two 10 s crash clips of this camera (they look generated
  from a sample frame), whose contacts it reports at the right time, and 108 candidate windows of the four samples, none
  reported. The margin is modest (crash clips 0.77 and 0.80, highest sample 0.62, threshold 0.7). On other cameras,
  scanning whole TAD clips it flagged 11 of 20 accident clips and 0 of 19 normal ones. Recall on the hidden set also
  depends on the track-based candidates.
- The website demo (CPU) does not report `accident` or `fire_smoke`: the verifier needs a GPU.
- `jaywalking` ignores people within ~1 m of a zebra, island or kerb, which trades recall for precision.
- The three turn/marking classes are off (see Class policy).
- Part B is a heuristic (time-to-collision). It was not calibrated on real crashes because the samples contain none.

---

## Determinism

`src/config.py:seed_everything(42)` seeds `random`, NumPy and PyTorch, and sets cuDNN
to deterministic mode. The pipeline has no other randomness: ByteTrack and the
rules are deterministic given the detections. The verifier reads p(yes) from one FP16
forward pass (no sampling), so it is deterministic as well. GPU kernels can still
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
  devset/                   labelling helpers (CSV -> ground truth, review clips, report)
scripts/
  replay_rules.py           cache perception once, replay the rules in seconds (tuning)
  signal_timeline.py        signal phases per video
  vlm_probe.py              speed and p(yes) of the verifier on chosen windows
  deep_eda.py, eda_extractor.py   EDA artefacts in eda_results/ (website EDA page)
  make_examples.py          example frames per class in assets/examples/ (website Results page)
  make_clip.py, smoke.sh, eval_dev.sh   test clip, test run, dev-set evaluation
tests/                      unit tests, harness smoke test, website pages + live-demo click-through
docs/                       scene.md, class_policy.md, PROGRESS.md (verification log)
examples/                   starter-kit ground truth / predictions examples for evaluate.py
eda_results/                EDA of the sample videos (metadata, counts, heatmaps, trajectories)
assets/scene_ref.jpg        reference frame for registration
assets/examples/            one or two frames per detected class (website Results)
assets/team/team.json       team page content (roles, contributions, links, previous projects)
samples/previews/           annotated sample videos; samples/demo/ a 35 s 720p clip for the demo
deploy/                     self-hosted website: install/update/check/uninstall scripts (nginx, Let's Encrypt, systemd)
weights/                    download.sh / download.py / SHA256SUMS
app.py                      team website (Streamlit)
requirements*.txt           submission / website / development (see "Which requirements file")
Dockerfile                  optional container for the submission
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
- **Soliyev Siroj** — sample-video EDA, scene annotation notes, evaluation runs ([GitHub](https://github.com/team-dewew))


