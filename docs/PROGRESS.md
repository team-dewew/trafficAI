# Verification log

Every entry was produced by running the command shown, on the machine noted, against the code in this commit.

Machine: Windows 11, RTX 3050 laptop GPU (8 GB), Python 3.12, torch 2.6.0+cu124.

## Unit, smoke and website tests

```
$ python -m pytest -q
30 passed in 57.98s
```

- `tests/test_units.py`: post-processing, signal read-out and debounce, registration (identity and a known shift), the congestion regression (brand-new tracks), rider ≠ jaywalker, jaywalker detected, Part B pair risk, accident candidates and the verifier (stubbed), weak fire hits.
- `tests/test_smoke.py`: 8 s clip through the unchanged `run_submission.py` (no errors in the log, risk in [0, 1]), `evaluate.py --validate-only`, `import app`.
- `tests/test_website.py`: all 7 website sections render (Streamlit AppTest), and a click-through of the live demo on the
  bundled clip (Run -> events `stop_line` 6.8-16.2 s and `red_light` 19.0-34.0 s, the same events the full pipeline
  finds at 66.5-76.3 s and 78.9-93.9 s of C3896; risk curve; annotated clips).

## Official harness on the four samples

```
$ python run_submission.py --videos samples --out predictions_samples.json --team dewew
[C3896.MP4] 340.3s @ 29.97 fps, budget 1021s
[C3896.MP4] 14 events, 10200 risk samples, 382.7s — OK
[C3897.MP4] 317.8s @ 29.97 fps, budget 953s
[C3897.MP4] 16 events, 9525 risk samples, 352.8s — OK
[C3902.MP4] 317.8s @ 29.97 fps, budget 953s
[C3902.MP4] 24 events, 9525 risk samples, 366.4s — OK
[C3905.MP4] 127.6s @ 29.97 fps, budget 383s
[C3905.MP4] 9 events, 3825 risk samples, 149.0s — OK
wrote predictions_samples.json
$ python evaluate.py --pred predictions_samples.json --validate-only
format: 4 video(s), 63 event(s), 0 error(s), 0 warning(s) -> VALID
```

## Determinism

Perception was cached in one full run (`scripts/replay_rules.py cache`) and the rules replayed from it.
The events are identical to those of a separate harness run on all four videos (every start, end and label).

## Dependency resolution on Linux (pip-like: `--index-strategy unsafe-best-match`)

```
$ uv pip compile requirements.txt --python-version 3.10 --python-platform x86_64-manylinux_2_28 --index-strategy unsafe-best-match
torch==2.6.0  nvidia-cudnn-cu12==9.1.0.70  supervision==0.30.5  numpy==2.2.6   (CUDA build of torch)
$ uv pip compile space/requirements.txt + streamlit==1.64.0 --python-version 3.11 ...
torch==2.6.0+cpu  streamlit==1.64.0  pandas==2.3.3   (website, CPU)
```

An intermediate commit had put the website's CPU torch index into the root `requirements.txt`. With it, pip resolved
`torch==2.14.0+cpu` on the GPU machine, and Part A on CPU would exceed the 3x time budget. The submission file is pinned
again, and the website has its own `space/requirements.txt`.

## Website

- `python scripts/build_space.py` then run the bundle on CPU (`CUDA_VISIBLE_DEVICES=""`, 2 torch threads) with only
  `dist/space/` on disk: every page renders, the demo downloads `yolo11s.pt` on first use, and the bundled 35 s 720p clip
  takes 18-36 s. An 8 s 4K clip runs at about 2x real time.
- Checked in a browser at 1440x900 and at phone width (375 px); the sidebar collapses on phones.

A full install inside a clean Docker container was **not** run here, because the Docker daemon was unavailable on this machine.

## Harness files unchanged

```
$ git diff --exit-code a76904e -- run_submission.py evaluate.py
(no output)
```

## Clean-clone check (what the organizers do)

Fresh `git clone` of the submitted commit into an empty folder, then:

```
$ bash weights/download.sh
yolo11l.pt: OK / yolov8n.pt: OK / accident_model.pt: OK / yolo11s.pt: OK   -> "Weights download completed."
$ python run_submission.py --videos samples --out predictions.json
C3896 14 events 1.13x | C3897 16 events 1.12x | C3902 24 events 1.16x | C3905 9 events 1.18x — all OK, 0 errors
$ python evaluate.py --pred predictions.json --validate-only
format: 4 video(s), 63 event(s), 0 error(s), 0 warning(s) -> VALID
```

- **Reproducibility:** every event (start, end, label) of all four videos is identical to the committed
  `predictions_samples.json`, and the risk curves differ by 0.0.
- **Offline:** the harness run with all HTTP(S) traffic routed to a dead proxy finished with no errors,
  so nothing is downloaded at run time.
- **Fixed during this check:** `weights/download.sh` failed its final checksum step when `SHA256SUMS` was
  checked out with CRLF line endings (Windows `core.autocrlf`). The committed file is LF, so Linux was not
  affected. The script now strips `\r` before `sha256sum -c`, and `.gitattributes` pins LF for it.

## Accident verifier (InternVL2.5-1B)

Weights: the whole `OpenGVLab/InternVL2_5-1B` repository at commit `9d423ea` (21 files, 1.8 GB), fetched by both
`weights/download.sh` and `weights/download.py` and checked against `weights/SHA256SUMS`. The `model.safetensors` hash
equals the LFS hash published by Hugging Face. All weights together: ≈2.1 GB (limit 5 GB).

Zero-shot checks (fp16, RTX 3050), on data from other cameras, used for evaluation only:

```
CCTV stills (sherlockab/accident-detection-from-cctv-footage, test split, 47 accident + 53 normal)
  ROC AUC 0.756; p >= 0.5: precision 0.81, recall 0.53
TAD clips (via wbfwonderful/Vad-R1; 20 accident + 19 normal, random, seed 0), 1.5 s windows every 1 s, full frame
  clip AUC 0.747; p >= 0.7: 11/20 accident clips flagged, 0/19 normal clips (~1,000 normal windows)
Close vehicle pairs in C3896 / C3902 (50 random windows, cropped): max p(yes) 0.36
```

Harness on the four samples with the verifier on:

```
C3896  25 questions  54.9 s  max p(yes) 0.11   14 events
C3897  20 questions  36.6 s  max p(yes) 0.14   16 events
C3902  31 questions  52.2 s  max p(yes) 0.29   24 events
C3905   8 questions  14.8 s  max p(yes) 0.10    9 events
```

Every event and every risk value is identical to `predictions_samples.json`: no accident was reported, and nothing
else changed. The run shared the GPU with another application, so its wall-clock times are not comparable with the
table above. Measured back to back under the same conditions on C3905, Part A took 125–137 s without the verifier and
131 s with it (8 questions, 10.5 s).

- Loading the verifier takes a few seconds. `solution.py` loads every model when the harness imports it, so this
  is not charged to the first video's budget.
- Offline: the harness run on an 8 s clip with every HTTP(S) request routed to a dead proxy and an empty `HF_HOME`
  finished with no errors (the model code is read from `weights/InternVL2_5-1B/`, `local_files_only=True`).
- Linux resolution of `requirements.txt` (Python 3.10/3.11/3.12): torch 2.6.0 (CUDA), transformers 4.46.3,
  timm 1.0.15, einops 0.8.1, tokenizers 0.20.3.
- Unit tests: a touching-then-stopping pair is asked about exactly once and reported at p = 0.95; the same pair is
  not reported at p = 0.1; a pair that drives on is never asked about; a weak fire hit is verified.
