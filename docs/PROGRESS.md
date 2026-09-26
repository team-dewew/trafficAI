# Verification log

Every entry was produced by running the command shown, on the machine noted, against the code in this commit.

Machine: Windows 11, RTX 3050 laptop GPU (8 GB), Python 3.12, torch 2.6.0+cu124.

## Unit, smoke and website tests

```
$ python -m pytest -q
25 passed in 33.78s
```

- `tests/test_units.py`: post-processing, signal read-out and debounce, registration (identity and a known shift), the congestion regression (brand-new tracks), rider ≠ jaywalker, jaywalker detected, Part B pair risk.
- `tests/test_smoke.py`: 8 s clip through the unchanged `run_submission.py` (no errors in the log, risk in [0, 1]), `evaluate.py --validate-only`, `import app`.
- `tests/test_website.py`: all 7 website sections render (Streamlit AppTest).

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

## Dependency resolution on Linux

```
$ uv pip compile requirements.txt --python-version 3.10 --python-platform x86_64-manylinux_2_28
numpy==2.2.6  opencv-python-headless==4.11.0.86  torch==2.6.0  ultralytics==8.4.161  lap==0.5.13   (resolves)
$ ... --python-version 3.12 ...   (same versions, resolves)
```

A full install inside a clean Docker container was **not** run here, because the Docker daemon was unavailable on this machine.

## Harness files unchanged

```
$ git diff --exit-code a76904e -- run_submission.py evaluate.py
(no output)
```
