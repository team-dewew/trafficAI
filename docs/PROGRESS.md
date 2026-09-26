# Verification log

Every entry was produced by running the command shown, on the machine noted, against the code in this commit.

Machine: Windows 11, RTX 3050 laptop GPU (8 GB), Python 3.12, torch 2.6.0+cu124.

## Unit, smoke and website tests

```
$ python -m pytest -q
26 passed in 41.95s
```

- `tests/test_units.py`: post-processing, signal read-out and debounce, registration (identity and a known shift), the congestion regression (brand-new tracks), rider ≠ jaywalker, jaywalker detected, Part B pair risk.
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
