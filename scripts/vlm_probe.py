"""Probe the InternVL2.5-1B verifier: speed, memory and p(yes) on chosen windows.

    python scripts/vlm_probe.py negatives samples/C3896.MP4 --n 30      # random close vehicle pairs (no crash)
    python scripts/vlm_probe.py clip some_crash.mp4 --every 1           # sliding windows over a clip, full frame
    python scripts/vlm_probe.py clip some_crash.mp4 --t 12.5 --box 100 200 400 380

A window is 4 frames spaced `--dt` seconds apart, ending at time t.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vlm import QUESTIONS, VLMVerifier, square_crop  # noqa: E402


def read_frames(cap, fps: float, t_end: float, n: int, dt: float) -> list[np.ndarray]:
    out = []
    for k in range(n):
        t = max(0.0, t_end - (n - 1 - k) * dt)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    return out


def close_pairs(frame: np.ndarray, det) -> list[np.ndarray]:
    """Union boxes of the closest vehicle pairs in a frame (what a collision candidate looks like)."""
    res = det.predict(frame, imgsz=960, conf=0.3, classes=[2, 3, 5, 7], verbose=False)[0]
    b = res.boxes.xyxy.cpu().numpy()
    pairs = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            gi, gj = b[i], b[j]
            d = np.hypot((gi[0] + gi[2] - gj[0] - gj[2]) / 2, gi[3] - gj[3])
            size = max(np.hypot(gi[2] - gi[0], gi[3] - gi[1]), np.hypot(gj[2] - gj[0], gj[3] - gj[1]))
            if size > 120 and d < 0.9 * size:
                pairs.append((d / size, np.array([min(gi[0], gj[0]), min(gi[1], gj[1]),
                                                  max(gi[2], gj[2]), max(gi[3], gj[3])])))
    return [u for _, u in sorted(pairs, key=lambda p: p[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["negatives", "clip"])
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=None)
    ap.add_argument("--box", type=float, nargs=4, default=None)
    ap.add_argument("--question", default="accident", choices=list(QUESTIONS))
    ap.add_argument("--save", default=None, help="folder for the crops that were asked about")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import torch

    t0 = time.perf_counter()
    vlm = VLMVerifier()
    print(f"loaded in {time.perf_counter() - t0:.1f}s on {vlm.device} ({vlm.dtype}); "
          f"yes ids {vlm.yes_ids} no ids {vlm.no_ids}")
    q = QUESTIONS[args.question]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    rows = []

    def ask(t: float, box) -> float:
        frames = read_frames(cap, fps, t, args.frames, args.dt)
        crops = [square_crop(f, box) for f in frames]
        p = vlm.p_yes(crops, q)
        rows.append({"t": round(t, 2), "box": None if box is None else [round(float(v)) for v in box], "p": round(p, 4)})
        if save:
            cv2.imwrite(str(save / f"{Path(args.video).stem}_{t:07.2f}_{p:.3f}.jpg"),
                        np.hstack([cv2.resize(c, (224, 224)) for c in crops]))
        print(f"t={t:7.2f}s p(yes)={p:.3f}")
        return p

    if args.mode == "negatives":
        from ultralytics import YOLO

        det = YOLO(str(Path(__file__).resolve().parent.parent / "weights" / "yolo11l.pt"))
        rng = random.Random(0)
        tries = 0
        while len(rows) < args.n and tries < args.n * 10:
            tries += 1
            t = rng.uniform(3, dur - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            pairs = close_pairs(fr, det)
            if pairs:
                ask(t, pairs[0])
    elif args.t is not None:
        ask(args.t, args.box)
    else:
        t = args.dt * (args.frames - 1)
        while t < dur:
            ask(t, args.box)
            t += args.every

    ps = np.array([r["p"] for r in rows])
    if len(ps):
        print(f"\n{len(ps)} windows: mean p={ps.mean():.3f}  max={ps.max():.3f}  >0.5: {(ps > 0.5).sum()}  "
              f">0.8: {(ps > 0.8).sum()}")
        print(f"{vlm.seconds / vlm.calls * 1000:.0f} ms per window ({args.frames} frames), "
              f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if vlm.device == "cuda" else "")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
