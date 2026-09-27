"""Vision-language verifier: OpenGVLab/InternVL2_5-1B (MIT), loaded from weights/InternVL2_5-1B.

The model is never run on the whole video. The rules propose short candidate
windows (two vehicles touching and then stopping, a crash/fire hint of the
anomaly model), and the verifier answers one yes/no question about a few frames
of each window. The answer is read from a single forward pass as
p(yes) = softmax(logit("Yes"), logit("No")), so it is deterministic and costs
no token generation.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

VLM_DIR = Path(__file__).resolve().parent.parent / "weights" / "InternVL2_5-1B"
TILE = 448                                   # InternVL input size; one tile per frame
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

QUESTIONS = {
    "accident": ("These are consecutive frames from a road traffic camera. "
                 "Has a traffic accident or collision happened (vehicles crashed into each other, "
                 "or a vehicle hit a person or an object)? Answer Yes or No."),
    "fire_smoke": ("This is a frame from a road traffic camera. "
                   "Is there fire or smoke visible? Answer Yes or No."),
}


def available() -> bool:
    return (VLM_DIR / "model.safetensors").exists() and (VLM_DIR / "modeling_internvl_chat.py").exists()


def square_crop(frame: np.ndarray, box, scale: float = 2.5, min_side: int = 448) -> np.ndarray:
    """Square region around `box` (x1, y1, x2, y2), `scale` times its larger side, clipped to the frame."""
    h, w = frame.shape[:2]
    if box is None:
        return frame
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(min_side, scale * max(x2 - x1, y2 - y1))
    side = min(side, w, h)
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    s = int(round(side))
    return frame[y0:y0 + s, x0:x0 + s]


def _to_tile(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (TILE, TILE), interpolation=cv2.INTER_AREA if rgb.shape[0] > TILE else cv2.INTER_CUBIC)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


class VLMVerifier:
    def __init__(self, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
        # fp16 on GPU: the T4 has no bfloat16 support
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(str(VLM_DIR), trust_remote_code=True, use_fast=False,
                                                       local_files_only=True)
        self.model = AutoModel.from_pretrained(str(VLM_DIR), torch_dtype=self.dtype, trust_remote_code=True,
                                               use_flash_attn=False,
                                               local_files_only=True).eval().to(self.device)
        self.img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = self.img_ctx_id
        self.yes_ids = self._first_ids(["Yes", "yes", " Yes"])
        self.no_ids = self._first_ids(["No", "no", " No"])
        self.calls = 0
        self.seconds = 0.0

    def _first_ids(self, words: list[str]) -> list[int]:
        ids = {self.tokenizer.encode(w, add_special_tokens=False)[0] for w in words}
        return sorted(ids)

    def _prompt(self, question: str, n_images: int) -> str:
        from importlib import import_module

        conv = import_module(type(self.model).__module__.rsplit(".", 1)[0] + ".conversation")
        template = conv.get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.model.num_image_token + "</img>"
        body = "".join(f"Frame{i + 1}: {image_tokens}\n" for i in range(n_images)) + question
        template.append_message(template.roles[0], body)
        template.append_message(template.roles[1], None)
        return template.get_prompt()

    def p_yes(self, images_bgr: list[np.ndarray], question: str) -> float:
        """Probability that the answer to `question` about `images_bgr` (in time order) is Yes."""
        torch = self.torch
        t0 = time.perf_counter()
        pixel_values = torch.from_numpy(np.stack([_to_tile(im) for im in images_bgr])).to(self.device, self.dtype)
        ids = self.tokenizer(self._prompt(question, len(images_bgr)), return_tensors="pt").input_ids.to(self.device)
        with torch.inference_mode():
            vit = self.model.extract_feature(pixel_values)
            emb = self.model.language_model.get_input_embeddings()(ids)
            sel = ids[0] == self.img_ctx_id
            emb[0, sel] = vit.reshape(-1, emb.shape[-1]).to(emb.dtype)
            logits = self.model.language_model(inputs_embeds=emb).logits[0, -1].float()
        yes = torch.logsumexp(logits[self.yes_ids], 0)
        no = torch.logsumexp(logits[self.no_ids], 0)
        p = float(torch.sigmoid(yes - no))
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return p


_VERIFIER: VLMVerifier | None = None
_LOAD_FAILED = False


def get_verifier() -> VLMVerifier | None:
    """The shared verifier (loaded once per process), or None if the weights are missing or loading fails:
    Part A then simply runs without it."""
    global _VERIFIER, _LOAD_FAILED
    if _VERIFIER is None and not _LOAD_FAILED:
        if not available():
            _LOAD_FAILED = True
            return None
        try:
            _VERIFIER = VLMVerifier()
        except Exception as exc:  # noqa: BLE001 - any failure here must not stop Part A
            print(f"[vlm] verifier disabled: {type(exc).__name__}: {exc}")
            _LOAD_FAILED = True
    return _VERIFIER


class FrameWindowVerifier:
    """Keeps the last few seconds of frames and answers the rule engine's questions about them,
    within a per-video call and time budget."""

    def __init__(self, vlm: VLMVerifier, duration: float, buffer_sec: float, buffer_width: int,
                 max_calls: int, max_frac: float) -> None:
        from collections import deque

        self.vlm = vlm
        self.buffer_sec = buffer_sec
        self.buffer_width = buffer_width
        self.frames: deque = deque()                 # (t, frame, scale from video px to buffer px)
        self.max_calls = max_calls
        self.max_sec = max_frac * max(duration, 1.0)
        self.calls = 0
        self.seconds = 0.0

    def push(self, t: float, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        scale = min(1.0, self.buffer_width / w)
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.frames.append((t, frame, scale))
        while self.frames and t - self.frames[0][0] > self.buffer_sec:
            self.frames.popleft()

    def _nearest(self, t: float):
        if not self.frames:
            return None
        best = min(self.frames, key=lambda f: abs(f[0] - t))
        return best if abs(best[0] - t) <= 0.5 else None

    def __call__(self, kind: str, t_from: float, t_to: float, box, n_frames: int) -> float | None:
        if self.calls >= self.max_calls or self.seconds >= self.max_sec:
            return None
        times = [t_from] if n_frames == 1 else list(np.linspace(t_from, t_to, n_frames))
        picked = [f for f in (self._nearest(tt) for tt in times) if f is not None]
        if len(picked) < max(1, n_frames - 1):
            return None
        crops = [square_crop(fr, None if box is None else np.asarray(box, dtype=np.float64) * sc)
                 for _, fr, sc in picked]
        t0 = time.perf_counter()
        try:
            p = self.vlm.p_yes(crops, QUESTIONS[kind])
        except Exception as exc:  # noqa: BLE001
            print(f"[vlm] question failed: {type(exc).__name__}: {exc}")
            p = None
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return p
