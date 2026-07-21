"""Stage 1.5 — jersey-number binding over tracklets (read-until-bound).

The user-specified semantics: a tracked player with no bound number gets
digit-model attempts spread across their track's lifetime; the FIRST clean
read binds the number to that track for good and attempts stop — the tracker
carries the identity from there. When the player leaves the frame the track
dies; their next track repeats the process.

This runs locally over stage-1 observations (works identically for local and
batch-GPU stage 1), decoding the source video in ONE sequential pass and
evaluating only the attempts that are due, skipping every track already
bound. Cost is bounded: at most ``number_max_attempts`` reads per tracklet,
zero after a successful bind.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ReelcutConfig
from .types import FrameObservation, OcrRead

# Reads a player crop -> (digits, confidence) or None when nothing legible.
Reader = Callable[[np.ndarray], "tuple[str, float] | None"]

_CROP_MARGIN = 0.10   # widen player boxes slightly so numbers at the edge survive


def make_digit_reader(
    model_id: str, api_key: str, min_conf: float
) -> Reader:
    """Digit-detector-backed reader: detections' classes ARE the characters;
    left-to-right x-order assembles the number (same idea as the workflow's
    stitch block). Confidence = weakest digit's confidence."""
    from .workflow_client import _prepare_inference_env

    _prepare_inference_env()
    from inference import get_model

    model = get_model(model_id, api_key=api_key)

    def read(crop: np.ndarray) -> tuple[str, float] | None:
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            return None
        r = model.infer(crop, confidence=min_conf)
        preds = r[0].predictions if isinstance(r, list) else r.predictions
        digits = sorted(
            ((p.x, str(p.class_name), float(p.confidence)) for p in preds
             if str(p.class_name).isdigit()),
            key=lambda t: t[0],
        )
        if not digits:
            return None
        return "".join(d[1] for d in digits), min(d[2] for d in digits)

    return read


def _prebound(frames: list[FrameObservation]) -> dict[int, str]:
    """Tracks already bound by existing reads (all non-empty reads agree)."""
    seen: dict[int, set[str]] = defaultdict(set)
    for f in frames:
        for r in f.ocr:
            digits = "".join(c for c in r.text if c.isdigit())
            if digits:
                seen[r.track_id].add(digits)
    return {tid: nums.copy().pop() for tid, nums in seen.items() if len(nums) == 1}


def bind_numbers(
    frames: list[FrameObservation],
    video: Path,
    cfg: ReelcutConfig,
    reader: Reader,
) -> tuple[list[FrameObservation], dict[int, str]]:
    """Returns (frames enriched with the new OcrReads, track_id -> number).

    Attempt schedule per unbound track: observations spaced at least
    ``1 / cfg.number_attempt_hz`` seconds apart, at most
    ``cfg.number_max_attempts`` per track, evaluated in one sequential decode
    of the video. A clean read binds immediately; conflicting later evidence
    never accrues because bound tracks are skipped (the tracker owns identity
    from then on, per the read-once design).
    """
    import cv2

    bound: dict[int, str] = _prebound(frames)
    min_gap_s = 1.0 / max(cfg.number_attempt_hz, 1e-6)

    # attempt plan: frame_index -> [(track_id, bbox)]
    plan: dict[int, list] = defaultdict(list)
    attempts_left: dict[int, int] = {}
    last_attempt_ts: dict[int, float] = {}
    for f in frames:
        for p in f.players:
            tid = p.track_id
            if tid in bound:
                continue
            if attempts_left.get(tid, cfg.number_max_attempts) <= 0:
                continue
            if f.timestamp_s - last_attempt_ts.get(tid, -1e9) < min_gap_s:
                continue
            last_attempt_ts[tid] = f.timestamp_s
            attempts_left[tid] = attempts_left.get(tid, cfg.number_max_attempts) - 1
            plan[f.frame_index].append((tid, p.bbox))

    new_reads: dict[int, list[OcrRead]] = defaultdict(list)   # frame_index -> reads
    if plan and video.exists():
        cap = cv2.VideoCapture(str(video))
        try:
            src_idx = 0
            pending = sorted(plan.keys())
            pi = 0
            while pi < len(pending):
                ok, img = cap.read()
                if not ok or img is None:
                    break
                if src_idx == pending[pi]:
                    fh, fw = img.shape[:2]
                    for tid, bbox in plan[src_idx]:
                        if tid in bound:      # bound earlier in this same pass
                            continue
                        mx, my = bbox.w * _CROP_MARGIN, bbox.h * _CROP_MARGIN
                        x0 = max(0, int(bbox.x - mx)); y0 = max(0, int(bbox.y - my))
                        x1 = min(fw, int(bbox.x2 + mx)); y1 = min(fh, int(bbox.y2 + my))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        result = reader(img[y0:y1, x0:x1])
                        if result is not None and result[0]:
                            number, conf = result
                            bound[tid] = number
                            new_reads[src_idx].append(
                                OcrRead(track_id=tid, text=number, confidence=conf)
                            )
                    pi += 1
                src_idx += 1
        finally:
            cap.release()

    if not new_reads:
        return list(frames), bound
    enriched = [
        replace(f, ocr=f.ocr + tuple(new_reads[f.frame_index]))
        if f.frame_index in new_reads else f
        for f in frames
    ]
    return enriched, bound
