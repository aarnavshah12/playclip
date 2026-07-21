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


def assemble_number(
    digits: "list[tuple[float, float, str, float]]", crop_width: float
) -> tuple[str, float] | None:
    """(x, width, char, conf) detections -> the crop owner's number.

    Overlapping players put TWO kids' digits in one crop (measured: reads
    like "108" and "714" from a 10 next to an 8). Cluster digits by x-gap
    (a gap wider than 1.6x the median digit width separates players), keep
    the cluster nearest the crop's horizontal center, and reject 3+ digit
    results as ambiguous — youth numbers are 1-2 digits, and no read beats
    a wrong bind (the schedule simply tries again later).
    """
    if not digits:
        return None
    digits = sorted(digits, key=lambda d: d[0])
    widths = sorted(d[1] for d in digits)
    w_med = max(widths[len(widths) // 2], 1.0)
    clusters: list[list[tuple[float, float, str, float]]] = [[digits[0]]]
    for d in digits[1:]:
        if d[0] - clusters[-1][-1][0] > 1.6 * w_med:
            clusters.append([d])
        else:
            clusters[-1].append(d)
    center = crop_width / 2.0
    best = min(
        clusters,
        key=lambda c: abs(sum(d[0] for d in c) / len(c) - center),
    )
    if len(best) > 2:
        return None
    return "".join(d[2] for d in best), min(d[3] for d in best)


def make_digit_reader(
    model_id: str, api_key: str, min_conf: float
) -> Reader:
    """Digit-detector-backed reader: detections' classes ARE the characters;
    assemble_number picks the crop owner's digits."""
    from .workflow_client import _prepare_inference_env

    _prepare_inference_env()
    from inference import get_model

    model = get_model(model_id, api_key=api_key)

    def read(crop: np.ndarray) -> tuple[str, float] | None:
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            return None
        r = model.infer(crop, confidence=min_conf)
        preds = r[0].predictions if isinstance(r, list) else r.predictions
        digits = [
            (float(p.x), float(p.width), str(p.class_name), float(p.confidence))
            for p in preds if str(p.class_name).isdigit()
        ]
        return assemble_number(digits, float(crop.shape[1]))

    return read


def merge_reads(reads: list[str]) -> tuple[str | None, bool]:
    """Reconcile a track's reads -> (value, confirmed).

    Two reads agree when one is a substring of the other (a "1" off a "15"
    shirt is a partial read of the same number); the agreed value is the
    longest. Returns (majority value, True) once >= 2 reads agree, a single
    read as (value, False) — provisional, keep attempting — and (None, False)
    for unresolved conflicts, because a wrong lock is worse than no label.
    """
    if not reads:
        return None, False
    groups: list[list[str]] = []
    for read in reads:
        for g in groups:
            if all(read in v or v in read for v in g):
                g.append(read)
                break
        else:
            groups.append([read])
    groups.sort(key=len, reverse=True)
    best = groups[0]
    if len(best) >= 2 and (len(groups) == 1 or len(best) > len(groups[1])):
        return max(best, key=len), True
    if len(groups) == 1 and len(best) == 1:
        return best[0], False
    return None, False


def _existing_reads(frames: list[FrameObservation]) -> dict[int, list[str]]:
    seen: dict[int, list[str]] = defaultdict(list)
    for f in frames:
        for r in f.ocr:
            digits = "".join(c for c in r.text if c.isdigit())
            if digits:
                seen[r.track_id].append(digits)
    return seen


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

    pools: dict[int, list[str]] = _existing_reads(frames)
    confirmed: set[int] = {
        tid for tid, reads in pools.items() if merge_reads(reads)[1]
    }
    min_gap_s = 1.0 / max(cfg.number_attempt_hz, 1e-6)

    # attempt plan: frame_index -> [(track_id, bbox)]
    plan: dict[int, list] = defaultdict(list)
    attempts_left: dict[int, int] = {}
    last_attempt_ts: dict[int, float] = {}
    for f in frames:
        for p in f.players:
            tid = p.track_id
            if tid in confirmed:
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
                        if tid in confirmed:  # confirmed earlier in this pass
                            continue
                        mx, my = bbox.w * _CROP_MARGIN, bbox.h * _CROP_MARGIN
                        x0 = max(0, int(bbox.x - mx)); y0 = max(0, int(bbox.y - my))
                        x1 = min(fw, int(bbox.x2 + mx)); y1 = min(fh, int(bbox.y2 + my))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        result = reader(img[y0:y1, x0:x1])
                        if result is not None and result[0]:
                            number, conf = result
                            pools[tid].append(number)
                            new_reads[src_idx].append(
                                OcrRead(track_id=tid, text=number, confidence=conf)
                            )
                            if merge_reads(pools[tid])[1]:
                                confirmed.add(tid)   # two agreeing reads: lock
                    pi += 1
                src_idx += 1
        finally:
            cap.release()

    bound = {
        tid: value
        for tid, reads in pools.items()
        for value, _ok in [merge_reads(reads)]
        if value is not None
    }

    if not new_reads:
        return list(frames), bound
    enriched = [
        replace(f, ocr=f.ocr + tuple(new_reads[f.frame_index]))
        if f.frame_index in new_reads else f
        for f in frames
    ]
    return enriched, bound
