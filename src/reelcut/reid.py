"""Appearance re-identification: is this track the seed kid?

The identity backbone (product decision 2026-07-21): the parent's single
seed click yields a reference bank of appearance embeddings (hair, build,
shoes — everything a generic CLIP image encoder captures at player-crop
scale), harvested automatically from the seed tracklet. Every other tracklet
is scored by cosine similarity against the bank, so the kid is re-acquired
when they re-enter the frame — no per-child training, no jersey read
required. Jersey numbers remain as confirmation evidence only.

Pure logic is separated from I/O: crop harvesting decodes the video once,
sequentially, like numbers.bind_numbers; scoring and label application are
pure functions unit-tested with fake embedders.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ReelcutConfig
from .types import IdentityLabel, LabeledTracklet, Tracklet

# Maps a BGR player crop -> 1-D unit-norm embedding.
Embedder = Callable[[np.ndarray], np.ndarray]

_CROP_MARGIN = 0.05


def make_clip_embedder(api_key: str) -> Embedder:
    """CLIP image encoder via the local inference package (no cloud calls
    after the one-time weight download)."""
    from .workflow_client import _prepare_inference_env

    _prepare_inference_env()
    from inference.models import Clip

    model = Clip(api_key=api_key)

    def embed(crop: np.ndarray) -> np.ndarray:
        v = np.asarray(model.embed_image(crop), dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    return embed


def sample_points(
    tracklet: Tracklet, n: int, min_crop_h: float
) -> list:
    """Up to n TrackletPoints spread evenly across the track's life, biggest
    crops preferred within each segment (big crop = clean appearance)."""
    eligible = [p for p in tracklet.points if p.bbox.h >= min_crop_h]
    if not eligible or n <= 0:
        return []
    if len(eligible) <= n:
        return list(eligible)
    segments = np.array_split(np.arange(len(eligible)), n)
    return [
        max((eligible[i] for i in seg), key=lambda p: p.bbox.h)
        for seg in segments if len(seg)
    ]


def harvest_crops(
    video: Path, wanted: "dict[int, list[tuple[int, object]]]"
) -> "dict[tuple[int, int], np.ndarray]":
    """Decode the video once; return crops for {frame_index: [(track_id,
    bbox)]} as {(frame_index, track_id): BGR image}."""
    import cv2

    out: dict[tuple[int, int], np.ndarray] = {}
    if not wanted or not video.exists():
        return out
    cap = cv2.VideoCapture(str(video))
    try:
        pending = sorted(wanted)
        pi = 0
        src_idx = 0
        while pi < len(pending):
            ok, img = cap.read()
            if not ok or img is None:
                break
            if src_idx == pending[pi]:
                fh, fw = img.shape[:2]
                for tid, bbox in wanted[src_idx]:
                    mx, my = bbox.w * _CROP_MARGIN, bbox.h * _CROP_MARGIN
                    x0 = max(0, int(bbox.x - mx)); y0 = max(0, int(bbox.y - my))
                    x1 = min(fw, int(bbox.x2 + mx)); y1 = min(fh, int(bbox.y2 + my))
                    if x1 > x0 and y1 > y0:
                        out[(src_idx, tid)] = img[y0:y1, x0:x1]
                pi += 1
            src_idx += 1
    finally:
        cap.release()
    return out


def track_embeddings(
    tracklets: list[Tracklet],
    seed_track_id: int,
    video: Path,
    embedder: Embedder,
    cfg: ReelcutConfig,
) -> "dict[int, np.ndarray]":
    """track_id -> (K, D) matrix of crop embeddings.

    The seed track contributes ``reid_ref_samples`` crops (it is the
    reference bank); other tracks ``reid_samples_per_track``. Tracks with no
    crop tall enough to embed are absent — no evidence, not negative
    evidence.
    """
    by_id = {t.track_id: t for t in tracklets}
    if seed_track_id not in by_id:
        return {}
    plan: dict[int, list] = {}

    def want(tid: int, points: list) -> None:
        for p in points:
            plan.setdefault(p.frame_index, []).append((tid, p.bbox))

    for t in tracklets:
        n = cfg.reid_ref_samples if t.track_id == seed_track_id else cfg.reid_samples_per_track
        want(t.track_id, sample_points(t, n, cfg.reid_min_crop_h))

    crops = harvest_crops(video, plan)
    vecs: dict[int, list[np.ndarray]] = {}
    for (_, tid), crop in crops.items():
        vecs.setdefault(tid, []).append(embedder(crop))
    return {tid: np.stack(v) for tid, v in vecs.items()}


def contrastive_scores(
    embeds: "dict[int, np.ndarray]",
    seed_track_id: int,
    negative_ids: "set[int]",
) -> "dict[int, tuple[float, float]]":
    """track_id -> (sim to seed bank, best sim to any negative bank).

    Absolute CLIP similarities between full player crops are useless here —
    measured on the CSKA fixture, every kid on the pitch lands in a
    0.88-0.95 band because kit and grass dominate the embedding. What
    separates identities is the CONTRAST: the target's other tracks sit
    closer to the seed bank than to the banks of kids independently known to
    be someone else (dominant jersey read of a different number). Both sims
    are max-over-pairs; tracks in ``negative_ids`` score their negative sim
    against the OTHER negative banks, not themselves.
    """
    seed_bank = embeds.get(seed_track_id)
    if seed_bank is None:
        return {}
    neg_banks = {
        tid: embeds[tid] for tid in negative_ids
        if tid in embeds and tid != seed_track_id
    }
    out: dict[int, tuple[float, float]] = {}
    for tid, vecs in embeds.items():
        s_pos = float(np.max(seed_bank @ vecs.T))
        s_neg = 0.0
        for ntid, bank in neg_banks.items():
            if ntid == tid:
                continue
            s_neg = max(s_neg, float(np.max(bank @ vecs.T)))
        out[tid] = (s_pos, s_neg)
    return out


def negative_track_ids(
    tracklets: list[Tracklet], jersey: str, min_reads: int
) -> "set[int]":
    """Tracks whose DOMINANT confident jersey read is a different number.

    Free, high-precision negative references for contrastive scoring: 100+
    agreeing reads of "8" say "not the 7 kid" far more reliably than any
    appearance threshold. Substring-compatible dominants (partials/fusions
    of the target number) are excluded."""
    from collections import Counter

    jd = "".join(c for c in jersey if c.isdigit())
    out: set[int] = set()
    for t in tracklets:
        counts: Counter[str] = Counter()
        for r in t.ocr_reads:
            digits = "".join(ch for ch in r.text if ch.isdigit())
            if digits:
                counts[digits] += 1
        if not counts:
            continue
        value, n = counts.most_common(1)[0]
        total = sum(counts.values())
        if (
            total >= min_reads
            and n >= 0.6 * total
            and value != jd
            and value not in jd
            and jd not in value
        ):
            out.add(t.track_id)
    return out


def apply_reid(
    labeled: list[LabeledTracklet],
    scores: "dict[int, tuple[float, float]]",
    seed_track_id: int,
    cfg: ReelcutConfig,
) -> list[LabeledTracklet]:
    """Fold contrastive appearance scores into track labels.

    * UNKNOWN, closer to the seed bank than to every known-other-kid bank by
      >= reid_contrast_margin (and s_pos >= reid_min_sim) -> TARGET;
      confidence 0.6-0.9 scaled by the margin (evidence "reid"/"reid_neg").
    * TARGET (non-seed), closer to a negative bank by >= the margin ->
      back to UNKNOWN: appearance contrast outranks whatever promoted it.
    * NOT_TARGET stays NOT_TARGET; unembedded tracks change nothing.
    """
    out: list[LabeledTracklet] = []
    for lt in labeled:
        tid = lt.tracklet.track_id
        score = scores.get(tid)
        if tid == seed_track_id or score is None:
            out.append(lt)
            continue
        s_pos, s_neg = score
        contrast = s_pos - s_neg
        evidence = dict(lt.evidence)
        evidence["reid"] = s_pos
        evidence["reid_neg"] = s_neg
        if (
            lt.label is IdentityLabel.UNKNOWN
            and contrast >= cfg.reid_contrast_margin
            and s_pos >= cfg.reid_min_sim
        ):
            conf = 0.6 + min(0.3, (contrast - cfg.reid_contrast_margin) * 10.0)
            out.append(replace(lt, label=IdentityLabel.TARGET,
                               confidence=conf, evidence=evidence))
        elif (
            lt.label is IdentityLabel.TARGET
            and -contrast >= cfg.reid_contrast_margin
        ):
            out.append(replace(lt, label=IdentityLabel.UNKNOWN,
                               confidence=0.0, evidence=evidence))
        else:
            out.append(replace(lt, evidence=evidence))
    return out
