"""Debug video renderer (local, from cached stage outputs).

Draws on sampled frames: all tracked players (thin gray box + track id),
ball (yellow circle), target (THICK VIOLET box — Roboflow Violet 600
#7C3AED -> BGR (237, 58, 124)), identity confidence + involvement score bar,
active tags. Output plays at the sampling fps so time maps 1:1 to analysis.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ReelcutConfig
from .types import FrameObservation, IdentityPoint, ScorePoint

VIOLET_BGR = (237, 58, 124)   # Roboflow Violet 600 (#7C3AED) in BGR

_GRAY_BGR = (160, 160, 160)
_YELLOW_BGR = (0, 255, 255)
_WHITE_BGR = (240, 240, 240)
_DARK_BGR = (30, 30, 30)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BAR_H = 18   # px, bottom score-bar strip


def _ts_key(timestamp_s: float) -> int:
    """Millisecond bucket so float timestamps line up across stage outputs."""
    return round(timestamp_s * 1000.0)


def _put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """Text with a dark halo so it reads over grass and kits alike."""
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _lerp_bbox(a, b, t: float):
    from .types import BBox

    return BBox(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.w + (b.w - a.w) * t,
        a.h + (b.h - a.h) * t,
    )


def _tween(prev: FrameObservation, nxt: FrameObservation | None, t: float) -> FrameObservation:
    """Interpolated view between two sampled observations so annotations move
    at the SOURCE frame rate instead of holding still between samples.
    Tracks present in both samples glide; everything else holds the previous
    sample (a vanished track shouldn't slide toward a stranger)."""
    from dataclasses import replace

    if nxt is None or t <= 0.0:
        return prev
    next_by_id = {p.track_id: p for p in nxt.players}
    players = tuple(
        replace(p, bbox=_lerp_bbox(p.bbox, next_by_id[p.track_id].bbox, t))
        if p.track_id in next_by_id else p
        for p in prev.players
    )
    ball = prev.ball
    if prev.ball is not None and nxt.ball is not None:
        ball = replace(prev.ball, bbox=_lerp_bbox(prev.ball.bbox, nxt.ball.bbox, t))
    return replace(prev, players=players, ball=ball)


def _tween_identity(ip: IdentityPoint | None, ip_next: IdentityPoint | None, t: float):
    from dataclasses import replace

    if (
        ip is None or ip_next is None or t <= 0.0
        or ip.bbox is None or ip_next.bbox is None
        or ip.track_id != ip_next.track_id
    ):
        return ip
    return replace(ip, bbox=_lerp_bbox(ip.bbox, ip_next.bbox, t))


def _jersey_numbers(frames: list[FrameObservation]) -> dict[int, str]:
    """track_id -> display label: confirmed numbers plain ("15"), single
    unconfirmed reads marked provisional ("1?"), conflicts unlabeled."""
    from .numbers import merge_reads

    seen: dict[int, list[str]] = {}
    for fo in frames:
        for read in fo.ocr:
            digits = "".join(c for c in read.text if c.isdigit())
            if digits:
                seen.setdefault(read.track_id, []).append(digits)
    out: dict[int, str] = {}
    for tid, reads in seen.items():
        value, ok = merge_reads(reads)
        if value is not None:
            out[tid] = value if ok else value + "?"
    return out


def _draw_players(
    img: np.ndarray, fo: FrameObservation, numbers: dict[int, str]
) -> None:
    for p in fo.players:
        b = p.bbox
        number = numbers.get(p.track_id)
        if number is not None:
            # known jersey number: the label a parent actually understands
            cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x2), int(b.y2)), _WHITE_BGR, 2)
            _put_text(img, f"#{number}", (int(b.x), max(14, int(b.y) - 4)), _WHITE_BGR, 0.55, 2)
        else:
            cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x2), int(b.y2)), _GRAY_BGR, 1)
            _put_text(img, str(p.track_id), (int(b.x), max(12, int(b.y) - 4)), _GRAY_BGR, 0.4)


def _draw_ball(img: np.ndarray, fo: FrameObservation) -> None:
    if fo.ball is None:
        return
    b = fo.ball.bbox
    radius = max(4, int(max(b.w, b.h) / 2))
    thickness = 1 if fo.ball.interpolated else 2   # thin ring = interpolated
    cv2.circle(img, (int(b.cx), int(b.cy)), radius, _YELLOW_BGR, thickness)


def _draw_target(img: np.ndarray, ip: IdentityPoint | None) -> None:
    if ip is None or ip.bbox is None or ip.confidence <= 0:
        return
    b = ip.bbox
    cv2.rectangle(img, (int(b.x), int(b.y)), (int(b.x2), int(b.y2)), VIOLET_BGR, 3)
    _put_text(
        img,
        f"TARGET {ip.confidence:.2f}",
        (int(b.x), max(14, int(b.y) - 8)),
        VIOLET_BGR,
        0.5,
        2,
    )


def _draw_hud(
    img: np.ndarray,
    fo: FrameObservation,
    ip: IdentityPoint | None,
    sp: ScorePoint | None,
    have_scores: bool,
) -> None:
    id_conf = ip.confidence if ip is not None else 0.0
    _put_text(img, f"t={fo.timestamp_s:.1f}s  id_conf={id_conf:.2f}", (8, 20), _WHITE_BGR)
    if not have_scores:
        return
    score = sp.score if sp is not None else 0.0
    tags = ",".join(sp.tags) if sp is not None and sp.tags else "-"
    tag_color = VIOLET_BGR if tags != "-" else _WHITE_BGR
    _put_text(img, f"tags: {tags}", (8, 40), tag_color)
    # bottom strip: violet fill proportional to the smoothed score
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - _BAR_H), (w, h), _DARK_BGR, -1)
    fill = int(round(max(0.0, min(1.0, score)) * w))
    if fill > 0:
        cv2.rectangle(img, (0, h - _BAR_H), (fill, h), VIOLET_BGR, -1)
    _put_text(img, f"score {score:.2f}", (8, h - 5), _WHITE_BGR, 0.4)


def render_debug_video(
    video: Path,
    out_path: Path,
    frames: list[FrameObservation],
    identity: list[IdentityPoint],
    scores: list[ScorePoint] | None,
    cfg: ReelcutConfig,
) -> None:
    """Sequentially decode EVERY source frame, drawing the most recent sampled
    observation on each (hold-last), so the output plays at the source frame
    rate with annotations updating at the sampling rate. Players with a
    cleanly-read jersey number are labeled "#<number>"; others show their
    track id. Missing scores (None) -> identity info only."""
    if not frames:
        return
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open video for debug render: {video}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or cfg.sample_fps
    w, h = frames[0].frame_w, frames[0].frame_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"cannot open debug video writer: {out_path}")

    have_scores = scores is not None
    numbers = _jersey_numbers(frames)
    id_by_ts = {_ts_key(p.timestamp_s): p for p in identity}
    score_by_ts = {_ts_key(p.timestamp_s): p for p in (scores or [])}

    try:
        fi = 0
        src_idx = 0
        while True:
            ok, img = cap.read()
            if not ok or img is None:
                break
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            while fi + 1 < len(frames) and frames[fi + 1].frame_index <= src_idx:
                fi += 1
            fo = frames[fi]
            if fo.frame_index <= src_idx:   # first sample may start later
                nxt = frames[fi + 1] if fi + 1 < len(frames) else None
                span = (nxt.frame_index - fo.frame_index) if nxt else 0
                t = (src_idx - fo.frame_index) / span if span > 0 else 0.0
                t = min(max(t, 0.0), 1.0)
                view = _tween(fo, nxt, t)
                ip = id_by_ts.get(_ts_key(fo.timestamp_s))
                ip_next = id_by_ts.get(_ts_key(nxt.timestamp_s)) if nxt else None
                ip_view = _tween_identity(ip, ip_next, t)
                sp = score_by_ts.get(_ts_key(fo.timestamp_s))
                _draw_players(img, view, numbers)
                _draw_ball(img, view)
                _draw_target(img, ip_view)
                _draw_hud(img, view, ip_view, sp, have_scores)
            writer.write(img)
            src_idx += 1
    finally:
        cap.release()
        writer.release()
