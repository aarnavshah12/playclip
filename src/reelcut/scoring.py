"""Involvement scoring — pure functions, no I/O.

Input: identity timeline (target presence per sampled timestamp) + per-frame
observations (ball position). Output: smoothed 0..1 score per timestamp with
reason tags, then thresholded into events.

All distances are normalized by the target bbox height at that timestamp
("player-heights"), so constants are resolution-independent.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import replace

from .config import ReelcutConfig
from .types import (
    BallObs,
    BBox,
    FrameObservation,
    IdentityPoint,
    InvolvementEvent,
    ScorePoint,
)

_EPS = 1e-9
# Speed std-dev (px/s) below which a speed series counts as "idle" for the
# possession correlation gate.
_IDLE_SPEED_STD = 1e-3


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])


def _sample_period(timestamps: list[float]) -> float:
    """Median spacing between consecutive timestamps; 0.0 with < 2 samples."""
    if len(timestamps) < 2:
        return 0.0
    return _median([b - a for a, b in zip(timestamps, timestamps[1:])])


def _is_real_ball(frames: list[FrameObservation], i: int) -> bool:
    """True when frame i exists and carries a non-interpolated ball."""
    if not 0 <= i < len(frames):
        return False
    b = frames[i].ball
    return b is not None and not b.interpolated


def _direction_change_deg(
    a: tuple[float, float], b: tuple[float, float]
) -> float | None:
    """Angle in degrees between velocities a and b; None if either is ~zero."""
    na, nb = math.hypot(*a), math.hypot(*b)
    if na < _EPS or nb < _EPS:
        return None
    cos = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _speeds_correlated(
    ball_speeds: list[float], target_speeds: list[float], min_corr: float
) -> bool:
    """Pearson-correlation gate for possession.

    Degenerate variance: both series idle (near-zero variance) -> correlated
    (kid standing with the ball IS possession); exactly one idle -> not.
    """
    n = len(ball_speeds)
    if n < 2:
        return False
    ma = sum(ball_speeds) / n
    mb = sum(target_speeds) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ball_speeds) / n)
    sb = math.sqrt(sum((x - mb) ** 2 for x in target_speeds) / n)
    a_idle, b_idle = sa < _IDLE_SPEED_STD, sb < _IDLE_SPEED_STD
    if a_idle and b_idle:
        return True
    if a_idle or b_idle:
        return False
    cov = sum(
        (x - ma) * (y - mb) for x, y in zip(ball_speeds, target_speeds)
    ) / n
    return cov / (sa * sb) > min_corr


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def interpolate_ball(
    frames: list[FrameObservation], max_gap_s: float
) -> list[FrameObservation]:
    """Fill short ball gaps by linear interpolation.

    For runs of consecutive frames with ``ball is None`` bounded on both sides
    by real ball detections and shorter than ``max_gap_s``, insert interpolated
    ``BallObs(..., interpolated=True)`` with confidence linearly decayed toward
    the middle of the gap. Longer gaps stay None. Returns new list; input
    unmodified.
    """
    out = list(frames)
    real = [
        i for i, f in enumerate(frames)
        if f.ball is not None and not f.ball.interpolated
    ]
    for a, b in zip(real, real[1:]):
        if b - a < 2:
            continue
        if any(frames[j].ball is not None for j in range(a + 1, b)):
            continue  # already (partially) filled; leave alone
        t0, t1 = frames[a].timestamp_s, frames[b].timestamp_s
        span = t1 - t0
        if span <= 0 or span >= max_gap_s:  # only strictly-shorter gaps
            continue
        ba, bb = frames[a].ball, frames[b].ball
        conf_base = min(ba.confidence, bb.confidence)
        for j in range(a + 1, b):
            f = (frames[j].timestamp_s - t0) / span
            falloff = min(f, 1.0 - f)  # 0 at the bounds, peaks mid-gap
            box = BBox(
                x=ba.bbox.x + (bb.bbox.x - ba.bbox.x) * f,
                y=ba.bbox.y + (bb.bbox.y - ba.bbox.y) * f,
                w=ba.bbox.w + (bb.bbox.w - ba.bbox.w) * f,
                h=ba.bbox.h + (bb.bbox.h - ba.bbox.h) * f,
            )
            out[j] = replace(
                frames[j],
                ball=BallObs(
                    bbox=box,
                    confidence=conf_base * (1.0 - falloff),
                    interpolated=True,
                ),
            )
    return out


def score_timeline(
    identity: list[IdentityPoint],
    frames: list[FrameObservation],
    cfg: ReelcutConfig,
) -> list[ScorePoint]:
    """Compute the raw involvement score at every identity timestamp.

    Components (see config.SportConfig for constants):
      * proximity: 1.0 inside ``ball_near_dist`` player-heights, linear falloff
        to 0 at ``ball_far_dist``; tag "near_ball" when > 0.5.
      * possession: ball inside ``possession_radius`` for >= ``possession_min_s``
        AND ball speed correlates with target speed -> add w_possession,
        tag "possession".
      * touch: ball direction change > ``touch_direction_change_deg`` while
        inside ``touch_radius`` -> add w_touch, tag "touch".
      * sprint fallback: target speed > ``sprint_speed`` player-heights/s ->
        add w_sprint, tag "sprint". Works when ball is unknown.

    Raw score = clamp01(weighted sum) * identity confidence at that timestamp.
    Ball speed/direction are derived from consecutive ball positions; frames
    where the ball is interpolated still count for proximity but not for touch.
    Timestamps where identity confidence is 0 (target absent) score 0.
    """
    sport = cfg.sport
    n_f = len(frames)
    frame_ts = [f.timestamp_s for f in frames]
    period = _sample_period(frame_ts) or (1.0 / cfg.sample_fps)
    tol = period / 2.0 + _EPS

    # Ball centers and finite-difference velocities between ADJACENT samples;
    # a None gap between detections leaves the bordering velocities unset.
    centers = [
        (f.ball.bbox.cx, f.ball.bbox.cy) if f.ball is not None else None
        for f in frames
    ]
    vel_in: list[tuple[float, float] | None] = [None] * n_f
    for i in range(1, n_f):
        c0, c1 = centers[i - 1], centers[i]
        dt = frame_ts[i] - frame_ts[i - 1]
        if c0 is not None and c1 is not None and dt > 0:
            vel_in[i] = ((c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt)

    def vel_out(i: int) -> tuple[float, float] | None:
        return vel_in[i + 1] if i + 1 < n_f else None

    ball_speed: list[float | None] = []
    for i in range(n_f):
        v = vel_in[i] if vel_in[i] is not None else vel_out(i)
        ball_speed.append(math.hypot(*v) if v is not None else None)

    def nearest_frame(t: float) -> int | None:
        """Frame index nearest t, within half a sample period; else None."""
        if not frames:
            return None
        k = bisect_left(frame_ts, t)
        best: int | None = None
        for j in (k - 1, k):
            if 0 <= j < n_f and (
                best is None or abs(frame_ts[j] - t) < abs(frame_ts[best] - t)
            ):
                best = j
        if best is not None and abs(frame_ts[best] - t) <= tol:
            return best
        return None

    # Target speed (px/s): reported speed when present, else finite-differenced
    # from consecutive identity bboxes.
    t_speed: list[float | None] = []
    for i, ip in enumerate(identity):
        if ip.speed is not None:
            t_speed.append(ip.speed)
            continue
        s: float | None = None
        if i > 0 and ip.bbox is not None and identity[i - 1].bbox is not None:
            dt = ip.timestamp_s - identity[i - 1].timestamp_s
            if dt > 0:
                prev = identity[i - 1].bbox
                s = math.hypot(ip.bbox.cx - prev.cx, ip.bbox.cy - prev.cy) / dt
        t_speed.append(s)

    # Ball geometry per identity sample, normalized by target bbox height.
    match: list[int | None] = []
    norm_dist: list[float | None] = []
    for ip in identity:
        fi = nearest_frame(ip.timestamp_s)
        match.append(fi)
        dn: float | None = None
        if fi is not None and ip.bbox is not None and ip.bbox.h > 0:
            fball = frames[fi].ball
            if fball is not None:
                dn = math.hypot(
                    fball.bbox.cx - ip.bbox.cx, fball.bbox.cy - ip.bbox.cy
                ) / ip.bbox.h
        norm_dist.append(dn)

    points: list[ScorePoint] = []
    run_start: int | None = None  # first index of current within-radius run
    for i, ip in enumerate(identity):
        dn = norm_dist[i]
        within = dn is not None and dn <= sport.possession_radius + _EPS
        if within and run_start is None:
            run_start = i
        elif not within:
            run_start = None

        if ip.confidence <= 0.0 or ip.bbox is None or ip.bbox.h <= 0:
            points.append(ScorePoint(ip.timestamp_s, 0.0, ()))
            continue

        raw = 0.0
        tags: list[str] = []

        # proximity
        prox = 0.0
        if dn is not None:
            if dn <= sport.ball_near_dist:
                prox = 1.0
            elif dn < sport.ball_far_dist:
                prox = (sport.ball_far_dist - dn) / (
                    sport.ball_far_dist - sport.ball_near_dist
                )
        raw += sport.w_proximity * prox
        if prox > 0.5:
            tags.append("near_ball")

        # possession: sustained closeness + speed correlation over that window
        if (
            within
            and run_start is not None
            and ip.timestamp_s - identity[run_start].timestamp_s + _EPS
            >= sport.possession_min_s
        ):
            bs: list[float] = []
            ts_: list[float] = []
            for j in range(run_start, i + 1):
                fj = match[j]
                if fj is None or ball_speed[fj] is None or t_speed[j] is None:
                    continue
                bs.append(ball_speed[fj])
                ts_.append(t_speed[j])
            if _speeds_correlated(bs, ts_, sport.possession_speed_corr):
                raw += sport.w_possession
                tags.append("possession")

        # touch: sharp ball direction change nearby, real detections only
        fi = match[i]
        if (
            fi is not None
            and dn is not None
            and dn <= sport.touch_radius + _EPS
            and _is_real_ball(frames, fi)
            and _is_real_ball(frames, fi - 1)
            and _is_real_ball(frames, fi + 1)
        ):
            vi, vo = vel_in[fi], vel_out(fi)
            if vi is not None and vo is not None:
                ang = _direction_change_deg(vi, vo)
                if ang is not None and ang > sport.touch_direction_change_deg:
                    raw += sport.w_touch
                    tags.append("touch")

        # sprint fallback (player-heights/s)
        sp = t_speed[i]
        if sp is not None and sp / ip.bbox.h > sport.sprint_speed:
            raw += sport.w_sprint
            tags.append("sprint")

        score = min(1.0, max(0.0, raw)) * ip.confidence
        points.append(ScorePoint(ip.timestamp_s, score, tuple(tags)))
    return points


def score_opportunities(
    frames: list[FrameObservation], cfg: ReelcutConfig
) -> list[ScorePoint]:
    """Player-agnostic goal-chance score per frame timestamp, tag "goal_chance".

    Needs a ball (interpolated counts, confidence-scaled) and at least one
    detected goal box. Distance = ball center to the nearest edge of the
    closest goal bbox, in units of that goal's bbox height: full weight inside
    ``sport.goal_chance_dist``, linear falloff to zero at
    ``sport.goal_chance_far_dist``. A ball moving faster than
    ``sport.goal_chance_speed`` goal-heights/s while in range adds a shot
    bonus (+0.3, clamped to 1). Identity plays no part — this is the fallback
    that keeps reels alive when the target kid is lost.
    """
    sport = cfg.sport
    n_f = len(frames)
    frame_ts = [f.timestamp_s for f in frames]

    centers = [
        (f.ball.bbox.cx, f.ball.bbox.cy) if f.ball is not None else None
        for f in frames
    ]
    speeds: list[float | None] = [None] * n_f
    for i in range(1, n_f):
        c0, c1 = centers[i - 1], centers[i]
        dt = frame_ts[i] - frame_ts[i - 1]
        if c0 is not None and c1 is not None and dt > 0:
            speeds[i] = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) / dt

    points: list[ScorePoint] = []
    for i, frame in enumerate(frames):
        ball = frame.ball
        if ball is None or not frame.goal_boxes:
            points.append(ScorePoint(frame_ts[i], 0.0, ()))
            continue
        bx, by = ball.bbox.cx, ball.bbox.cy
        best = None  # (normalized distance, goal height)
        for goal in frame.goal_boxes:
            gh = max(goal.h, 1.0)
            dx = max(goal.x - bx, 0.0, bx - goal.x2)
            dy = max(goal.y - by, 0.0, by - goal.y2)
            d = math.hypot(dx, dy) / gh
            if best is None or d < best[0]:
                best = (d, gh)
        d, gh = best
        if d <= sport.goal_chance_dist:
            score = 1.0
        elif d >= sport.goal_chance_far_dist:
            score = 0.0
        else:
            span = sport.goal_chance_far_dist - sport.goal_chance_dist
            score = 1.0 - (d - sport.goal_chance_dist) / span
        if score <= 0.0:
            points.append(ScorePoint(frame_ts[i], 0.0, ()))
            continue
        speed = speeds[i]
        if speed is not None and speed / gh > sport.goal_chance_speed:
            score = min(1.0, score + 0.3)
        # interpolated/low-confidence balls count for less
        conf = ball.confidence if not ball.interpolated else ball.confidence * 0.7
        score *= max(0.3, min(1.0, conf + 0.4))
        points.append(ScorePoint(frame_ts[i], min(1.0, score), ("goal_chance",)))
    return points


def blend_scores(
    primary: list[ScorePoint], fallback: list[ScorePoint], weight: float
) -> list[ScorePoint]:
    """Pointwise max of the target-involvement score and ``weight`` x the
    player-agnostic fallback, matched by timestamp (nearest within half the
    primary sample period; unmatched fallback points are ignored). Tags union
    whenever the fallback meaningfully contributes (> 0.05 weighted)."""
    if not fallback:
        return list(primary)
    if not primary:
        return [
            ScorePoint(p.timestamp_s, min(1.0, p.score * weight), p.tags)
            for p in fallback
        ]
    fb_ts = [p.timestamp_s for p in fallback]
    period = _sample_period([p.timestamp_s for p in primary])
    tol = (period / 2.0 + _EPS) if period else _EPS
    out: list[ScorePoint] = []
    for p in primary:
        k = bisect_left(fb_ts, p.timestamp_s)
        match: ScorePoint | None = None
        for j in (k - 1, k):
            if 0 <= j < len(fallback) and (
                match is None
                or abs(fb_ts[j] - p.timestamp_s) < abs(match.timestamp_s - p.timestamp_s)
            ):
                match = fallback[j]
        if match is None or abs(match.timestamp_s - p.timestamp_s) > tol:
            out.append(p)
            continue
        weighted = min(1.0, match.score * weight)
        if weighted <= 0.05:
            out.append(p)
        else:
            tags = tuple(sorted(set(p.tags) | set(match.tags)))
            out.append(ScorePoint(p.timestamp_s, max(p.score, weighted), tags))
    return out


def smooth_scores(
    points: list[ScorePoint], window_s: float
) -> list[ScorePoint]:
    """Centered rolling-mean over ``window_s``; tags are unioned over the window."""
    if not points:
        return []
    ts = [p.timestamp_s for p in points]
    half = window_s / 2.0 + _EPS
    out: list[ScorePoint] = []
    for p in points:
        lo = bisect_left(ts, p.timestamp_s - half)
        hi = bisect_right(ts, p.timestamp_s + half)
        window = points[lo:hi]
        mean = sum(q.score for q in window) / len(window)
        tags = tuple(sorted({t for q in window for t in q.tags}))
        out.append(ScorePoint(p.timestamp_s, mean, tags))
    return out


def extract_events(
    points: list[ScorePoint], threshold: float, min_duration_s: float
) -> list[InvolvementEvent]:
    """Contiguous runs of score >= threshold lasting >= min_duration_s.

    Event tags = union of member tags; peak/mean over member scores.
    """
    period = _sample_period([p.timestamp_s for p in points])
    events: list[InvolvementEvent] = []
    i, n = 0, len(points)
    while i < n:
        if points[i].score < threshold:
            i += 1
            continue
        j = i
        while j + 1 < n and points[j + 1].score >= threshold:
            j += 1
        run = points[i : j + 1]
        duration = run[-1].timestamp_s - run[0].timestamp_s + period
        if duration + _EPS >= min_duration_s:
            scores = [p.score for p in run]
            events.append(
                InvolvementEvent(
                    start_s=run[0].timestamp_s,
                    end_s=run[-1].timestamp_s,
                    peak_score=max(scores),
                    mean_score=sum(scores) / len(scores),
                    tags=tuple(sorted({t for p in run for t in p.tags})),
                )
            )
        i = j + 1
    return events
