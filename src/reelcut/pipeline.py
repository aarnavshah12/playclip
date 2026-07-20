"""Stage orchestration: INFER -> IDENTITY -> SCORE -> CUT (+ optional debug).

Each stage: pure core (other modules) + thin cached I/O here. One-line health
summary printed per stage, e.g.:
  [stage1 infer]    3012 frames, ball in 61.2% of frames, 48 track ids
  [stage2 identity] 39 tracklets -> 6 target / 21 not_target / 12 unknown, coverage 71%
  [stage3 score]    14 events, mean score 0.44
  [stage4 cut]      9 clips, 96.5s total (8.0% of source)
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from . import clipcutter, ffmpeg, scoring, stitching
from .cache import StageCache, video_cache_key
from .config import Paths, ReelcutConfig
from .types import (
    Clip,
    FrameObservation,
    IdentityLabel,
    IdentityPoint,
    InvolvementEvent,
    LabeledTracklet,
    ScorePoint,
    TargetSpec,
    to_jsonable,
)
from .workflow_client import StubWorkflowClient, WorkflowClient


@dataclass
class PipelineResult:
    frames: list[FrameObservation]
    labeled: list[LabeledTracklet]
    identity: list[IdentityPoint]
    scores: list[ScorePoint]
    events: list[InvolvementEvent]
    clips: list[Clip]


def _workflow_ref(cfg: ReelcutConfig, client: WorkflowClient) -> str:
    """Cache-key component naming the inference configuration."""
    if isinstance(client, StubWorkflowClient):
        seed = client.seed if client.seed is not None else cfg.seed
        # target_jersey shapes stage-1 output (OCR read texts and the
        # confuser-number pool), so it must be part of the inference key.
        return f"stub:{seed}:{client.target_jersey}"
    return f"{cfg.workspace}/{cfg.workflow_id}:{cfg.model_id}"


def _target_fingerprint(spec: TargetSpec, cfg: ReelcutConfig) -> str:
    """Digest of the target spec + config: everything (beyond stage-1 output)
    that determines stages 2-4. A stored mismatch means the cached identity/
    scores/clips belong to a different kid or tuning and must be recomputed."""
    payload = {"spec": to_jsonable(spec), "cfg": asdict(cfg)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _health(stage_index: int, name: str, message: str) -> None:
    tag = f"[stage{stage_index} {name}]"
    print(f"{tag:<17} {message}")


def run_pipeline(
    video: Path,
    spec: TargetSpec,
    cfg: ReelcutConfig,
    paths: Paths,
    client: WorkflowClient,
    force_stage: int | None = None,
    debug_video: bool = False,
) -> PipelineResult:
    """Run all stages with caching.

    * cache key from cache.video_cache_key(video, cfg.sample_fps, workflow ref)
      — for the stub client the workflow ref is "stub:<seed>".
    * if force_stage is given, cache.invalidate_from(force_stage) first.
    * stage 1 payload name "observations"; 2 "identity" (labeled + timeline);
      3 "scores" (points + events); 4 "clips".
    * stage 4 also writes highlights.mp4 + highlights.json via clipcutter.
    * debug_video -> debugviz.render_debug_video into paths.debug_mp4.
    """
    key = video_cache_key(video, cfg.sample_fps, _workflow_ref(cfg, client))
    cache = StageCache(paths.cache_dir, key)
    if force_stage is not None:
        cache.invalidate_from(force_stage)

    # Stages 2-4 depend on WHO the target is (and identity/scoring/cut
    # config), which the video key cannot see. Re-running the same video for
    # a different kid must never silently reuse the previous kid's identity,
    # scores, or clips.
    fingerprint = _target_fingerprint(spec, cfg)
    if cache.has(2, "identity"):
        stored = (
            cache.load(2, "fingerprint") if cache.has(2, "fingerprint") else None
        )
        if stored != fingerprint:
            cache.invalidate_from(2)

    # ------------------------------------------------------------------ #
    # Stage 1 — inference
    # ------------------------------------------------------------------ #
    if not cache.has(1, "observations"):
        frames: list[FrameObservation] = list(client.run(video, cfg))
        cache.save(1, "observations", frames)
    else:
        frames = cache.load(1, "observations")

    n_frames = len(frames)
    ball_pct = (
        100.0 * sum(1 for f in frames if f.ball is not None) / n_frames
        if n_frames
        else 0.0
    )
    track_ids = {p.track_id for f in frames for p in f.players}
    _health(
        1,
        "infer",
        f"{n_frames} frames, ball in {ball_pct:.1f}% of frames, "
        f"{len(track_ids)} track ids",
    )

    # ------------------------------------------------------------------ #
    # Stage 2 — identity
    # ------------------------------------------------------------------ #
    if not cache.has(2, "identity"):
        tracklets = stitching.build_tracklets(frames)
        seed_track = stitching.find_seed_tracklet(tracklets, spec, frames)
        if seed_track is None:
            print(
                f"reelcut: the seed click (frame {spec.seed_frame_index}, box "
                f"{spec.seed_box.x:.0f},{spec.seed_box.y:.0f},"
                f"{spec.seed_box.w:.0f},{spec.seed_box.h:.0f}) matched no "
                "tracked player. Pick a frame where the kid is clearly "
                "visible and re-run with an accurate --target-frame / "
                "--target-box.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        labeled = stitching.label_tracklets(tracklets, spec, frames, cfg)
        labeled = stitching.chain_target_tracklets(labeled, cfg)
        identity = stitching.identity_timeline(labeled, frames, cfg)
        cache.save(2, "identity", {"labeled": labeled, "timeline": identity})
        cache.save(2, "fingerprint", fingerprint)
    else:
        payload = cache.load(2, "identity")
        labeled = payload["labeled"]
        identity = payload["timeline"]

    n_target = sum(1 for t in labeled if t.label is IdentityLabel.TARGET)
    n_not = sum(1 for t in labeled if t.label is IdentityLabel.NOT_TARGET)
    n_unknown = sum(1 for t in labeled if t.label is IdentityLabel.UNKNOWN)
    coverage = (
        100.0 * sum(1 for p in identity if p.confidence > 0) / len(identity)
        if identity
        else 0.0
    )
    _health(
        2,
        "identity",
        f"{len(labeled)} tracklets -> {n_target} target / {n_not} not_target "
        f"/ {n_unknown} unknown, coverage {coverage:.0f}%",
    )

    # ------------------------------------------------------------------ #
    # Stage 3 — involvement scoring
    # ------------------------------------------------------------------ #
    if not cache.has(3, "scores"):
        interp_frames = scoring.interpolate_ball(frames, cfg.ball_gap_interp_max_s)
        raw = scoring.score_timeline(identity, interp_frames, cfg)
        if cfg.fallback_enabled:
            opportunities = scoring.score_opportunities(interp_frames, cfg)
            raw = scoring.blend_scores(raw, opportunities, cfg.fallback_weight)
        scores = scoring.smooth_scores(raw, cfg.smooth_window_s)
        events = scoring.extract_events(scores, cfg.event_threshold, cfg.event_min_s)
        cache.save(3, "scores", {"points": scores, "events": events})
    else:
        payload = cache.load(3, "scores")
        scores = payload["points"]
        events = payload["events"]

    mean_score = sum(p.score for p in scores) / len(scores) if scores else 0.0
    _health(3, "score", f"{len(events)} events, mean score {mean_score:.2f}")

    # ------------------------------------------------------------------ #
    # Stage 4 — clip cutting
    # ------------------------------------------------------------------ #
    video_exists = video.exists()
    if video_exists:
        duration_s = ffmpeg.probe(video).duration_s
    else:
        duration_s = frames[-1].timestamp_s if frames else 0.0

    stage4_computed = False
    if not cache.has(4, "clips"):
        clips: list[Clip] = clipcutter.plan_clips(
            events, scores, identity, duration_s, cfg
        )
        cache.save(4, "clips", clips)
        stage4_computed = True
    else:
        clips = cache.load(4, "clips")

    clipcutter.write_highlights_json(clips, paths.highlights_json, video, cfg)
    if not video_exists:
        print(
            f"reelcut: source video {video} not found; skipping highlights.mp4 "
            "(clip plan written to highlights.json)",
            file=sys.stderr,
        )
    elif not clips:
        # Zero events surviving thresholding is a normal outcome (kid benched
        # or uninvolved in the recorded stretch), not an error.
        print(
            "reelcut: no involvement events found; skipping highlights.mp4 "
            "(empty clip plan written to highlights.json)",
            file=sys.stderr,
        )
    elif stage4_computed or not paths.highlights_mp4.exists():
        work_dir = cache.dir / "segments"
        work_dir.mkdir(parents=True, exist_ok=True)
        clipcutter.cut_reel(clips, video, paths.highlights_mp4, work_dir)

    total_s = sum(c.end_s - c.start_s for c in clips)
    total_pct = 100.0 * total_s / duration_s if duration_s > 0 else 0.0
    _health(
        4,
        "cut",
        f"{len(clips)} clips, {total_s:.1f}s total ({total_pct:.1f}% of source)",
    )

    # ------------------------------------------------------------------ #
    # Optional debug render
    # ------------------------------------------------------------------ #
    if debug_video:
        if video_exists:
            from . import debugviz  # deferred: pulls in cv2

            debugviz.render_debug_video(
                video, paths.debug_mp4, frames, identity, scores, cfg
            )
        else:
            print(
                f"reelcut: source video {video} not found; skipping debug video",
                file=sys.stderr,
            )

    return PipelineResult(
        frames=frames,
        labeled=labeled,
        identity=identity,
        scores=scores,
        events=events,
        clips=clips,
    )
