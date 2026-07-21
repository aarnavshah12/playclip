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
import os
import sys
from dataclasses import asdict, dataclass, replace
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

    # frame hygiene (applied after load so cached runs behave identically):
    # duplicate boxes on one kid poison reads, votes, and rendering alike.
    frames = stitching.dedupe_player_boxes(frames, cfg.player_dedupe_iou)

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
        # Stage 1.5: read-until-bound jersey numbers (skipped for the stub,
        # which synthesizes its own reads, and when no video/model/key exists).
        number_reads: list[tuple[int, object]] = []
        number_map: dict[int, str] = {}
        if (
            cfg.digit_model_id
            and video.exists()
            and not isinstance(client, StubWorkflowClient)
            and os.environ.get("ROBOFLOW_API_KEY")
        ):
            from . import numbers

            reader = numbers.make_digit_reader(
                cfg.digit_model_id,
                os.environ["ROBOFLOW_API_KEY"],
                cfg.digit_min_conf,
            )
            enriched, number_map = numbers.bind_numbers(frames, video, cfg, reader)
            number_reads = [
                (f.frame_index, r)
                for f, orig in zip(enriched, frames)
                for r in f.ocr[len(orig.ocr):]
            ]
            frames = enriched
            n_splits = 0
            if cfg.number_flip_splits and cfg.number_continuous:
                frames, n_splits = numbers.split_on_number_flips(frames, cfg)
            n_confirmed = sum(1 for v in number_map.values() if not v.endswith("?"))
            _health(2, "numbers",
                    f"{n_confirmed} confirmed + "
                    f"{len(number_map) - n_confirmed} provisional numbers "
                    f"({len(number_reads)} fresh reads, "
                    f"{n_splits} handover splits)")
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
        if (
            cfg.reid_enabled
            and video.exists()
            and not isinstance(client, StubWorkflowClient)
            and os.environ.get("ROBOFLOW_API_KEY")
        ):
            from . import reid

            embedder = reid.make_clip_embedder(os.environ["ROBOFLOW_API_KEY"])
            embeds = reid.track_embeddings(
                tracklets, seed_track, video, embedder, cfg
            )
            negatives = reid.negative_track_ids(
                tracklets, spec.jersey, cfg.reid_neg_min_reads
            )
            scores = reid.contrastive_scores(embeds, seed_track, negatives)
            before = {lt.tracklet.track_id: lt.label for lt in labeled}
            labeled = reid.apply_reid(labeled, scores, seed_track, cfg)
            n_changed = sum(
                1 for lt in labeled
                if before[lt.tracklet.track_id] is not lt.label
            )
            _health(2, "reid",
                    f"{len(embeds)} tracks embedded vs {len(negatives)} "
                    f"negative banks, {n_changed} labels changed")
        labeled = stitching.chain_target_tracklets(labeled, cfg)
        identity = stitching.identity_timeline(labeled, frames, cfg)
        cache.save(2, "identity", {
            "labeled": labeled,
            "timeline": identity,
            "number_reads": [[fi, r] for fi, r in number_reads],
            "numbers_map": number_map,
        })
        cache.save(2, "fingerprint", fingerprint)
    else:
        payload = cache.load(2, "identity")
        labeled = payload["labeled"]
        identity = payload["timeline"]
        number_map = {
            int(k): v for k, v in (payload.get("numbers_map") or {}).items()
        }
        stored_reads = payload.get("number_reads") or []
        if stored_reads:
            by_index: dict[int, list] = {}
            for fi, r in stored_reads:
                by_index.setdefault(int(fi), []).append(r)
            frames = [
                replace(f, ocr=f.ocr + tuple(by_index[f.frame_index]))
                if f.frame_index in by_index else f
                for f in frames
            ]
            if cfg.number_flip_splits and cfg.number_continuous:
                # deterministic replay of the same splits the computing run
                # applied, so cached and fresh runs see identical frames
                from . import numbers

                frames, _ = numbers.split_on_number_flips(frames, cfg)

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
        interp_frames = scoring.persist_goal_boxes(interp_frames, cfg.goal_box_hold_s)
        raw = scoring.score_timeline(identity, interp_frames, cfg)
        if cfg.fallback_enabled:
            opportunities = scoring.score_opportunities(interp_frames, cfg)
            raw = scoring.blend_scores(raw, opportunities, cfg.fallback_weight)
        scores = scoring.smooth_scores(raw, cfg.smooth_window_s)
        threshold = scoring.adaptive_threshold(
            scores, cfg.event_threshold,
            cfg.target_reel_fraction, cfg.max_reel_fraction,
        )
        if threshold > cfg.event_threshold:
            print(f"[stage3] scores saturate timeline; threshold raised "
                  f"{cfg.event_threshold:.2f} -> {threshold:.2f}")
        events = scoring.extract_events(scores, threshold, cfg.event_min_s)
        if threshold > cfg.event_threshold:
            # A high-action goal-mouth moment never loses to the reel budget:
            # re-extract at the base threshold and keep events the raised bar
            # dropped IF the RAW score inside them spikes at the goal mouth.
            # A shot is a transient — the 2s smoothing that stabilizes event
            # boundaries also dilutes exactly this peak, so the smoothed
            # series must not be the judge of it. Action gating in
            # score_opportunities keeps static loitering out of this lane.
            def has_goal_transient(e: InvolvementEvent) -> bool:
                return any(
                    p.score >= cfg.goal_transient_score
                    and "goal_mouth" in p.tags
                    and e.start_s <= p.timestamp_s <= e.end_s
                    for p in raw
                )

            goal_events = [
                e for e in scoring.extract_events(
                    scores, cfg.event_threshold, cfg.event_min_s
                )
                if has_goal_transient(e)
            ]
            seen = {(e.start_s, e.end_s) for e in events}
            events = sorted(
                events + [e for e in goal_events
                          if (e.start_s, e.end_s) not in seen],
                key=lambda e: e.start_s,
            )
        if cfg.goal_only_reel:
            goal_evts = scoring.goal_transient_events(
                raw, cfg.goal_event_lead_s, cfg.goal_event_tail_s,
                cfg.goal_transient_score,
            )
            if goal_evts:
                n_found = len(goal_evts)
                if cfg.max_goal_clips > 0 and len(goal_evts) > cfg.max_goal_clips:
                    # Rank by cluster DURATION, then peak: after a real goal
                    # the ball STAYS at the net (celebration, fetching it
                    # out), so goal clusters run long; saved shots bounce
                    # away and cluster short. Peaks saturate (a great save
                    # and a goal both max the score) and cannot discriminate.
                    goal_evts = sorted(
                        sorted(goal_evts,
                               key=lambda e: (e.end_s - e.start_s,
                                              e.peak_score),
                               reverse=True)[: cfg.max_goal_clips],
                        key=lambda e: e.start_s,
                    )
                _health(3, "goals",
                        f"goal-only reel: {len(goal_evts)} of {n_found} "
                        f"goal moments kept (from raw transients)")
                events = goal_evts
            else:
                # no goals in this footage: an empty reel helps nobody, so
                # the involvement events stand
                _health(3, "goals",
                        f"goal-only reel: no goal transients found; keeping "
                        f"{len(events)} involvement events")
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

    def _clip_stem(i: int, c: Clip) -> str:
        tag = "_".join(c.reasons[:2]) if c.reasons else "clip"
        return f"clip{i + 1}_{c.start_s:.0f}s-{c.end_s:.0f}s_{tag}"

    def _export_clip_files(
        source: Path, work_dir: Path, suffix: str, transcode: bool = False
    ) -> None:
        """Per-clip files with human names in <out>/clips/ (parents never dig
        in the cache for seg_XXX.mp4 again). transcode=True re-encodes to
        h264 — needed for the annotated cuts, whose mp4v source browsers
        refuse to play."""
        import shutil
        import subprocess

        clips_dir = paths.out_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        expected = [work_dir / f"seg_{i:03d}.mp4" for i in range(len(clips))]
        if not all(p.exists() for p in expected):
            clipcutter.cut_reel(
                clips, source,
                work_dir.with_suffix(".reel.mp4"), work_dir,
            )
        for old in clips_dir.glob(f"*{suffix}.mp4"):
            old.unlink()
        for i, (c, seg) in enumerate(zip(clips, expected)):
            if not seg.exists():
                continue
            dest = clips_dir / f"{_clip_stem(i, c)}{suffix}.mp4"
            if transcode:
                subprocess.run(
                    [ffmpeg.ffmpeg_exe(), "-y", "-loglevel", "error",
                     "-i", str(seg), "-c:v", "libx264", "-crf", "20",
                     "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                     str(dest)],
                    check=True,
                )
            else:
                shutil.copy(seg, dest)

    if video_exists and clips:
        _export_clip_files(video, cache.dir / "segments", "")

    # ------------------------------------------------------------------ #
    # Optional debug render + annotated deliverables
    # ------------------------------------------------------------------ #
    if debug_video:
        if video_exists:
            import subprocess

            from . import debugviz  # deferred: pulls in cv2

            debugviz.render_debug_video(
                video, paths.debug_mp4, frames, identity, scores, cfg,
            )
            if clips:
                clipcutter.cut_reel(
                    clips, paths.debug_mp4,
                    paths.out_dir / "highlights_annotated.mp4",
                    cache.dir / "segments_annotated",
                )
                _export_clip_files(
                    paths.debug_mp4, cache.dir / "segments_annotated",
                    "_annotated", transcode=True,
                )
            # browser/share-friendly h264 of the full annotated video (the
            # raw debug.mp4 is mp4v, which browsers refuse to play)
            subprocess.run(
                [ffmpeg.ffmpeg_exe(), "-y", "-loglevel", "error",
                 "-i", str(paths.debug_mp4),
                 "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart",
                 str(paths.out_dir / "annotated_full.mp4")],
                check=True,
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
