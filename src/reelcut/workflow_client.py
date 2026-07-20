"""Stage-1 clients: turn a video into per-frame FrameObservations.

Two implementations behind one Protocol:

* StubWorkflowClient — deterministic synthetic "game" (seeded RNG); lets every
  downstream stage run end-to-end with no Roboflow account, network, or GPU.
* RoboflowWorkflowClient — the real thing: `inference.InferencePipeline`
  running our saved Roboflow workflow (detection -> filters -> BoT-SORT ->
  class lock -> stabilizer -> velocity -> OCR branch) at cfg.sample_fps.

The client is also where torso HSV histograms are computed (locally, from the
frame + player bbox — cheap numpy, no extra model), because identity stitching
needs an appearance descriptor and the frames are already in hand here.
"""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import cv2
import numpy as np

from .config import ReelcutConfig
from .ffmpeg import probe
from .types import BallObs, BBox, FrameObservation, OcrRead, PlayerObs

logger = logging.getLogger(__name__)


def _prepare_inference_env() -> None:
    """Must run BEFORE the first `import inference`.

    * With max_fps on a video FILE, inference's default is to throttle
      wall-clock to max_fps while still processing EVERY frame (measured:
      45 s clip -> 280 s). This flag makes it drop frames instead, which is
      what sampling means for post-hoc batch work.
    * torch 2.13 removed torch.mps.current_device, but inference_models'
      ONNX preprocess enters a torch.cuda.stream() context whose generalized
      accelerator lookup still calls it on Apple Silicon; restore it.
    """
    import os

    os.environ.setdefault("ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING", "True")
    try:
        import torch
    except ImportError:
        return
    if not hasattr(torch.mps, "current_device"):
        torch.mps.current_device = lambda: 0


# Back-compat alias (scripts import the old name).
_shim_torch_mps = _prepare_inference_env

# Histogram binning shared with stitching.team_color_reference:
# HSV, H in 12 bins x S in 4 bins (V dropped for lighting robustness),
# flattened to 48 floats, L1-normalized.
HIST_H_BINS = 12
HIST_S_BINS = 4

_ZERO_HIST: tuple[float, ...] = tuple(0.0 for _ in range(HIST_H_BINS * HIST_S_BINS))


def torso_histogram(frame_bgr: np.ndarray, bbox_xywh: tuple[float, float, float, float]) -> tuple[float, ...]:
    """HSV histogram of the upper-torso region of a player bbox.

    Torso region: central 60% width, rows 15%..50% of bbox height (skips head
    and legs). Returns the flattened normalized histogram; all-zeros if the
    region is degenerate/out of frame.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return _ZERO_HIST
    frame_h, frame_w = frame_bgr.shape[:2]
    x, y, w, h = bbox_xywh
    x0 = max(int(round(x + 0.20 * w)), 0)
    x1 = min(int(round(x + 0.80 * w)), frame_w)
    y0 = max(int(round(y + 0.15 * h)), 0)
    y1 = min(int(round(y + 0.50 * h)), frame_h)
    if x1 <= x0 or y1 <= y0:
        return _ZERO_HIST
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist, _, _ = np.histogram2d(
        hsv[..., 0].ravel().astype(np.float64),
        hsv[..., 1].ravel().astype(np.float64),
        bins=(HIST_H_BINS, HIST_S_BINS),
        range=((0.0, 180.0), (0.0, 256.0)),
    )
    total = float(hist.sum())
    if total <= 0.0:
        return _ZERO_HIST
    return tuple(float(v) for v in (hist / total).ravel())


class WorkflowClient(Protocol):
    def run(self, video: Path, cfg: ReelcutConfig) -> Iterator[FrameObservation]:
        """Yield FrameObservations in frame order at ~cfg.sample_fps."""
        ...


class StubWorkflowClient:
    """Synthetic 22-player + ball game, deterministic per cfg.seed.

    * If ``video`` exists, duration/size come from ffmpeg.probe; otherwise a
      120 s / 1280x720 game is synthesized (so unit tests need no video file).
    * Players do smooth random walks (bounded to the frame); one designated
      "target" player (track_id 7) wears the jersey number the stub is
      configured to emit in OCR reads (~30% of frames, conf 0.5-0.95).
    * The ball bounces between random players, occasionally missing
      (None ~25% of frames) to exercise interpolation.
    * Tracks fragment (id shifts) every ~20 s to exercise stitching.
    """

    _N_PLAYERS = 22
    _TARGET_SLOT = 6            # initial track_id 7; slots 0..10 are the blue team
    _SOURCE_FPS = 30.0          # notional source frame rate for frame_index

    def __init__(self, seed: int | None = None, target_jersey: str = "10") -> None:
        self.seed = seed
        self.target_jersey = target_jersey

    def run(self, video: Path, cfg: ReelcutConfig) -> Iterator[FrameObservation]:
        rng = np.random.default_rng(cfg.seed if self.seed is None else self.seed)
        if video.exists():
            meta = probe(video)
            duration_s, frame_w, frame_h = meta.duration_s, meta.width, meta.height
        else:
            duration_s, frame_w, frame_h = 120.0, 1280, 720
        yield from self._simulate(rng, cfg, duration_s, frame_w, frame_h)

    # ------------------------------------------------------------------ #
    # simulation internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _kit_histogram(rng: np.random.Generator, team: str) -> tuple[float, ...]:
        """Synthetic 48-bin torso hist with mass in the team's hue range
        (config.TEAM_COLORS: blue H 86-125, red H 0-10 & 170-179)."""
        hist = np.zeros((HIST_H_BINS, HIST_S_BINS))
        h_weights = (
            {5: 0.15, 6: 0.35, 7: 0.35, 8: 0.15}   # bins spanning H 75..135
            if team == "blue"
            else {0: 0.5, 11: 0.5}                  # bins spanning H 0..15, 165..180
        )
        for h_bin, weight in h_weights.items():
            hist[h_bin, 3] = weight * 0.7          # mostly saturated
            hist[h_bin, 2] = weight * 0.3
        hist += rng.uniform(0.0, 0.008, hist.shape)  # per-player grass/skin noise
        hist /= hist.sum()
        return tuple(float(v) for v in hist.ravel())

    def _simulate(
        self,
        rng: np.random.Generator,
        cfg: ReelcutConfig,
        duration_s: float,
        frame_w: int,
        frame_h: int,
    ) -> Iterator[FrameObservation]:
        n = self._N_PLAYERS
        dt = 1.0 / cfg.sample_fps
        n_samples = max(1, int(round(duration_s * cfg.sample_fps)))

        heights = rng.uniform(70.0, 90.0, n)
        widths = heights * 0.45
        lo = np.column_stack([widths / 2.0, heights / 2.0])
        hi = np.column_stack([frame_w - widths / 2.0, frame_h - heights / 2.0])
        pos = np.column_stack([
            rng.uniform(frame_w * 0.08, frame_w * 0.92, n),
            rng.uniform(frame_h * 0.15, frame_h * 0.92, n),
        ])
        vel = rng.normal(0.0, 40.0, (n, 2))

        # slots 0..10 (target + 10 teammates) blue, slots 11..21 red
        hists = [
            self._kit_histogram(rng, "blue" if s <= 10 else "red") for s in range(n)
        ]

        # 2-3 non-target players occasionally OCR as a DIFFERENT number
        other_slots = [s for s in range(n) if s != self._TARGET_SLOT]
        other_numbers = [str(k) for k in range(2, 40) if str(k) != self.target_jersey]
        confusers = [
            (other_slots[int(s)], other_numbers[int(rng.integers(len(other_numbers)))])
            for s in rng.choice(len(other_slots), size=int(rng.integers(2, 4)), replace=False)
        ]

        ids = list(range(1, n + 1))
        next_id = n + 1
        frag_every = max(1, int(round(20.0 * cfg.sample_fps)))

        carrier = int(rng.integers(n))
        carrier_left = max(1, int(round(rng.uniform(5.0, 11.0) * cfg.sample_fps)))

        for i in range(n_samples):
            t = i * dt

            # ~every 20 s, 30% of tracks get fresh ids (tracker fragmentation)
            if i > 0 and i % frag_every == 0:
                for s in rng.choice(n, size=max(1, round(0.3 * n)), replace=False):
                    ids[int(s)] = next_id
                    next_id += 1

            # momentum random walk, bounced off frame edges
            vel = vel * 0.90 + rng.normal(0.0, 35.0, (n, 2))
            mag = np.hypot(vel[:, 0], vel[:, 1])
            fast = mag > 280.0
            vel[fast] *= (280.0 / mag[fast])[:, None]
            prev = pos
            pos = pos + vel * dt
            for axis in (0, 1):
                below = pos[:, axis] < lo[:, axis]
                above = pos[:, axis] > hi[:, axis]
                pos[below, axis] = 2.0 * lo[below, axis] - pos[below, axis]
                pos[above, axis] = 2.0 * hi[above, axis] - pos[above, axis]
                vel[below | above, axis] *= -1.0
            pos = np.clip(pos, lo, hi)
            disp = (pos - prev) / dt          # actual px/s of the walk
            speeds = np.hypot(disp[:, 0], disp[:, 1])
            confs = rng.uniform(0.6, 0.95, n)

            # ball: sticks to a randomly re-picked carrier, ~25% missed frames
            carrier_left -= 1
            if carrier_left <= 0:
                carrier = int(rng.integers(n))
                carrier_left = max(1, int(round(rng.uniform(5.0, 11.0) * cfg.sample_fps)))
            ball_off = rng.normal(0.0, 10.0, 2)
            ball: BallObs | None = None
            if rng.random() >= 0.25:
                bx = float(np.clip(pos[carrier, 0] + ball_off[0], 0.0, frame_w))
                by = float(np.clip(
                    pos[carrier, 1] + heights[carrier] * 0.35 + ball_off[1],
                    0.0, frame_h,
                ))
                ball = BallObs(
                    bbox=BBox(bx - 7.0, by - 7.0, 14.0, 14.0),
                    confidence=float(rng.uniform(0.4, 0.9)),
                )

            # jersey OCR: target reads ~30% of frames, confusers occasionally
            ocr: list[OcrRead] = []
            if rng.random() < 0.30:
                ocr.append(OcrRead(
                    track_id=ids[self._TARGET_SLOT],
                    text=self.target_jersey,
                    confidence=float(rng.uniform(0.5, 0.95)),
                ))
            for slot, text in confusers:
                if rng.random() < 0.08:
                    ocr.append(OcrRead(
                        track_id=ids[slot],
                        text=text,
                        confidence=float(rng.uniform(0.5, 0.95)),
                    ))

            players = tuple(
                PlayerObs(
                    track_id=ids[s],
                    bbox=BBox(
                        float(pos[s, 0] - widths[s] / 2.0),
                        float(pos[s, 1] - heights[s] / 2.0),
                        float(widths[s]),
                        float(heights[s]),
                    ),
                    class_name="player",
                    confidence=float(confs[s]),
                    speed=float(speeds[s]),
                    velocity=(float(disp[s, 0]), float(disp[s, 1])),
                    torso_hsv=hists[s],
                )
                for s in range(n)
            )
            yield FrameObservation(
                frame_index=int(round(t * self._SOURCE_FPS)),
                timestamp_s=float(t),
                frame_w=int(frame_w),
                frame_h=int(frame_h),
                players=players,
                ball=ball,
                ocr=tuple(ocr),
                goal_boxes=(
                    # static goal mouths at both touchlines (fallback fodder)
                    BBox(0.0, frame_h * 0.40, frame_w * 0.03, frame_h * 0.20),
                    BBox(frame_w * 0.97, frame_h * 0.40, frame_w * 0.03, frame_h * 0.20),
                ),
            )


# --------------------------------------------------------------------------- #
# Real client: InferencePipeline over the saved workflow
# --------------------------------------------------------------------------- #

def _canonical_class(name: str | None, cfg: ReelcutConfig) -> str:
    """Map raw workflow class names onto 'player' | 'goalkeeper' | 'referee'."""
    low = (name or "").strip().lower()
    if low in {c.lower() for c in cfg.referee_classes}:
        return "referee"
    if low in {c.lower() for c in cfg.player_classes} and low.startswith("goal"):
        return "goalkeeper"
    return "player"


def _detections_len(det: Any) -> int:
    """Row count of an sv.Detections-like object; 0 for None/malformed."""
    xyxy = getattr(det, "xyxy", None)
    if xyxy is None:
        return 0
    try:
        return int(len(xyxy))
    except TypeError:
        return 0


def _int_at(arr: Any, k: int) -> int | None:
    if arr is None or k >= len(arr) or arr[k] is None:
        return None
    try:
        v = float(arr[k])
    except (TypeError, ValueError):
        return None
    return int(v) if np.isfinite(v) else None


def _float_at(arr: Any, k: int) -> float | None:
    if arr is None or k >= len(arr) or arr[k] is None:
        return None
    try:
        v = float(arr[k])
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _pair_at(arr: Any, k: int) -> tuple[float, float] | None:
    if arr is None or k >= len(arr) or arr[k] is None:
        return None
    try:
        v = np.asarray(arr[k], dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if v.size < 2 or not np.all(np.isfinite(v[:2])):
        return None
    return float(v[0]), float(v[1])


def _observation_from_prediction(
    predictions: Mapping[str, Any],
    frame_bgr: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    cfg: ReelcutConfig,
) -> FrameObservation:
    """Convert one workflow result dict + decoded frame into a FrameObservation.

    Pure and defensive: missing/None outputs act as empty detections, jersey
    texts shorter than the player list are treated as no-reads, untracked or
    sub-min-height boxes are dropped. Torso histograms are computed here from
    the frame pixels.
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    players: list[PlayerObs] = []
    ocr: list[OcrRead] = []

    det = predictions.get("tracked_players")
    n = _detections_len(det)
    if n:
        xyxy = np.asarray(det.xyxy, dtype=float).reshape(-1, 4)
        tracker_ids = getattr(det, "tracker_id", None)
        confidences = getattr(det, "confidence", None)
        data = getattr(det, "data", None) or {}
        class_names = data.get("class_name")
        velocities = data.get("velocity")
        speeds = data.get("speed")
        texts: Sequence[Any] = predictions.get("jersey_texts") or ()
        min_box_h = cfg.min_player_box_h_frac * frame_h
        for k in range(n):
            track_id = _int_at(tracker_ids, k)
            if track_id is None:
                continue                        # untracked box: useless downstream
            x1, y1, x2, y2 = (float(v) for v in xyxy[k])
            if (y2 - y1) < min_box_h:
                continue
            bbox = BBox.from_xyxy(x1, y1, x2, y2)
            raw_class = None
            if class_names is not None and k < len(class_names):
                raw_class = class_names[k]
            players.append(PlayerObs(
                track_id=track_id,
                bbox=bbox,
                class_name=_canonical_class(
                    None if raw_class is None else str(raw_class), cfg
                ),
                confidence=_float_at(confidences, k) or 0.0,
                speed=_float_at(speeds, k),
                velocity=_pair_at(velocities, k),
                torso_hsv=torso_histogram(frame_bgr, (bbox.x, bbox.y, bbox.w, bbox.h)),
            ))
            text = texts[k] if k < len(texts) else None
            if text:  # workflow OCR emits text only when it has a read
                ocr.append(OcrRead(track_id=track_id, text=str(text), confidence=1.0))

    ball: BallObs | None = None
    ball_det = predictions.get("ball")
    n_ball = _detections_len(ball_det)
    if n_ball:
        ball_xyxy = np.asarray(ball_det.xyxy, dtype=float).reshape(-1, 4)
        ball_conf = getattr(ball_det, "confidence", None)
        k = 0
        if ball_conf is not None and len(ball_conf) == n_ball:
            k = int(np.argmax(np.asarray(ball_conf, dtype=float)))
        ball = BallObs(
            bbox=BBox.from_xyxy(*(float(v) for v in ball_xyxy[k])),
            confidence=_float_at(ball_conf, k) or 0.0,
        )

    goal_boxes: list[BBox] = []
    goal_det = predictions.get("goal_detections")
    for k in range(_detections_len(goal_det)):
        gx = np.asarray(goal_det.xyxy, dtype=float).reshape(-1, 4)[k]
        goal_boxes.append(BBox.from_xyxy(*(float(v) for v in gx)))

    return FrameObservation(
        frame_index=int(frame_index),
        timestamp_s=float(timestamp_s),
        frame_w=int(frame_w),
        frame_h=int(frame_h),
        players=tuple(players),
        ball=ball,
        ocr=tuple(ocr),
        goal_boxes=tuple(goal_boxes),
    )


class RoboflowWorkflowClient:
    """InferencePipeline over the saved workspace workflow.

    Contract with the workflow outputs (see workflows/reelcut_tracking.json):
      * "tracked_players": sv.Detections with tracker_id, class, confidence,
        and data keys "velocity"/"speed" from the Velocity block
      * "ball": sv.Detections (0 or 1 efter highest-confidence filtering)
      * "jersey_texts": list[str] aligned with tracked_players order (may be
        shorter/None-padded; treat missing as no read)
    Frames are sampled by InferencePipeline at max_fps=cfg.sample_fps; each
    emitted FrameObservation carries the SOURCE frame index and timestamp.
    """

    def __init__(
        self,
        api_key: str,
        workspace: str,
        workflow_id: str,
        api_url: str | None = None,
        image_input_name: str = "image",
    ) -> None:
        self.api_key = api_key
        self.workspace = workspace
        self.workflow_id = workflow_id
        self.api_url = api_url
        self.image_input_name = image_input_name

    def run(self, video: Path, cfg: ReelcutConfig) -> Iterator[FrameObservation]:
        """Run InferencePipeline.init_with_workflow, collect results via an
        internal queue sink, convert each workflow result + video frame into a
        FrameObservation (computing torso_histogram locally per player), and
        yield in order. Blocks until the video is exhausted."""
        # Heavy import kept local so importing this module stays cheap/offline.
        _prepare_inference_env()
        from inference import InferencePipeline

        meta = probe(video)
        source_fps = meta.fps if meta.fps > 0 else 30.0
        # Unbounded on purpose: the sink must never block the dispatch thread.
        results: queue.Queue[FrameObservation | None] = queue.Queue()

        def sink(predictions: Any, video_frame: Any) -> None:
            # Single source + SinkMode.ADAPTIVE => scalar (dict, VideoFrame);
            # tolerate the batch (list, list) shape anyway.
            if isinstance(predictions, list):
                pairs = zip(predictions, video_frame)
            else:
                pairs = [(predictions, video_frame)]
            for preds, frame in pairs:
                if preds is None or frame is None:
                    continue
                try:
                    # VideoFrame.frame_id is 1-based (inference increments its
                    # counter before emitting), but reelcut frame indices and
                    # cv2 seeks are 0-based: subtract 1 so the first sample
                    # lands at frame 0 / t=0 instead of one frame late.
                    source_index = int(frame.frame_id) - 1
                    results.put(_observation_from_prediction(
                        predictions=preds,
                        frame_bgr=frame.image,
                        frame_index=source_index,
                        timestamp_s=float(source_index) / source_fps,
                        cfg=cfg,
                    ))
                except Exception:  # never kill the dispatch thread on one frame
                    logger.exception("dropping malformed workflow result")

        # NOTE: init_with_workflow in the installed inference version has no
        # api_url parameter (the workflow runs in-process); self.api_url is
        # kept for a future remote-execution path.
        pipeline = InferencePipeline.init_with_workflow(
            video_reference=str(video),
            workspace_name=self.workspace,
            workflow_id=self.workflow_id,
            api_key=self.api_key,
            image_input_name=self.image_input_name,
            # The saved workflow declares a `model_id` WorkflowParameter;
            # passing cfg.model_id here makes --model-id actually take effect
            # (and keeps the cache key, which bakes model_id in, honest).
            workflows_parameters={"model_id": cfg.model_id},
            on_prediction=sink,
            max_fps=cfg.sample_fps,
        )
        pipeline.start(use_main_thread=False)

        def wait_and_seal() -> None:
            try:
                pipeline.join()
            finally:
                results.put(None)  # sentinel: no more observations

        waiter = threading.Thread(
            target=wait_and_seal, name="reelcut-pipeline-join", daemon=True
        )
        waiter.start()
        try:
            while True:
                obs = results.get()
                if obs is None:
                    break
                yield obs
        finally:
            if waiter.is_alive():  # consumer bailed early: stop the pipeline
                pipeline.terminate()
                waiter.join()
