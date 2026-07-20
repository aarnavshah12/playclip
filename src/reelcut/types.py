"""Core datatypes shared across all reelcut stages.

Everything here is JSON-serializable via ``to_jsonable`` / ``from_jsonable``
so stage outputs can be cached to disk and reloaded without re-inference.
Units: pixels for geometry, seconds for time, unless a name says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box, top-left origin, pixel units."""

    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def iou(self, other: "BBox") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def center_dist(self, other: "BBox") -> float:
        return ((self.cx - other.cx) ** 2 + (self.cy - other.cy) ** 2) ** 0.5

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> "BBox":
        return BBox(x1, y1, x2 - x1, y2 - y1)


# --------------------------------------------------------------------------- #
# Stage 1 output: per-frame observations from the Roboflow workflow
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class PlayerObs:
    """One tracked player/referee detection in one frame."""

    track_id: int
    bbox: BBox
    class_name: str          # "player" | "goalkeeper" | "referee" (post class-lock)
    confidence: float
    speed: float | None = None                      # px/s, smoothed (Velocity block)
    velocity: tuple[float, float] | None = None     # (vx, vy) px/s
    torso_hsv: tuple[float, ...] | None = None      # normalized HSV hist of torso region


@dataclass(frozen=True, slots=True)
class BallObs:
    bbox: BBox
    confidence: float
    interpolated: bool = False   # True when filled by local gap interpolation


@dataclass(frozen=True, slots=True)
class OcrRead:
    """Jersey OCR read for one player crop in one frame."""

    track_id: int
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """Everything the workflow (plus the local sink) tells us about one frame."""

    frame_index: int             # index in the SOURCE video
    timestamp_s: float           # seconds into the source video
    frame_w: int
    frame_h: int
    players: tuple[PlayerObs, ...] = ()
    ball: BallObs | None = None
    ocr: tuple[OcrRead, ...] = ()
    goal_boxes: tuple[BBox, ...] = ()   # detected goal frames (class `goal`)


# --------------------------------------------------------------------------- #
# Stage 2: tracklets and identity
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class TrackletPoint:
    frame_index: int
    timestamp_s: float
    bbox: BBox
    speed: float | None = None
    torso_hsv: tuple[float, ...] | None = None


@dataclass(slots=True)
class Tracklet:
    """A contiguous single-id track produced by the workflow tracker."""

    track_id: int
    points: list[TrackletPoint] = field(default_factory=list)
    class_name: str = "player"
    ocr_reads: list[OcrRead] = field(default_factory=list)

    @property
    def start_s(self) -> float:
        return self.points[0].timestamp_s

    @property
    def end_s(self) -> float:
        return self.points[-1].timestamp_s

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class IdentityLabel(str, Enum):
    TARGET = "target"
    NOT_TARGET = "not_target"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class LabeledTracklet:
    tracklet: Tracklet
    label: IdentityLabel
    confidence: float                       # 0..1, meaningful for TARGET
    evidence: dict[str, float] = field(default_factory=dict)
    # evidence keys: "seed", "ocr_pos", "ocr_neg", "color", "kinematic"


@dataclass(frozen=True, slots=True)
class IdentityPoint:
    """Target presence at one sampled timestamp."""

    timestamp_s: float
    confidence: float                       # 0 when target absent/unknown
    bbox: BBox | None = None
    track_id: int | None = None
    speed: float | None = None


# --------------------------------------------------------------------------- #
# Stage 3: involvement
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class ScorePoint:
    timestamp_s: float
    score: float                            # 0..1, already identity-weighted
    tags: tuple[str, ...] = ()              # "near_ball" | "possession" | "touch" | "sprint"


@dataclass(frozen=True, slots=True)
class InvolvementEvent:
    start_s: float
    end_s: float
    peak_score: float
    mean_score: float
    tags: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Stage 4: clips
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class Clip:
    """A cut decision against the SOURCE video timeline."""

    start_s: float
    end_s: float
    score: float
    reasons: tuple[str, ...] = ()
    confidence: float = 1.0                 # identity confidence over the clip


# --------------------------------------------------------------------------- #
# User input
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class TargetSpec:
    """How the user identified their kid."""

    jersey: str                             # e.g. "10" (string: leading zeros happen)
    team_color: str                         # e.g. "blue" — key into config.TEAM_COLORS
    seed_frame_index: int                   # frame of the user's click
    seed_box: BBox                          # clicked box around the kid


class Stage(IntEnum):
    INFER = 1
    IDENTITY = 2
    SCORE = 3
    CUT = 4


# --------------------------------------------------------------------------- #
# JSON (de)serialization — hand-rolled, no dependencies
# --------------------------------------------------------------------------- #

def to_jsonable(obj: Any) -> Any:
    """Recursively convert reelcut dataclasses to JSON-safe structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {
            "__type__": type(obj).__name__,
            **{
                name: to_jsonable(getattr(obj, name))
                for name in obj.__dataclass_fields__
            },
        }
    raise TypeError(f"cannot serialize {type(obj)!r}")


_TYPES: dict[str, type] = {}


def _register_types() -> None:
    for cls in (
        BBox, PlayerObs, BallObs, OcrRead, FrameObservation,
        TrackletPoint, Tracklet, LabeledTracklet, IdentityPoint,
        ScorePoint, InvolvementEvent, Clip, TargetSpec,
    ):
        _TYPES[cls.__name__] = cls


_register_types()

_TUPLE_FIELDS = {
    ("PlayerObs", "velocity"), ("PlayerObs", "torso_hsv"),
    ("TrackletPoint", "torso_hsv"),
    ("FrameObservation", "players"), ("FrameObservation", "ocr"),
    ("FrameObservation", "goal_boxes"),
    ("ScorePoint", "tags"), ("InvolvementEvent", "tags"), ("Clip", "reasons"),
}


def from_jsonable(data: Any) -> Any:
    """Inverse of :func:`to_jsonable`."""
    if isinstance(data, list):
        return [from_jsonable(d) for d in data]
    if not isinstance(data, dict):
        return data
    if "__type__" not in data:
        return {k: from_jsonable(v) for k, v in data.items()}
    tname = data["__type__"]
    cls = _TYPES[tname]
    kwargs: dict[str, Any] = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        v = from_jsonable(v)
        if k == "label" and tname == "LabeledTracklet":
            v = IdentityLabel(v)
        if isinstance(v, list) and (tname, k) in _TUPLE_FIELDS:
            v = tuple(v)
        kwargs[k] = v
    return cls(**kwargs)
