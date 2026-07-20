"""All tunable constants in one place.

Sport-specific numbers live in :class:`SportConfig` presets so basketball is a
config change, not a rewrite. Distances are normalized by the target's player
height (bbox height) so constants survive resolution changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


# HSV reference hues (OpenCV scale: H 0-179, S/V 0-255) for user-named kit colors.
# Each entry: (h_lo, h_hi, s_min, v_min). Red wraps around; listed twice.
TEAM_COLORS: dict[str, list[tuple[int, int, int, int]]] = {
    "red": [(0, 10, 70, 50), (170, 179, 70, 50)],
    "orange": [(11, 25, 70, 50)],
    "yellow": [(26, 34, 70, 50)],
    "green": [(35, 85, 60, 40)],
    "blue": [(86, 125, 60, 40)],
    "purple": [(126, 155, 60, 40)],
    "pink": [(156, 169, 50, 80)],
    "white": [(0, 179, 0, 180)],     # low saturation, high value
    "black": [(0, 179, 0, 0)],       # low value (v_max handled in code)
}


@dataclass(frozen=True)
class SportConfig:
    """Sport-dependent scoring constants (soccer defaults)."""

    name: str = "soccer"
    # Involvement scoring: distances in units of target player-height
    ball_near_dist: float = 1.5          # full proximity weight inside this
    ball_far_dist: float = 4.0           # zero proximity weight beyond this
    possession_radius: float = 0.8       # ball within this counts toward possession
    possession_min_s: float = 0.6        # sustained closeness needed for `possession`
    possession_speed_corr: float = 0.5   # min correlation of ball/target speed
    touch_radius: float = 1.0            # ball direction change inside this = `touch`
    touch_direction_change_deg: float = 40.0
    sprint_speed: float = 2.5            # target speed (player-heights/s) tagged `sprint`
    # weights, summed then clamped to 1.0
    w_proximity: float = 0.55
    w_possession: float = 0.35
    w_touch: float = 0.35
    w_sprint: float = 0.25
    # player-agnostic goal-chance fallback: distances in units of the detected
    # goal bbox height (youth goal ~2 m — same visual scale as a player)
    goal_chance_dist: float = 2.0        # full weight inside this
    goal_chance_far_dist: float = 4.0    # zero beyond this
    goal_chance_speed: float = 3.0       # ball speed (goal-heights/s) = shot bonus


@dataclass(frozen=True)
class ReelcutConfig:
    # sampling
    sample_fps: float = 5.0
    seed: int = 1337

    # workflow / inference
    workspace: str = "aarnavs-space"
    workflow_id: str = "reelcut-tracking"       # URL slug of the saved workflow
    model_id: str = "my-first-project-gsrpg-p5v9o/1"  # our RF-DETR-small (mAP50 81.1)
    api_url: str | None = None                  # None = in-process; else inference server URL
    player_classes: tuple[str, ...] = ("player", "goalkeeper", "goalie")
    referee_classes: tuple[str, ...] = ("referee", "ref")
    ball_classes: tuple[str, ...] = ("ball",)
    min_player_box_h_frac: float = 0.02         # drop boxes shorter than 2% of frame height

    # identity stitching
    ocr_pos_confidence: float = 0.4     # OCR conf needed to count a jersey read
    ocr_pos_votes: int = 2              # matching reads to call tracklet TARGET
    ocr_neg_votes: int = 2              # confident different-number reads to call NOT_TARGET
    color_veto_dist: float = 0.75       # histogram distance beyond which team color vetoes
    max_plausible_speed: float = 8.0    # player-heights/s; faster linking = implausible
    link_max_gap_s: float = 3.0         # max gap to kinematically link tracklets
    link_max_dist: float = 3.0          # player-heights at gap=0, scaled by gap
    identity_floor: float = 0.25        # min confidence to keep TARGET claims

    # goal-chance fallback: when the target is lost (off-screen, no OCR, bad
    # tracking), player-agnostic goal-mouth action still makes the reel.
    fallback_enabled: bool = True
    fallback_weight: float = 0.85       # opportunity score vs target involvement

    # involvement
    smooth_window_s: float = 2.0
    event_threshold: float = 0.30
    event_min_s: float = 1.0
    ball_gap_interp_max_s: float = 1.5  # interpolate ball across gaps up to this

    # clip cutting
    clip_pad_s: float = 4.0
    clip_merge_gap_s: float = 3.0
    clip_min_s: float = 5.0
    clip_max_s: float = 25.0

    sport: SportConfig = field(default_factory=SportConfig)

    def for_sport(self, name: str) -> "ReelcutConfig":
        presets = {"soccer": SportConfig()}
        if name not in presets:
            raise ValueError(f"unknown sport {name!r}; have {sorted(presets)}")
        return replace(self, sport=presets[name])


@dataclass(frozen=True)
class Paths:
    """Filesystem layout for one run."""

    out_dir: Path

    @property
    def cache_dir(self) -> Path:
        return self.out_dir / "cache"

    @property
    def highlights_mp4(self) -> Path:
        return self.out_dir / "highlights.mp4"

    @property
    def highlights_json(self) -> Path:
        return self.out_dir / "highlights.json"

    @property
    def debug_mp4(self) -> Path:
        return self.out_dir / "debug.mp4"
