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
    model_id: str = "ia-foot-8ecu7/1"  # RF-DETR-small on IA Foot (mAP50 90.1)
    api_url: str | None = None                  # None = in-process; else inference server URL
    player_classes: tuple[str, ...] = ("player", "goalkeeper", "goalie")
    referee_classes: tuple[str, ...] = ("referee", "ref")
    ball_classes: tuple[str, ...] = ("ball",)
    min_player_box_h_frac: float = 0.02         # drop boxes shorter than 2% of frame height

    # stage 1.5: jersey-number binding — read-until-bound, then the tracker
    # carries the number; a dead track's successor repeats the process.
    digit_model_id: str = "jersey-number-detection-8a55j-ob8fb/1"
    number_attempt_hz: float = 1.0      # attempts per second of unbound track life
    number_max_attempts: int = 8        # hard cap per tracklet
    digit_min_conf: float = 0.35

    # identity stitching
    ocr_pos_confidence: float = 0.4     # OCR conf needed to count a jersey read
    ocr_pos_votes: int = 2              # matching reads to promote to TARGET —
                                        # pairs with confirm-then-lock (numbers.py
                                        # keeps reading until 2 agree), so real
                                        # binds always reach 2; one misread can't
                                        # steal the violet target box
    number_min_crop_h: int = 60         # skip number attempts on crops shorter
                                        # than this (tiny crops = systematic misreads)
    number_audit_gap_s: float = 4.0     # economy mode: min gap between audit reads
    number_audit_attempts: int = 3      # economy mode: max big-crop audit reads
    number_continuous: bool = True      # always-on: read every player at every
                                        # sampled frame; display carries the
                                        # last-read number per track
                                        # (False = old stop-at-lock economy mode)
    ocr_neg_votes: int = 2              # confident different-number reads to call NOT_TARGET

    # appearance re-ID: CLIP embeddings of player crops vs a reference bank
    # harvested from the seed click's tracklet. DISABLED by default: measured
    # on the CSKA fixture (scripts/calibrate_reid.py), full-crop CLIP cannot
    # separate same-team kids — true-target tracks scored contrast +0.015 to
    # -0.004 while junk tracks hit +0.061. Track-level jersey-read dominance
    # is the working identity backbone; revisit re-ID with a torso-crop or
    # dedicated person-reID embedder.
    reid_enabled: bool = False
    reid_ref_samples: int = 16          # crops harvested from the seed track
    reid_samples_per_track: int = 6     # crops embedded per candidate track
    reid_min_crop_h: int = 80           # px; smaller crops embed unreliably
    # Scoring is CONTRASTIVE (seed bank vs banks of tracks whose dominant
    # jersey read is a different number): absolute CLIP sims are useless on
    # player crops — measured 0.88-0.95 for EVERY kid on the CSKA fixture,
    # kit + grass dominate. See scripts/calibrate_reid.py.
    reid_min_sim: float = 0.90          # floor on sim-to-seed for any claim
    reid_contrast_margin: float = 0.02  # required (sim_seed - sim_negatives)
    reid_neg_min_reads: int = 30        # reads needed to trust a track as a
                                        # negative reference
    color_veto_dist: float = 0.75       # histogram distance beyond which team color vetoes
    max_plausible_speed: float = 8.0    # player-heights/s; faster linking = implausible
    link_max_gap_s: float = 3.0         # max gap to kinematically link tracklets
    link_max_dist: float = 3.0          # player-heights at gap=0, scaled by gap
    identity_floor: float = 0.25        # min confidence to keep TARGET claims

    # goal-chance fallback: when the target is lost (off-screen, no OCR, bad
    # tracking), player-agnostic goal-mouth action still makes the reel.
    fallback_enabled: bool = True
    fallback_weight: float = 0.85       # opportunity score vs target involvement
    goal_box_hold_s: float = 2.5        # carry goal boxes across detection
                                        # dropouts (players occlude the goal
                                        # exactly when shots happen)

    # involvement
    smooth_window_s: float = 2.0
    event_threshold: float = 0.30
    event_min_s: float = 1.0
    ball_gap_interp_max_s: float = 1.5  # interpolate ball across gaps up to this

    # reel budget: if events would cover more than max_reel_fraction of the
    # source, the event threshold rises until coverage ~ target_reel_fraction.
    # Guards against any saturating score signal (e.g. goal boxes everywhere
    # in close-up small-sided footage).
    target_reel_fraction: float = 0.12
    max_reel_fraction: float = 0.20

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
