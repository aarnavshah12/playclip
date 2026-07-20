"""Goal-chance fallback: player-agnostic scoring when the target is lost."""
from __future__ import annotations

from dataclasses import replace

from fixtures import FPS, H, W, ball, make_frame
from reelcut.config import ReelcutConfig
from reelcut.scoring import blend_scores, extract_events, score_opportunities
from reelcut.types import BBox, IdentityPoint, ScorePoint

CFG = ReelcutConfig()

GOAL = BBox(0.0, H * 0.40, 40.0, 120.0)          # left goal, height 120 px
FAR_FROM_GOAL = (W * 0.6, H * 0.5)                # midfield


def frames_with_goal(ball_positions):
    """One frame per position; ball at the position (or absent for None)."""
    out = []
    for i, p in enumerate(ball_positions):
        f = make_frame(i, [], ball(p[0], p[1]) if p is not None else None)
        out.append(replace(f, goal_boxes=(GOAL,)))
    return out


def test_ball_at_goal_scores_high_with_tag():
    frames = frames_with_goal([(GOAL.x2 + 10, GOAL.cy)] * 10)
    pts = score_opportunities(frames, CFG)
    assert all(p.score > 0.6 for p in pts)
    assert all("goal_chance" in p.tags for p in pts)


def test_ball_midfield_scores_zero():
    frames = frames_with_goal([FAR_FROM_GOAL] * 10)
    pts = score_opportunities(frames, CFG)
    assert all(p.score == 0.0 for p in pts)
    assert all(p.tags == () for p in pts)


def test_no_ball_or_no_goal_scores_zero():
    no_ball = frames_with_goal([None] * 5)
    assert all(p.score == 0.0 for p in score_opportunities(no_ball, CFG))
    no_goal = [make_frame(i, [], ball(50.0, GOAL.cy)) for i in range(5)]
    assert all(p.score == 0.0 for p in score_opportunities(no_goal, CFG))


def test_falloff_between_near_and_far():
    gh = GOAL.h
    mid_d = (CFG.sport.goal_chance_dist + CFG.sport.goal_chance_far_dist) / 2.0
    frames = frames_with_goal([(GOAL.x2 + mid_d * gh, GOAL.cy)] * 3)
    pts = score_opportunities(frames, CFG)
    assert all(0.0 < p.score < 0.75 for p in pts)


def test_blend_lets_fallback_carry_when_target_lost():
    primary = [ScorePoint(i / FPS, 0.0, ()) for i in range(10)]
    fb = [ScorePoint(i / FPS, 0.9, ("goal_chance",)) for i in range(10)]
    blended = blend_scores(primary, fb, weight=0.85)
    assert all(abs(p.score - 0.9 * 0.85) < 1e-9 for p in blended)
    assert all(p.tags == ("goal_chance",) for p in blended)
    events = extract_events(blended, CFG.event_threshold, CFG.event_min_s)
    assert events and "goal_chance" in events[0].tags


def test_blend_keeps_higher_target_score_and_merges_tags():
    primary = [ScorePoint(i / FPS, 0.95, ("possession",)) for i in range(5)]
    fb = [ScorePoint(i / FPS, 0.5, ("goal_chance",)) for i in range(5)]
    blended = blend_scores(primary, fb, weight=0.85)
    assert all(p.score == 0.95 for p in blended)
    assert all(set(p.tags) == {"goal_chance", "possession"} for p in blended)


def test_blend_ignores_negligible_fallback():
    primary = [ScorePoint(i / FPS, 0.4, ("near_ball",)) for i in range(5)]
    fb = [ScorePoint(i / FPS, 0.01, ("goal_chance",)) for i in range(5)]
    blended = blend_scores(primary, fb, weight=0.85)
    assert all(p.tags == ("near_ball",) for p in blended)


def test_shot_speed_bonus():
    gh = GOAL.h
    d = CFG.sport.goal_chance_dist + 1.0          # in the falloff zone
    x0 = GOAL.x2 + d * gh
    slow = frames_with_goal([(x0, GOAL.cy), (x0 - 1.0, GOAL.cy)])
    # fast: covers > goal_chance_speed goal-heights in one 0.2 s sample
    step = CFG.sport.goal_chance_speed * gh / FPS * 1.5
    fast = frames_with_goal([(x0, GOAL.cy), (x0 - step, GOAL.cy)])
    slow_p = score_opportunities(slow, CFG)[1].score
    fast_p = score_opportunities(fast, CFG)[1].score
    assert fast_p > slow_p
