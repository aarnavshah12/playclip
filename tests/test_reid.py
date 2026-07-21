"""Appearance re-ID: sampling, contrastive scoring, label application.

Note: re-ID is disabled by default (reid_enabled=False) — measured on real
footage, full-crop CLIP cannot separate same-team kids. The machinery stays
tested so a better embedder can be dropped in.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fixtures import make_frame, player
from reelcut.config import ReelcutConfig
from reelcut.reid import (
    apply_reid,
    contrastive_scores,
    negative_track_ids,
    sample_points,
    track_embeddings,
)
from reelcut.stitching import build_tracklets
from reelcut.types import IdentityLabel, LabeledTracklet, OcrRead

CFG = ReelcutConfig()


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(450):
        w.write(np.full((240, 320, 3), 90, np.uint8))
    w.release()
    return path


def tracks(frames):
    return build_tracklets(frames)


def test_reid_disabled_by_default():
    assert ReelcutConfig().reid_enabled is False


def test_sample_points_spread_and_min_height():
    frames = [make_frame(i, [player(5, 60, 60, h=100 + i)]) for i in range(20)]
    t = tracks(frames)[0]
    picked = sample_points(t, 4, min_crop_h=105)
    assert len(picked) == 4
    assert all(p.bbox.h >= 105 for p in picked)
    idx = [p.frame_index for p in picked]
    assert idx == sorted(idx) and idx[0] != idx[-1]   # spread, not clumped


def test_sample_points_too_small_returns_empty():
    frames = [make_frame(i, [player(5, 60, 60, h=40)]) for i in range(5)]
    assert sample_points(tracks(frames)[0], 4, min_crop_h=80) == []


def test_track_embeddings_and_contrast(tiny_video):
    frames = [
        make_frame(i, [player(1, 40, 40, h=120), player(2, 200, 40, h=120)])
        for i in range(10)
    ]
    embeds = track_embeddings(
        tracks(frames), seed_track_id=1, video=tiny_video,
        embedder=lambda crop: np.array([1.0, 0.0], dtype=np.float32),
        cfg=CFG,
    )
    assert set(embeds) == {1, 2}
    scores = contrastive_scores(embeds, seed_track_id=1, negative_ids=set())
    s_pos, s_neg = scores[2]
    assert s_pos == pytest.approx(1.0)   # identical appearance
    assert s_neg == 0.0                  # no negative banks


def test_contrastive_scores_negative_excludes_self():
    embeds = {
        1: np.array([[1.0, 0.0]], dtype=np.float32),          # seed
        2: np.array([[0.0, 1.0]], dtype=np.float32),          # negative bank
        3: np.array([[0.0, 1.0]], dtype=np.float32),          # looks like 2
    }
    scores = contrastive_scores(embeds, seed_track_id=1, negative_ids={2, 3})
    s_pos, s_neg = scores[3]
    assert s_pos == pytest.approx(0.0)
    assert s_neg == pytest.approx(1.0)   # matched bank 2, not itself
    # the negative track's own negative sim ignores its own bank too
    assert scores[2][1] == pytest.approx(1.0)


def _track_with_reads(tid, reads):
    frames = [
        make_frame(i, [player(tid, 10, 10, h=100)],
                   ocr=[OcrRead(tid, r, 0.9)])
        for i, r in enumerate(reads)
    ]
    return [t for t in tracks(frames) if t.track_id == tid][0]


def test_negative_track_ids_requires_dominant_different_number():
    a = _track_with_reads(2, ["8"] * 40)            # clearly the 8 kid
    b = _track_with_reads(3, ["7"] * 40)            # the target's own number
    c = _track_with_reads(4, ["71"] * 40)           # fusion of the target: excluded
    d = _track_with_reads(5, ["9"] * 10)            # too few reads
    out = negative_track_ids([a, b, c, d], jersey="7", min_reads=30)
    assert out == {2}


def _lt(tid, label, conf=0.0):
    frames = [make_frame(0, [player(tid, 10, 10, h=100)])]
    t = [x for x in tracks(frames) if x.track_id == tid][0]
    return LabeledTracklet(t, label, conf, {})


def test_apply_reid_promotes_on_contrast():
    lt = _lt(9, IdentityLabel.UNKNOWN)
    out = apply_reid([lt], {9: (0.95, 0.90)}, seed_track_id=1, cfg=CFG)
    assert out[0].label is IdentityLabel.TARGET
    assert out[0].confidence >= 0.6
    assert out[0].evidence["reid"] == 0.95


def test_apply_reid_no_promotion_without_margin():
    lt = _lt(9, IdentityLabel.UNKNOWN)
    out = apply_reid([lt], {9: (0.95, 0.945)}, seed_track_id=1, cfg=CFG)
    assert out[0].label is IdentityLabel.UNKNOWN


def test_apply_reid_vetoes_target_closer_to_negatives():
    lt = _lt(9, IdentityLabel.TARGET, conf=0.9)
    out = apply_reid([lt], {9: (0.88, 0.93)}, seed_track_id=1, cfg=CFG)
    assert out[0].label is IdentityLabel.UNKNOWN
    assert out[0].confidence == 0.0


def test_apply_reid_leaves_seed_and_unscored_alone():
    seed = _lt(1, IdentityLabel.TARGET, conf=1.0)
    silent = _lt(7, IdentityLabel.UNKNOWN)
    out = apply_reid([seed, silent], {1: (0.4, 0.9)}, seed_track_id=1, cfg=CFG)
    assert out[0].label is IdentityLabel.TARGET and out[0].confidence == 1.0
    assert out[1].label is IdentityLabel.UNKNOWN


def test_apply_reid_not_target_stays():
    lt = _lt(9, IdentityLabel.NOT_TARGET)
    out = apply_reid([lt], {9: (0.99, 0.5)}, seed_track_id=1, cfg=CFG)
    assert out[0].label is IdentityLabel.NOT_TARGET
