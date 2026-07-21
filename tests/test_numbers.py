"""Read-until-bound number binding: scheduling, stopping, re-entry."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reelcut.config import ReelcutConfig
from reelcut.numbers import bind_numbers
from reelcut.types import OcrRead

from fixtures import make_frame, player

CFG = ReelcutConfig()


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """60 source frames of flat color at 30fps, 320x240."""
    path = tmp_path_factory.mktemp("vid") / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(150):
        w.write(np.full((240, 320, 3), 90, np.uint8))
    w.release()
    return path


def track_frames(tid=5, n=10, start=0):
    """n sampled frames (5 fps grid over a 30fps source) of one player."""
    return [make_frame(start + i, [player(tid, 60, 60, h=100)]) for i in range(n)]


class CountingReader:
    def __init__(self, answers):
        self.calls = 0
        self.answers = answers          # per-call results

    def __call__(self, crop):
        result = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return result


def test_binds_on_first_success_then_stops(tiny_video):
    # 20 samples span 3.8s -> attempts scheduled at ~t=0,1,2,3 (1 Hz)
    reader = CountingReader([None, None, ("24", 0.9), ("99", 0.9)])
    frames = track_frames(n=20)
    enriched, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {5: "24"}
    assert reader.calls == 3            # stopped right after the success
    reads = [r for f in enriched for r in f.ocr]
    assert reads == [OcrRead(track_id=5, text="24", confidence=0.9)]


def test_attempts_capped(tiny_video):
    reader = CountingReader([None])
    frames = track_frames(n=10)         # 10 samples over 2s at 1 Hz -> ~2-3 eligible
    enriched, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {}
    assert reader.calls <= CFG.number_max_attempts


def test_attempt_spacing_respects_hz(tiny_video):
    reader = CountingReader([None])
    frames = track_frames(n=10)         # spans 1.8s of track life
    bind_numbers(frames, tiny_video, CFG, reader)
    # at 1 attempt/sec across 1.8s: at most 2 attempts (t=0 and t>=1.0)
    assert reader.calls == 2


def test_reentry_gets_fresh_attempts(tiny_video):
    reader = CountingReader([("24", 0.8)])
    frames = track_frames(tid=5, n=3) + track_frames(tid=9, n=3, start=5)
    enriched, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {5: "24", 9: "24"}  # same kid re-entering = new track = re-read
    assert reader.calls == 2


def test_prebound_tracks_skip_reader(tiny_video):
    reader = CountingReader([("7", 0.9)])
    frames = track_frames(n=4)
    frames[0] = make_frame(0, [player(5, 60, 60, h=100)],
                           ocr=[OcrRead(5, "24", 0.9)])
    enriched, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {5: "24"}
    assert reader.calls == 0


def test_missing_video_is_graceful():
    reader = CountingReader([("24", 0.9)])
    frames = track_frames(n=4)
    enriched, bound = bind_numbers(frames, Path("/nope.mp4"), CFG, reader)
    assert bound == {} and reader.calls == 0
    assert enriched == frames
