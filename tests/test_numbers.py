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

CFG = ReelcutConfig(number_continuous=False)   # economy-mode behavior tests
CONT = ReelcutConfig()                          # continuous is the default


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """60 source frames of flat color at 30fps, 320x240."""
    path = tmp_path_factory.mktemp("vid") / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(450):
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


def test_locks_after_two_agreeing_reads(tiny_video):
    # 20 samples span 3.8s -> attempts scheduled at ~t=0,1,2,3 (1 Hz)
    reader = CountingReader([None, ("24", 0.9), ("24", 0.85), ("99", 0.9)])
    frames = track_frames(n=20)
    enriched, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {5: "24"}
    assert reader.calls == 3            # confirmed on 2nd agreeing read, stopped
    reads = [r for f in enriched for r in f.ocr]
    assert [r.text for r in reads] == ["24", "24"]


def test_partial_read_upgrades_to_full_number(tiny_video):
    # the "#1 on a 15 shirt" bug: "1" then "15" agree -> confirmed "15"
    reader = CountingReader([("1", 0.6), ("15", 0.7), ("99", 0.9)])
    frames = track_frames(n=20)
    _, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {5: "15"}
    assert reader.calls == 2


def test_conflicting_reads_never_lock(tiny_video):
    reader = CountingReader([("11", 0.9), ("7", 0.9)])
    frames = track_frames(n=10)          # only 2 attempts fit
    _, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {}                   # wrong lock is worse than no label


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
    reader = CountingReader([("24", 0.8)])   # every call reads "24"
    frames = track_frames(tid=5, n=8) + track_frames(tid=9, n=8, start=10)
    enriched, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {5: "24", 9: "24"}  # same kid re-entering = new track = re-read
    assert reader.calls == 4            # 2 confirming reads per track


def test_preconfirmed_tracks_skip_reader(tiny_video):
    reader = CountingReader([("7", 0.9)])
    frames = track_frames(n=4)
    frames[0] = make_frame(0, [player(5, 60, 60, h=100)],
                           ocr=[OcrRead(5, "24", 0.9), OcrRead(5, "24", 0.8)])
    enriched, bound = bind_numbers(frames, tiny_video, CFG, reader)
    assert bound == {5: "24"}
    assert reader.calls == 0            # two agreeing existing reads = confirmed


def test_assemble_number_picks_central_cluster():
    from reelcut.numbers import assemble_number
    # "10" centered, a neighbor's "8" far right (the measured "108" bug)
    digits = [(70, 12, "1", 0.9), (84, 12, "0", 0.85), (150, 12, "8", 0.8)]
    assert assemble_number(digits, crop_width=160) == ("10", 0.85)


def test_assemble_number_rejects_ambiguous_triples():
    from reelcut.numbers import assemble_number
    digits = [(60, 12, "7", 0.9), (74, 12, "1", 0.9), (88, 12, "4", 0.9)]
    assert assemble_number(digits, crop_width=160) is None


def test_assemble_number_simple_cases():
    from reelcut.numbers import assemble_number
    assert assemble_number([(80, 12, "7", 0.8)], 160) == ("7", 0.8)
    assert assemble_number([], 160) is None


def test_missing_video_is_graceful():
    reader = CountingReader([("24", 0.9)])
    frames = track_frames(n=4)
    enriched, bound = bind_numbers(frames, Path("/nope.mp4"), CFG, reader)
    assert bound == {} and reader.calls == 0
    assert enriched == frames


def test_clear_read_outweighs_repeated_squints():
    from reelcut.numbers import merge_reads
    # two low-quality reads formed "11"; one high-quality read says "7"
    value, ok = merge_reads([("1", 1.0), ("11", 1.0), ("7", 2.0)])
    assert (value, ok) == ("7", True)


def test_audit_overturns_wrong_lock(tiny_video):
    """The '7 clearly visible but it keeps 11' case: small-crop reads lock
    '11'; later big-crop audits read '7' and overturn."""
    reader = CountingReader([("11", 0.8), ("11", 0.8), ("7", 0.9), ("7", 0.9)])
    small = [make_frame(i, [player(5, 60, 60, h=100)]) for i in range(8)]
    big = [make_frame(30 + i, [player(5, 60, 60, h=150)]) for i in range(40)]
    enriched, bound = bind_numbers(small + big, tiny_video, CFG, reader)
    assert bound.get(5) == "7"
    assert reader.calls >= 3            # audits actually ran


def test_continuous_mode_outvotes_early_misread(tiny_video):
    """User design: keep reading; wrong early reads get outvoted by the
    ongoing consensus once the number is genuinely visible."""
    reader = CountingReader([("11", 0.8), ("11", 0.8)] + [("7", 0.9)] * 20)
    frames = [make_frame(i, [player(5, 60, 60, h=100)]) for i in range(60)]
    enriched, bound = bind_numbers(frames, tiny_video, CONT, reader)
    assert bound.get(5) == "7"
    assert reader.calls > 4              # kept reading past any would-be lock


def test_continuous_mode_is_default():
    assert ReelcutConfig().number_continuous is True


def test_continuous_mode_reads_every_sampled_frame(tiny_video):
    """User design: every sampled frame is read during enrollment — no
    cadence gate (10 samples here span 1.8s, inside the 3s window)."""
    reader = CountingReader([("7", 0.9)])
    frames = track_frames(n=10)
    bind_numbers(frames, tiny_video, CONT, reader)
    assert reader.calls == 10


def test_continuous_reads_stop_after_enrollment(tiny_video):
    """After 3s of successful reads the majority is locked and the digit
    model stops running for that track — the tracker carries the number."""
    reader = CountingReader([("7", 0.9)])
    frames = track_frames(n=40)             # 7.8s of track life at 5 fps
    _, bound = bind_numbers(frames, tiny_video, CONT, reader)
    assert bound == {5: "7"}
    assert reader.calls == 16               # t=0.0 .. t=3.0 inclusive, then locked


def test_enrollment_clock_starts_at_first_successful_read(tiny_video):
    """User-specified: a player whose jersey is hidden at first keeps getting
    attempts every frame; THEIR 3 seconds start when the jersey first shows."""
    reader = CountingReader([None] * 20 + [("7", 0.9)])
    frames = track_frames(n=45)             # 8.8s of track life at 5 fps
    _, bound = bind_numbers(frames, tiny_video, CONT, reader)
    assert bound == {5: "7"}
    # 20 failed attempts (t=0..3.8) + reads from first success at t=4.0
    # until t=7.0 closes the window -> samples i=0..35 all attempted
    assert reader.calls == 36


def test_concatenation_never_hijacks_majority():
    """The measured '#71 on the 7 kid' bug: one fused read must not out-rank
    a track full of clean reads of the real number."""
    from reelcut.numbers import merge_reads

    value, ok = merge_reads(["7", "7", "7", "71"])
    assert (value, ok) == ("7", True)


def test_assemble_number_keeps_leading_zero_jerseys():
    from reelcut.numbers import assemble_number

    # TargetSpec documents jerseys like "07" — leading zeros are legitimate
    digits = [(70, 12, "0", 0.9), (84, 12, "7", 0.85)]
    assert assemble_number(digits, crop_width=160) == ("07", 0.85)


def _read(tid, text, conf=0.9):
    return OcrRead(track_id=tid, text=text, confidence=conf)


def test_timeline_first_read_labels_immediately_and_carries():
    from reelcut.numbers import number_timeline

    frames = [
        make_frame(0, [player(5, 60, 60)], ocr=[_read(5, "7")]),
        make_frame(1, [player(5, 60, 60)]),               # unreadable: carries
        make_frame(2, [player(5, 60, 60)]),
    ]
    assert number_timeline(frames) == {5: [(0, "7")]}


def test_timeline_one_glitch_read_never_flips():
    from reelcut.numbers import number_timeline

    frames = [
        make_frame(0, [player(5, 60, 60)], ocr=[_read(5, "7")]),
        make_frame(1, [player(5, 60, 60)], ocr=[_read(5, "13")]),   # glitch
        make_frame(2, [player(5, 60, 60)], ocr=[_read(5, "7")]),
    ]
    assert number_timeline(frames) == {5: [(0, "7")]}


def test_timeline_two_agreeing_reads_switch_the_label():
    """Visible-again semantics (and stolen-track recovery): a genuinely new
    number takes over after two consecutive agreeing reads."""
    from reelcut.numbers import number_timeline

    frames = [
        make_frame(0, [player(5, 60, 60)], ocr=[_read(5, "7")]),
        make_frame(1, [player(5, 60, 60)], ocr=[_read(5, "13")]),
        make_frame(2, [player(5, 60, 60)], ocr=[_read(5, "13")]),
    ]
    assert number_timeline(frames) == {5: [(0, "7"), (12, "13")]}


def test_timeline_partial_read_confirms_instead_of_flipping():
    from reelcut.numbers import number_timeline

    frames = [
        make_frame(0, [player(5, 60, 60)], ocr=[_read(5, "16")]),
        make_frame(1, [player(5, 60, 60)], ocr=[_read(5, "1")]),    # half-hidden 16
        make_frame(2, [player(5, 60, 60)], ocr=[_read(5, "6")]),    # other half
    ]
    assert number_timeline(frames) == {5: [(0, "16")]}


def test_timeline_superstring_upgrade():
    from reelcut.numbers import number_timeline

    frames = [
        make_frame(0, [player(5, 60, 60)], ocr=[_read(5, "1")]),
        make_frame(1, [player(5, 60, 60)], ocr=[_read(5, "16")]),
        make_frame(2, [player(5, 60, 60)], ocr=[_read(5, "16")]),
    ]
    # "1" and "16" agree (substring); tie broken toward the longer read
    assert number_timeline(frames) == {5: [(0, "1"), (6, "16")]}


def test_timeline_fusion_burst_is_outvoted_by_surrounding_clean_reads():
    """Review-confirmed failure: consecutive fused '71' reads (two kids
    adjacent for several frames) must not capture the label of a track full
    of clean '7's — the window vote keeps '7' in charge."""
    from reelcut.numbers import number_timeline

    reads = ["7"] * 5 + ["71", "71"] + ["7"] * 3
    frames = [
        make_frame(i, [player(5, 60, 60)], ocr=[_read(5, r)])
        for i, r in enumerate(reads)
    ]
    assert number_timeline(frames) == {5: [(0, "7")]}


def test_timeline_recovers_when_fusion_lands_first():
    """No absorbing state: even a track whose FIRST read was a fusion ('71')
    flips to the true number once clean reads outnumber it."""
    from reelcut.numbers import number_timeline

    frames = [
        make_frame(0, [player(5, 60, 60)], ocr=[_read(5, "71")]),
        make_frame(1, [player(5, 60, 60)], ocr=[_read(5, "7")]),
        make_frame(2, [player(5, 60, 60)], ocr=[_read(5, "7")]),
    ]
    assert number_timeline(frames) == {5: [(0, "71"), (12, "7")]}


def test_timeline_label_frozen_after_enrollment():
    """User-specified: the 3s enrollment majority IS the track's number for
    life — later conflicting reads never flicker the label."""
    from reelcut.numbers import number_timeline

    reads = ["7"] * 20 + ["13"] * 10       # "13"s arrive after t=3s: ignored
    frames = [
        make_frame(i, [player(5, 60, 60)], ocr=[_read(5, r)])
        for i, r in enumerate(reads)
    ]
    assert number_timeline(frames) == {5: [(0, "7")]}


def test_timeline_reentry_reenrolls_per_track():
    """A player who left the frame comes back as a NEW track id and enrolls
    from scratch — every individual player gets their own 3s vote."""
    from reelcut.numbers import number_timeline

    frames = [
        make_frame(i, [player(5, 60, 60)], ocr=[_read(5, "7")])
        for i in range(3)
    ] + [
        make_frame(30 + i, [player(9, 60, 60)], ocr=[_read(9, "13")])
        for i in range(3)
    ]
    tl = number_timeline(frames)
    assert tl[5] == [(0, "7")]
    assert tl[9] == [(180, "13")]
