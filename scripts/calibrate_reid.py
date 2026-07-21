"""Measure CLIP similarity separation on a cached run to set reid thresholds.

Usage: uv run python scripts/calibrate_reid.py <cache_dir> <video> <seed_track_id>
Prints every embedded track's max sim to the seed bank alongside its dominant
jersey read, so target/not-target separation is visible before picking
reid_match_sim / reid_veto_sim.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from reelcut import reid, stitching                      # noqa: E402
from reelcut.config import ReelcutConfig                 # noqa: E402
from reelcut.types import from_jsonable                  # noqa: E402


def main() -> None:
    cache_dir, video, seed_tid = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    with gzip.open(cache_dir / "stage1_observations.json.gz", "rt") as f:
        frames = from_jsonable(json.load(f))
    with gzip.open(cache_dir / "stage2_identity.json.gz", "rt") as f:
        payload = from_jsonable(json.load(f))
    reads_by_track: dict[int, Counter] = {}
    for fi, r in payload.get("number_reads") or []:
        digits = "".join(c for c in r.text if c.isdigit())
        if digits:
            reads_by_track.setdefault(r.track_id, Counter())[digits] += 1

    cfg = ReelcutConfig()
    tracklets = stitching.build_tracklets(frames)
    # attach the cached fresh reads so negative_track_ids sees them
    for t in tracklets:
        extra = []
        for fi, r in payload.get("number_reads") or []:
            if r.track_id == t.track_id:
                extra.append(r)
        t.ocr_reads = list(t.ocr_reads) + extra

    jersey = sys.argv[4] if len(sys.argv) > 4 else "7"
    negatives = reid.negative_track_ids(tracklets, jersey, cfg.reid_neg_min_reads)
    embedder = reid.make_clip_embedder(os.environ["ROBOFLOW_API_KEY"])
    embeds = reid.track_embeddings(tracklets, seed_tid, video, embedder, cfg)
    scores = reid.contrastive_scores(embeds, seed_tid, negatives)

    life = {t.track_id: (t.start_s, t.end_s) for t in tracklets}
    print(f"\nseed {seed_tid}; jersey #{jersey}; {len(embeds)} embedded; "
          f"{len(negatives)} negative banks: {sorted(negatives)}\n")
    for tid, (s_pos, s_neg) in sorted(scores.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
        top = reads_by_track.get(tid)
        read = f"#{top.most_common(1)[0][0]} x{sum(top.values())}" if top else "-"
        t0, t1 = life[tid]
        mark = " NEG" if tid in negatives else ""
        print(f"track {tid:4d}  pos {s_pos:.3f}  neg {s_neg:.3f}  "
              f"contrast {s_pos - s_neg:+.3f}  t={t0:5.1f}-{t1:5.1f}  reads {read}{mark}")


if __name__ == "__main__":
    main()
