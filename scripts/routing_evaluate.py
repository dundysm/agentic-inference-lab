#!/usr/bin/env python3
"""Evaluate a the distributed routing experiment cohort against docs/protocols/routing-primary-2026-10-12.md (H1-H5).

Usage: python scripts/routing_evaluate.py RESULTS_DIR   (holds rep_*/R*/ run directories)
"""
from __future__ import annotations

import glob
import importlib.util
import json
import statistics as s
import sys
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location("day2_analyze", Path(__file__).with_name("routing_analyze.py"))
ana = importlib.util.module_from_spec(spec); spec.loader.exec_module(ana)

DIGEST = "c9c7e42687f3"


def move_misses(d: Path) -> tuple[int, int]:
    """(resumed misses on turns where the session changed worker, all resumed misses)."""
    wmap = {int(k): v for k, v in json.load((d / "worker_map.json").open()).items()}
    by = defaultdict(dict)
    for r in map(json.loads, (d / "requests.jsonl").open()):
        by[r["session"]][r["turn"]] = r
    on_move = total = 0
    for turns in by.values():
        for t in range(1, len(turns)):
            prev, cur = turns[t - 1], turns[t]
            if cur["cached_tokens"] / cur["prompt_tokens"] < 0.5:
                total += 1
                on_move += wmap[prev["worker_id"]] != wmap[cur["worker_id"]]
    return on_move, total


def main() -> None:
    root = Path(sys.argv[1])
    runs = sorted({Path(x).parent for x in glob.glob(str(root / "rep_*" / "R*" / "requests.jsonl"))})
    R = {}
    invalid = []
    for d in runs:
        x = ana.summarize(d, 3000)
        R[(x["rep"], x["arm"])] = (x, d)
        if not (x["completed"] and x["ok"] == x["requests"] == 192 and x["placement_known"] == 192
                and x["prompt_digest"] == DIGEST):
            invalid.append(str(d))
    reps = sorted({k[0] for k in R})
    print(f"runs={len(R)} reps={reps} invalid={invalid or 'none'}")
    v = lambda rep, arm, k: R[(rep, arm)][0][k]

    print("\nPer rep (prefill ratio vs R0 | resumed p95 ms | wall s):")
    for rep in reps:
        cells = []
        for arm in ("R0", "R1", "R2", "R3"):
            ratio = v(rep, arm, "prefill_computed_total") / v(rep, "R0", "prefill_computed_total")
            cells.append(f"{arm} {ratio:.2f}x | p95 {v(rep, arm, 'resumed_ttft_p95_ms')} | {v(rep, arm, 'wall_s')}s")
        print(f"  rep {rep}: " + "   ".join(cells))

    h1 = all(v(r, a, "prefill_computed_total") < 0.5 * v(r, "R0", "prefill_computed_total") and
             v(r, a, "resumed_ttft_p95_ms") < 0.5 * v(r, "R0", "resumed_ttft_p95_ms")
             for r in reps for a in ("R2", "R3"))
    r1_below = sum(v(r, "R1", "prefill_computed_total") < v(r, "R0", "prefill_computed_total") for r in reps)
    r1_vs_r2 = s.median(v(r, "R1", "prefill_computed_total") / v(r, "R2", "prefill_computed_total") for r in reps)
    h2 = r1_below >= min(4, len(reps)) and r1_vs_r2 > 2
    mv = [move_misses(R[(r, a)][1]) for r in reps for a in ("R0", "R1")]
    on_move, total = sum(m for m, _ in mv), sum(t for _, t in mv)
    h3 = total > 0 and on_move / total >= 0.9
    r3r2 = s.median(v(r, "R3", "prefill_computed_total") / v(r, "R2", "prefill_computed_total") for r in reps)
    split_ok = all(0.45 <= v(r, "R2", "w0_requests") / 192 <= 0.55 for r in reps)
    h4 = abs(r3r2 - 1) <= 0.10 and split_ok
    h5 = all(v(r, "R2", "wall_s") <= v(r, "R0", "wall_s") for r in reps)

    verdict = lambda ok: "SUPPORTED" if ok else "NOT SUPPORTED"
    print(f"\nH1 affinity: R2,R3 prefill <0.5x R0 and p95 <0.5x R0, every rep: {verdict(h1)}")
    print(f"H2 KV partial: R1<R0 in {r1_below}/{len(reps)} reps; median R1/R2 prefill = {r1_vs_r2:.2f}x (needs >2): {verdict(h2)}")
    print(f"H3 misses on moves: {on_move}/{total} = {on_move / total if total else float('nan'):.3f} (needs >=0.9): {verdict(h3)}")
    print(f"H4 soft~hard: median R3/R2 prefill = {r3r2:.3f}; R2 split within 45-55% every rep: {split_ok}: {verdict(h4)}")
    print(f"H5 no balance penalty: R2 wall <= R0 wall every rep: {verdict(h5)}")


if __name__ == "__main__":
    main()
