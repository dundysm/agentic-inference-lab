#!/usr/bin/env python3
"""Evaluate the distributed routing experiment follow-up cohorts against docs/protocols/routing-sensitivity-2026-10-12.md.

Usage: python scripts/routing_followup_evaluate.py sens RESULTS_DIR   (hypotheses S-H1..S-H6)
       python scripts/routing_followup_evaluate.py supp RESULTS_DIR   (hypotheses X-H1..X-H4)
"""
from __future__ import annotations

import glob
import importlib.util
import json
import statistics as s
import sys
from collections import defaultdict
from pathlib import Path

here = Path(__file__).parent
def _load(name):
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
ana = _load("routing_analyze")
ev = _load("routing_evaluate")

DIGEST = "c9c7e42687f3"
verdict = lambda ok: "SUPPORTED" if ok else "NOT SUPPORTED"


def load(root: Path):
    R, invalid = {}, []
    for d in sorted({Path(x).parent for x in glob.glob(str(root / "rep_*" / "R*" / "requests.jsonl"))} |
                    {Path(x).parent for x in glob.glob(str(root / "rep_*" / "S*" / "requests.jsonl"))}):
        x = ana.summarize(d, 3000)
        rows = [json.loads(l) for l in (d / "requests.jsonl").open()]
        x["resumed_mean_cached"] = s.mean(r["cached_tokens"] / r["prompt_tokens"] for r in rows if r["turn"] > 0)
        R[(x["rep"], x["arm"])] = (x, d)
        if not (x["completed"] and x["ok"] == x["requests"] == 192 and x["placement_known"] == 192
                and x["prompt_digest"] == DIGEST):
            invalid.append(str(d))
    return R, invalid


def table(R, reps, arms, base):
    print(f"\nPer rep (recomputed prefill vs {base} | resumed p95 ms | wall s | resumed mean cached):")
    for rep in reps:
        cells = []
        for arm in arms:
            x = R[(rep, arm)][0]
            cells.append(f"{arm} {x['prefill_computed_total'] / R[(rep, base)][0]['prefill_computed_total']:.2f}x | "
                         f"p95 {x['resumed_ttft_p95_ms']} | {x['wall_s']}s | {x['resumed_mean_cached']:.3f}")
        print(f"  rep {rep}: " + "   ".join(cells))


def sens(R, reps):
    v = lambda rep, arm, k: R[(rep, arm)][0][k]
    ratio = lambda a, b, k: [v(r, a, k) / v(r, b, k) for r in reps]
    table(R, reps, ["R0", "R1", "R1S4", "R2"], "R0")
    h1 = all(x < 0.5 for x in ratio("R2", "R0", "prefill_computed_total"))
    r1r0 = s.median(ratio("R1", "R0", "prefill_computed_total"))
    s4r2 = s.median(ratio("R1S4", "R2", "prefill_computed_total"))
    p95 = s.median(ratio("R1S4", "R2", "resumed_ttft_p95_ms"))
    mv = [ev.move_misses(R[(r, a)][1]) for r in reps for a in ("R0", "R1")]
    on, tot = sum(m for m, _ in mv), sum(t for _, t in mv)
    share = [v(r, "R2", "w0_requests") / 192 for r in reps]
    print(f"\nS-H1 R2 prefill <0.5x R0 every rep: {verdict(h1)}")
    print(f"S-H2 median R1/R0 prefill = {r1r0:.2f} (needs >=0.75): {verdict(r1r0 >= 0.75)}")
    print(f"S-H3 median R1S4/R2 prefill = {s4r2:.2f} (needs <=1.2): {verdict(s4r2 <= 1.2)}")
    print(f"S-H4 median R1S4/R2 resumed p95 = {p95:.2f} (needs <=1.2): {verdict(p95 <= 1.2)}")
    print(f"S-H5 misses on moves (R0,R1) = {on}/{tot} = {on / tot if tot else float('nan'):.3f} (needs >=0.9): "
          f"{verdict(tot > 0 and on / tot >= 0.9)}")
    print(f"S-H6 R2 worker-0 share per rep = {[round(x, 3) for x in share]} (needs 0.40-0.60): "
          f"{verdict(all(0.4 <= x <= 0.6 for x in share))}")


def supp(R, reps):
    v = lambda rep, arm, k: R[(rep, arm)][0][k]
    table(R, reps, ["R2", "R3H", "S45", "S9"], "R2")
    r3h = s.median(v(r, "R3H", "prefill_computed_total") / v(r, "R2", "prefill_computed_total") for r in reps)
    s45 = [v(r, "S45", "resumed_mean_cached") for r in reps]
    s9 = [v(r, "S9", "resumed_mean_cached") for r in reps]
    s9r2 = s.median(v(r, "S9", "prefill_computed_total") / v(r, "R2", "prefill_computed_total") for r in reps)
    walls = [(v(r, "R2", "wall_s"), v(r, "S9", "wall_s")) for r in reps]
    print(f"\nX-H1 median R3H/R2 prefill = {r3h:.3f} (needs within 0.9-1.1): {verdict(0.9 <= r3h <= 1.1)}")
    print(f"X-H2 S45 resumed mean cached per rep = {[round(x, 3) for x in s45]} (needs <0.2 every rep): "
          f"{verdict(all(x < 0.2 for x in s45))}")
    print(f"X-H3 S9 resumed mean cached per rep = {[round(x, 3) for x in s9]} (>0.9 every rep) and median "
          f"S9/R2 prefill = {s9r2:.2f} (<=1.2): {verdict(all(x > 0.9 for x in s9) and s9r2 <= 1.2)}")
    print(f"X-H4 R2 wall < S9 wall every rep {walls}: {verdict(all(a < b for a, b in walls))}")


def main() -> None:
    which, root = sys.argv[1], Path(sys.argv[2])
    R, invalid = load(root)
    reps = sorted({k[0] for k in R})
    print(f"cohort={which} runs={len(R)} reps={reps} invalid={invalid or 'none'}")
    {"sens": sens, "supp": supp}[which](R, reps)


if __name__ == "__main__":
    main()
