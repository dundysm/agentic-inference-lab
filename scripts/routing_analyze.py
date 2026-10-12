#!/usr/bin/env python3
"""the distributed routing experiment per-run summary: placement, per-worker reuse and recomputation, latency, balance, validity.

Usage: python scripts/routing_analyze.py RESULTS_DIR [--csv OUT.csv]
RESULTS_DIR holds rep_*/R*/ run directories (e.g. results/day2/pilot).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as s
from collections import Counter, defaultdict
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families as parse


def metrics(path: Path) -> dict[str, float]:
    v: dict[str, float] = {}
    for fam in parse(path.read_text()):
        for x in fam.samples:
            v[x.name] = v.get(x.name, 0.0) + x.value
    return v


def delta(d: Path, name: str, key: str) -> float | None:
    a, b = d / f"{name}_metrics_before.prom", d / f"{name}_metrics_after.prom"
    if not (a.exists() and b.exists()):
        return None
    before, after = metrics(a), metrics(b)
    if key not in before or key not in after:
        return None
    return after[key] - before[key]


def p95(xs: list[float]) -> float | None:
    if len(xs) < 50:
        return None
    xs = sorted(xs)
    return xs[int(0.95 * (len(xs) - 1))]


def gpu_util(d: Path) -> dict[int, float]:
    # Mean utilization per GPU over the measured interval (nvidia-smi samples every 0.5 s).
    path = d / "gpu_util.csv"
    if not path.exists():
        return {}
    per: dict[int, list[float]] = defaultdict(list)
    for row in csv.reader(path.open()):
        if len(row) < 3 or not row[1].strip().isdigit():
            continue
        per[int(row[1])].append(float(row[2].strip().split()[0]))
    return {g: round(s.mean(v), 1) for g, v in per.items() if v}


def summarize(d: Path, slo_ms: float) -> dict:
    rows = [json.loads(l) for l in (d / "requests.jsonl").open()]
    wmap = {int(k): v for k, v in json.load((d / "worker_map.json").open()).items()}
    man = json.load((d / "manifest.json").open())
    ok = [r for r in rows if r.get("success")]
    res = [r for r in ok if r["turn"] > 0]
    cold = [r for r in ok if r["turn"] == 0]
    worker = {id(r): wmap.get(r.get("worker_id")) for r in rows}
    placed = [worker[id(r)] for r in rows]
    by_session: dict[int, list] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["turn"]):
        by_session[r["session"]].append(worker[id(r)])
    switches = sum(sum(1 for a, b in zip(w, w[1:]) if a != b) for w in by_session.values())
    sticky = sum(len(set(w)) == 1 for w in by_session.values())
    first = Counter(w[0] for w in by_session.values())
    fr = [r["cached_tokens"] / r["prompt_tokens"] for r in res
          if isinstance(r.get("cached_tokens"), int) and r.get("prompt_tokens")]
    wall = max(r["end_unix"] for r in rows) - min(r["start_unix"] for r in rows if "start_unix" in r)
    util = gpu_util(d)
    counts = Counter(placed)
    out = {
        "run": str(d), "arm": man["arm"], "rep": man["rep"], "completed": man.get("completed"),
        "requests": len(rows), "ok": len(ok), "placement_known": sum(p is not None for p in placed),
        "w0_requests": counts.get(0, 0), "w1_requests": counts.get(1, 0),
        "sessions_single_worker": sticky, "session_worker_switches": switches,
        "first_turn_w0_w1": f"{first.get(0, 0)}/{first.get(1, 0)}",
        "cached_coverage": f"{len(fr)}/{len(res)}",
        "resumed_cached_frac_median": round(s.median(fr), 3) if fr else None,
        "cold_ttft_median_ms": round(s.median(r["ttft_ms"] for r in cold)) if cold else None,
        "resumed_ttft_median_ms": round(s.median(r["ttft_ms"] for r in res)) if res else None,
        "resumed_ttft_p95_ms": (round(v) if (v := p95([r["ttft_ms"] for r in res])) is not None else None),
        "resumed_user_ttft_median_ms": round(s.median(r["user_ttft_ms"] for r in res)) if res else None,
        "wall_s": round(wall, 1),
        "goodput_rps_at_slo": round(sum(r["ttft_ms"] <= slo_ms for r in ok) / wall, 2),
        "prompt_digest": rows[0].get("prompt_set_sha256", "")[:12],
    }
    total = 0.0
    for i in range(int(man.get("workers", 2))):
        comp = delta(d, f"worker{i}", "vllm:request_prefill_kv_computed_tokens_sum")
        if comp is None or comp < 0:
            raise ValueError(f"{d}: missing or negative required prefill counter for worker{i}")
        out[f"w{i}_prefill_computed"] = comp
        out[f"w{i}_hit_tokens"] = delta(d, f"worker{i}", "vllm:prefix_cache_hits_total")
        out[f"w{i}_preemptions"] = delta(d, f"worker{i}", "vllm:num_preemptions_total")
        out[f"gpu{i}_util_mean"] = util.get(i)
        total += comp
    out["prefill_computed_total"] = total
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--slo-ms", type=float, default=3000)
    p.add_argument("--csv", type=Path)
    a = p.parse_args()
    # Routing arms are R*, single-worker controls are S*.
    runs = sorted({Path(x).parent for pat in ("R*", "S*")
                   for x in glob.glob(str(a.results / "rep_*" / pat / "requests.jsonl"))})
    out = [summarize(d, a.slo_ms) for d in runs]
    for r in out:
        print(json.dumps(r))
    if a.csv and out:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
