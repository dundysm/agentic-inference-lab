#!/usr/bin/env python3
"""Validate and summarize a locality-versus-load routing cohort without pooling GPU types."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as stats
from pathlib import Path

spec = importlib.util.spec_from_file_location("routing_metrics", Path(__file__).with_name("routing_analyze.py"))
ana = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ana)

ARMS = ("R0", "R1", "R1S4", "R2", "R3")
PROFILES = ("reference", "saturation", "hot-skew")


def quantile(samples: list[float], percentile: float) -> float:
    if not samples:
        raise ValueError("Quantile requested on empty sample")
    xs = sorted(samples)
    return xs[int((len(xs) - 1) * percentile)]


def queue_mean_ms(run: Path, worker: int) -> float:
    prefix = f"worker{worker}"
    total = ana.delta(run, prefix, "vllm:request_queue_time_seconds_sum")
    count = ana.delta(run, prefix, "vllm:request_queue_time_seconds_count")
    if total is None or count is None or count <= 0 or total < 0:
        raise ValueError(f"{run}: invalid/missing queue-time counter on worker{worker}")
    return 1000.0 * total / count


def summarize(run: Path) -> dict:
    x = ana.summarize(run, 500.0)  # fails if the required computed-prefill metric is absent
    m = json.loads((run / "manifest.json").read_text())
    wm = json.loads((run / "workload_manifest.json").read_text())
    rows = [json.loads(v) for v in (run / "requests.jsonl").read_text().splitlines() if v]
    expected = m["sessions"] * m["turns"]
    if not (m.get("completed") and wm["failures"] == 0 and
            len(rows) == x["ok"] == x["placement_known"] == expected and
            x["cached_coverage"] == f"{expected - m['sessions']}/{expected - m['sessions']}"):
        raise ValueError(f"{run}: incomplete responses, worker placement or cache token coverage")
    for field in ("schedule_sha256", "prompt_set_sha256"):
        if not wm.get(field):
            raise ValueError(f"{run}: missing {field}")
    if m["workers"] != 2:
        raise ValueError(f"{run}: expected exactly two workers")
    if any(r.get("ttft_ms") is None or r.get("user_ttft_ms") is None or
           r.get("completion_tokens") is None for r in rows):
        raise ValueError(f"{run}: incomplete latency or completion token fields")
    by_session = {}
    for r in rows:
        by_session.setdefault(r["session"], []).append(r)
    durations = []
    for session_rows in by_session.values():
        durations.append(max(r["end_unix"] for r in session_rows) -
                         min(r["request_ready_unix"] for r in session_rows))
    queue = [queue_mean_ms(run, i) for i in range(2)]
    out = {
        "run": str(run), "rep": m["rep"], "arm": m["arm"],
        "prompt_digest": wm["prompt_set_sha256"], "schedule_digest": wm["schedule_sha256"],
        "sessions": m["sessions"], "turns": m["turns"],
        "output_cap": wm["max_output_tokens"],
        "hot_count": wm["hot_session_count"],
        "prefill": x["prefill_computed_total"],
        "ttft_p95_ms": quantile([r["ttft_ms"] for r in rows if r["turn"]], .95),
        "user_ttft_p95_ms": quantile([r["user_ttft_ms"] for r in rows if r["turn"]], .95),
        "harness_wait_p95_ms": quantile([r["harness_wait_ms"] for r in rows], .95),
        "session_duration_p95_s": quantile(durations, .95),
        "median_output_tokens": stats.median(r["completion_tokens"] for r in rows),
        "wall_s": x["wall_s"], "queue_mean_ms": max(queue),
        "placement_skew": abs(x["w0_requests"] / expected - .5),
        "switches": x["session_worker_switches"],
        "hot_placement": None,
    }
    if m["hot_session_count"]:
        hot = [r for r in rows if r.get("hot_session")]
        if len(hot) != m["hot_session_count"] * m["turns"]:
            raise ValueError(f"{run}: missing hot-session labels")
        wm_ids = {int(k): v for k, v in json.loads((run / "worker_map.json").read_text()).items()}
        out["hot_placement"] = sum(wm_ids[r["worker_id"]] == 0 for r in hot) / len(hot)
    return out


def study(root: Path, reps: int, arms: tuple[str, ...] = ARMS, profiles: tuple[str, ...] = PROFILES) -> dict:
    report = {}
    for profile in profiles:
        values = {}
        for rep in range(1, reps + 1):
            cells = {}
            for arm in arms:
                run = root / profile / f"rep_{rep}" / arm
                if not (run / "requests.jsonl").exists():
                    raise ValueError(f"Missing preregistered cell: {run}")
                cells[arm] = summarize(run)
            if len({v["prompt_digest"] for v in cells.values()}) != 1 or len(
                   {v["schedule_digest"] for v in cells.values()}) != 1:
                raise ValueError(f"{profile}/rep_{rep}: arm workloads differ")
            values[rep] = cells
        report[profile] = values
    return report


def paired(values: dict, contender: str, metric: str, baseline: str = "R2") -> list[float]:
    return [arms[contender][metric] / arms[baseline][metric] for _, arms in sorted(values.items())]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help=".../locality-load/main or .../pilot")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.reps < 1:
        ap.error("--reps must be positive")
    values = study(a.root, a.reps)
    lines = [
        "# Locality versus load — measured cohort",
        "",
        f"Source: \`{a.root}\`. {a.reps} repetitions; each policy has the same "
        "model, host, session transcripts and timing schedule within a repetition.",
        "",
        "| Profile | Policy | Prefill vs affinity | Resumed TTFT p95 (ms) | Wall (s) | "
        "Queue (ms) | Placement skew | Output tokens median |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, reps in values.items():
        for arm in ARMS:
            v = [cells[arm] for cells in reps.values()]
            baseline = [cells["R2"]["prefill"] for cells in reps.values()]
            ratios = [x["prefill"] / b for x, b in zip(v, baseline)]
            lines.append(
                f"| {profile} | {arm} | {stats.median(ratios):.2f}× | "
                f"{stats.median(x['ttft_p95_ms'] for x in v):.1f} | "
                f"{stats.median(x['wall_s'] for x in v):.1f} | "
                f"{stats.median(x['queue_mean_ms'] for x in v):.2f} | "
                f"{stats.median(x['placement_skew'] for x in v):.2f} | "
                f"{stats.median(x['median_output_tokens'] for x in v):.0f} |"
            )
        if a.reps == 5:
            # Predeclared crossover: >=4/5 repetitions with >=10% better p95,
            # non-worse workload time, and observed queueing OR placement skew.
            for candidate in ("R0", "R1", "R1S4", "R3"):
                wins = sum(
                    int(cells[candidate]["ttft_p95_ms"] <= .9 * cells["R2"]["ttft_p95_ms"]
                        and cells[candidate]["wall_s"] <= cells["R2"]["wall_s"]
                        and (cells["R2"]["queue_mean_ms"] >= 5
                             or cells["R2"]["placement_skew"] >= .15))
                    for cells in reps.values()
                )
                lines.append(f"\n{profile}: {candidate} crossover criterion met in **{wins}/5** reps.")
        lines.append("")
    lines.append(
        "**Interpretation rule:** Zero qualifying profiles means no locality/load crossover "
        "was demonstrated. Check measured queueing, realized output lengths and skew before "
        "claiming that the hardware was saturated. These are repeated synthetic traces, "
        "not independent production traffic populations."
    )
    result = "\n".join(lines) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(result)
    print(result)


if __name__ == "__main__":
    main()
