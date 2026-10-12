#!/usr/bin/env python3
"""Day 1 benchmark analysis; descriptive results, no spurious confidence intervals."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def samples(path: Path) -> dict[str, float]:
    from prometheus_client.parser import text_string_to_metric_families

    values: dict[str, float] = {}
    for family in text_string_to_metric_families(path.read_text()):
        for sample in family.samples:
            if sample.name.endswith(("_bucket", "_created")):
                continue
            values[sample.name] = values.get(sample.name, 0.0) + float(sample.value)
    return values


def diff(before: dict[str, float], after: dict[str, float], *names: str) -> float | None:
    for name in names:
        if name in before and name in after:
            val = after[name] - before[name]
            return val if val >= 0 else None
    return None


def median(data: list[float]) -> float | None:
    return statistics.median(data) if data else None


def pctl(data: list[float], p: float) -> float | None:
    if not data:
        return None
    s = sorted(data)
    at = (len(s) - 1) * p
    lo = math.floor(at)
    hi = math.ceil(at)
    return s[lo] + (s[hi] - s[lo]) * (at - lo)


def frac(row: dict) -> float | None:
    cached, total = row.get("cached_tokens"), row.get("prompt_tokens")
    if isinstance(cached, int) and isinstance(total, int) and total > 0 and 0 <= cached <= total:
        return cached / total
    return None


def analyze_run(path: Path, slo: float) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines() if line.strip()]
    pre = samples(path / "metrics_before.prom")
    post = samples(path / "metrics_after.prom")
    manifest = json.loads((path / "workload_manifest.json").read_text())
    config = json.loads((path / "server_config.json").read_text())
    valid = [r for r in records if r.get("success") is True and isinstance(r.get("ttft_ms"), (int, float))]
    resumed = [r for r in valid if r["turn"] > 0]
    cold = [r for r in valid if r["turn"] == 0]
    duration = float(manifest["measured_end_unix"]) - float(manifest["measured_start_unix"])
    hits = diff(pre, post, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
    queries = diff(pre, post, "vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
    computed = diff(pre, post, "vllm:request_prefill_kv_computed_tokens_sum")
    cached_values = [frac(r) for r in resumed]
    cached_observed = [f for f in cached_values if f is not None]
    if cached_observed and len(cached_observed) == len(resumed):
        computed_from_usage = sum(int(r["prompt_tokens"]) - int(r["cached_tokens"]) for r in resumed)
    else:
        computed_from_usage = None
    errors = len(records) - len(valid)
    completed = bool(config.get("run_completed_without_exception")) and errors == 0 and len(records) == manifest["expected_requests"]
    row = {
        "run": str(path), "budget_bytes": config["kv_cache_memory_bytes"],
        "arm": manifest["scenario"], "repetition": config["rep"],
        "cache_enabled": config["prefix_cache_enabled"], "model_revision": config.get("model_revision"),
        "request_count": len(records), "success_count": len(valid), "errors": errors,
        "complete": completed, "resumed_n": len(resumed),
        "cold_ttft_median_ms": median([r["ttft_ms"] for r in cold]),
        "resumed_ttft_median_ms": median([r["ttft_ms"] for r in resumed]),
        "resumed_ttft_p95_ms": pctl([r["ttft_ms"] for r in resumed], .95) if len(resumed) >= 50 else None,
        "user_ttft_median_ms": median([r["user_ttft_ms"] for r in resumed if isinstance(r.get("user_ttft_ms"), (int, float))]),
        "harness_wait_median_ms": median([r["harness_wait_ms"] for r in resumed if isinstance(r.get("harness_wait_ms"), (int, float))]),
        "hit_ratio_run": hits / queries if config["prefix_cache_enabled"] and hits is not None and queries and queries > 0 else None,
        "cache_fraction_resumed_median": median(cached_observed) if len(cached_observed) == len(resumed) and resumed else None,
        "cache_fraction_coverage": len(cached_observed) / len(resumed) if resumed else None,
        "prefill_computed_tokens_total": computed,
        "resumed_computed_tokens_from_usage": computed_from_usage,
        "prefill_computed_tokens_per_s": computed / duration if computed is not None and duration > 0 else None,
        "goodput_rps_at_slo": sum(r["ttft_ms"] <= slo for r in valid) / duration if duration > 0 else None,
        "user_goodput_rps_at_slo": sum(r.get("user_ttft_ms", float("inf")) <= slo for r in valid) / duration if duration > 0 else None,
        "elapsed_s": duration,
    }
    return row, records


def csv_output(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(rows: list[dict]) -> list[dict]:
    indexed = {(r["budget_bytes"], r["repetition"], r["arm"]): r for r in rows if r["complete"]}
    comparisons = []
    for budget, rep in sorted({(r["budget_bytes"], r["repetition"]) for r in rows}):
        for off, on in (("A", "B"), ("D", "C")):
            a = indexed.get((budget, rep, off))
            b = indexed.get((budget, rep, on))
            if not a or not b:
                continue
            def delta(k: str):
                if a.get(k) is None or b.get(k) is None:
                    return None
                return b[k] - a[k]
            comparisons.append({
                "budget_bytes": budget, "repetition": rep,
                "comparison": f"{on} cache ON minus {off} cache OFF",
                "ttft_median_delta_ms": delta("resumed_ttft_median_ms"),
                "user_ttft_median_delta_ms": delta("user_ttft_median_ms"),
                "computed_prefill_tokens_delta": delta("prefill_computed_tokens_total"),
                "goodput_rps_delta": delta("goodput_rps_at_slo"),
            })
    return comparisons


def plot_results(output: Path, rows: list[dict], points: list[tuple[dict, dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Summary plots treat entire run as unit of replication; raw per-request dots are context only.
    budgets = sorted({r["budget_bytes"] for r in rows})
    fig, ax = plt.subplots(figsize=(12, 6))
    ticks, labels = [], []
    for bi, budget in enumerate(budgets):
        for ai, arm in enumerate("ABCD"):
            x = bi * 5 + ai
            ticks.append(x)
            labels.append(f"{budget / 2**30:g}GiB / {arm}")
            raw = [r["ttft_ms"] for s, r in points if s["budget_bytes"] == budget and s["arm"] == arm and r["turn"] > 0]
            if raw:
                ax.scatter([x] * len(raw), raw, s=6, alpha=.1)
            per_rep = [s["resumed_ttft_median_ms"] for s in rows if s["budget_bytes"] == budget and s["arm"] == arm and s["complete"] and s["resumed_ttft_median_ms"] is not None]
            if per_rep:
                ax.scatter([x] * len(per_rep), per_rep, marker="D", s=40, zorder=3)
                ax.plot(x, median(per_rep), "ko", markersize=5)
    ax.set_xticks(ticks, labels, rotation=45, ha="right")
    ax.set_ylabel("First-content-token latency, resumed turns (ms)")
    ax.set_title("Run-level medians (diamonds); faint dots are dependent individual turns")
    fig.tight_layout()
    fig.savefig(output / "resumed_ttft.png", dpi=160)
    plt.close(fig)

    # PRIMARY evidence: cache fraction by turn index, with independent repetition medians.
    grouped: dict[tuple[int, str, int, int], list[float]] = defaultdict(list)
    for s, r in points:
        v = frac(r)
        if v is not None and r["turn"] >= 1:
            grouped[(s["budget_bytes"], s["arm"], s["repetition"], r["turn"])].append(v)
    fig, ax = plt.subplots(figsize=(12, 6))
    for budget in budgets:
        for arm in "ABCD":
            # Do not plot missing cached-token data as zero reuse.
            rep_turns = {(rep, turn): median(v) for (b, a, rep, turn), v in grouped.items() if b == budget and a == arm}
            if not rep_turns:
                continue
            by_turn: dict[int, list[float]] = defaultdict(list)
            for (rep, turn), val in rep_turns.items():
                if val is not None:
                    by_turn[turn].append(val)
                    ax.scatter(turn, val, s=12, alpha=.3)
            turns = sorted(by_turn)
            ax.plot(turns, [median(by_turn[t]) for t in turns], marker="o", label=f"{budget/2**30:g}GiB / {arm}")
    ax.set_ylim(-.05, 1.05)
    ax.set_xlabel("Turn index (0 = cold first turn; plotted from resumed turn 1)")
    ax.set_ylabel("Cached prompt tokens / total prompt tokens")
    ax.set_title("Primary evidence: resumed-turn prefix reuse (per-repetition medians)")
    if ax.lines:
        ax.legend(fontsize="small", ncol=2)
    else:
        ax.text(.5, .5, "No valid per-request cached-token data; metric unavailable", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output / "resumed_cache_fraction_by_turn.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    count = 0
    for arm in "BC":
        values = [r for r in rows if r["arm"] == arm and r["complete"] and r["hit_ratio_run"] is not None]
        if values:
            ax.scatter([r["budget_bytes"] / 2**30 for r in values], [r["hit_ratio_run"] for r in values], label=f"arm {arm}")
            count += 1
    ax.set_ylim(-.05, 1.05)
    ax.set_xlabel("Allocated KV cache (GiB)")
    ax.set_ylabel("Aggregate prefix cache hit-token ratio")
    ax.set_title("Secondary evidence: server run-level prefix cache hit ratio")
    if count:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cache_reuse.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--slo-ms", type=float, default=3000, help="Choose before collecting results")
    args = p.parse_args()
    result_paths = sorted(args.results.glob("*/rep_*/*/requests.jsonl"))
    if not result_paths:
        p.error("No results under <results>/<budget>/rep_N/<arm>/requests.jsonl")
    rows: list[dict] = []
    points: list[tuple[dict, dict]] = []
    for path in result_paths:
        try:
            summary, records = analyze_run(path.parent, args.slo_ms)
        except Exception as exc:
            print(f"SKIP {path.parent}: {type(exc).__name__}: {exc}")
            continue
        rows.append(summary)
        points.extend((summary, r) for r in records if r.get("success") is True and r.get("ttft_ms") is not None)
    if not rows:
        raise SystemExit("No analyzable runs")
    output = args.results / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    csv_output(output / "summary.csv", rows)
    csv_output(output / "paired_comparisons.csv", paired_comparisons(rows))
    plot_results(output, rows, points)
    print(f"Analyzed {len(rows)} runs ({sum(r['complete'] for r in rows)} complete). Files: {output}")
    print("Missing cached_tokens remains missing; zero is never substituted. No GPU validation is implied.")


if __name__ == "__main__":
    main()
