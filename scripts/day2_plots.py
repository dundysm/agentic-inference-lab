#!/usr/bin/env python3
"""Day 2 figures (guide section 10) and Tables A/B, from committed raw data. CPU only.

Usage: python scripts/day2_plots.py results/2026-10-12-runpod-2x-rtxpro4500/day2-main OUT_DIR
Writes four PNGs and tables.md into OUT_DIR.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import statistics as s
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

spec = importlib.util.spec_from_file_location("day2_analyze", Path(__file__).with_name("day2_analyze.py"))
ana = importlib.util.module_from_spec(spec); spec.loader.exec_module(ana)

ARMS = ["R0", "R1", "R2", "R3"]
LABEL = {"R0": "R0 round robin", "R1": "R1 KV-aware", "R2": "R2 hard affinity", "R3": "R3 soft affinity",
         "R1S4": "R1S4 KV, prefill x4"}
# Categorical slots 1-4 of the validated reference palette (light mode); markers give secondary encoding.
COLOR = {"R0": "#2a78d6", "R1": "#eb6834", "R2": "#1baf7a", "R3": "#eda100",
         "R1S4": "#4a3aa7"}  # violet: magenta failed the normal-vision floor next to orange
MARKER = {"R0": "o", "R1": "s", "R2": "^", "R3": "D", "R1S4": "v"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
WORKER = {0: "#3b3a37", 1: "#bdbbb3"}   # neutral pair: workers are not arms

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(root: Path):
    runs = {}
    for d in sorted(p.parent for p in root.glob("rep_*/R*/requests.jsonl")):
        summary = ana.summarize(d, 3000)
        rows = [json.loads(l) for l in (d / "requests.jsonl").open()]
        wmap = {int(k): v for k, v in json.load((d / "worker_map.json").open()).items()}
        for r in rows:
            r["worker"] = wmap.get(r.get("worker_id"))
            r["frac"] = r["cached_tokens"] / r["prompt_tokens"]
        runs[(summary["rep"], summary["arm"])] = dict(summary=summary, rows=rows, dir=d)
    return runs


def finish(fig, path: Path, note: str):
    fig.text(0.01, -0.03, note, fontsize=8, color=MUTED, ha="left", va="top")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_reuse_by_turn(runs, reps, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    turns = range(1, 12)
    for arm in ARMS:
        per_rep = []
        for rep in reps:
            rows = runs[(rep, arm)]["rows"]
            per_rep.append([s.mean(r["frac"] for r in rows if r["turn"] == t) for t in turns])
            ax.plot(turns, per_rep[-1], color=COLOR[arm], alpha=0.18, linewidth=1)
        med = [s.median(v[i] for v in per_rep) for i in range(len(turns))]
        ax.plot(turns, med, color=COLOR[arm], linewidth=2, marker=MARKER[arm], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[arm],
                linestyle="--" if arm == "R3" else "-", zorder=4 if arm == "R3" else 3)
    ax.set_ylim(0, 1.1); ax.set_xticks(list(turns))
    ax.set_xlabel("Turn (resumed turns; turn 0 is always cold)")
    ax.set_ylabel("Mean cached fraction of prompt tokens")
    ax.set_title("Resumed-turn prefix reuse by routing arm")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    if "R3" in ARMS and "R2" in ARMS:
        ax.annotate("R2 and R3 overlap (identical reuse)", (6, 0.99), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK2)
    finish(fig, out / "day2_reuse_by_turn.png",
           "Bold line: median across 5 reps of the per-turn mean over 16 sessions; faint lines: individual reps. "
           "Mean, not median: medians hide the misses.")


def plot_ttft(runs, reps, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = random.Random(0)
    for i, arm in enumerate(ARMS):
        xs = [r["ttft_ms"] for rep in reps for r in runs[(rep, arm)]["rows"] if r["turn"] > 0 and r["success"]]
        ax.scatter([i + rng.uniform(-0.28, 0.28) for _ in xs], xs, s=6, color=MUTED, alpha=0.15, linewidths=0)
        p50 = [runs[(rep, arm)]["summary"]["resumed_ttft_median_ms"] for rep in reps]
        p95 = [runs[(rep, arm)]["summary"]["resumed_ttft_p95_ms"] for rep in reps]
        ax.scatter([i - 0.08] * len(p50), p50, s=60, marker=MARKER[arm], facecolors=SURFACE,
                   edgecolors=COLOR[arm], linewidths=2, zorder=3, label=None)
        ax.scatter([i + 0.08] * len(p95), p95, s=60, marker=MARKER[arm], color=COLOR[arm],
                   edgecolors=SURFACE, linewidths=1.5, zorder=3, label=None)
        ax.annotate(f"p95 {s.median(p95):.0f} ms", (i + 0.14, s.median(p95)), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS]); ax.set_yscale("log")
    ax.set_yticks([20, 30, 50, 100, 200, 300, 500], ["20", "30", "50", "100", "200", "300", "500"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Resumed-turn TTFT, client-observed (ms, log scale)")
    ax.set_title("Resumed-turn latency by routing arm")
    ax.scatter([], [], s=60, marker="o", facecolors=SURFACE, edgecolors=INK2, linewidths=2, label="per-rep median")
    ax.scatter([], [], s=60, marker="o", color=INK2, label="per-rep p95")
    ax.legend(frameon=False, loc="upper right")
    finish(fig, out / "day2_resumed_ttft.png",
           "Grey: every resumed request across 5 reps (176 per run). Hollow: per-rep median; filled: per-rep p95. "
           "Includes the Dynamo frontend; compare within Day 2 only.")


def plot_prefill(runs, reps, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, arm in enumerate(ARMS):
        w0 = s.median(runs[(rep, arm)]["summary"]["w0_prefill_computed"] for rep in reps) / 1000
        w1 = s.median(runs[(rep, arm)]["summary"]["w1_prefill_computed"] for rep in reps) / 1000
        ax.bar(i, w0, width=0.55, color=WORKER[0], edgecolor=SURFACE, linewidth=2,
               label="worker 0 (GPU 0)" if i == 0 else None)
        ax.bar(i, w1, bottom=w0, width=0.55, color=WORKER[1], edgecolor=SURFACE, linewidth=2,
               label="worker 1 (GPU 1)" if i == 0 else None)
        totals = [runs[(rep, arm)]["summary"]["prefill_computed_total"] / 1000 for rep in reps]
        ax.annotate(f"{s.median(totals):.0f}k\n({min(totals):.0f}–{max(totals):.0f}k)", (i, w0 + w1),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
    ax.set_ylabel("Recomputed prefill tokens per run (thousands)")
    ax.set_title("Recomputed prefill per worker (median of 5 reps)")
    ax.margins(y=0.18); ax.grid(axis="x", visible=False); ax.legend(frameon=False, loc="upper right")
    shares = [runs[k]["summary"]["w0_requests"] / runs[k]["summary"]["requests"] for k in runs]
    split = ("Requests were split 96/96 in every run." if all(abs(x - 0.5) < 1e-9 for x in shares) else
             f"Worker-0 request share ranged {min(shares):.0%}-{max(shares):.0%} across runs.")
    finish(fig, out / "day2_prefill_per_worker.png",
           "Worker vLLM counter deltas (vllm:request_prefill_kv_computed_tokens_sum). Bars stack each worker's median; "
           "label is the median total and range across reps. " + split)


def plot_placement(runs, rep, out):
    fig, axes = plt.subplots(1, len(ARMS), figsize=(13, 5), sharey=True)
    for ax, arm in zip(axes, ARMS):
        rows = runs[(rep, arm)]["rows"]
        by = {(r["session"], r["turn"]): r for r in rows}
        moves = 0
        for sess in range(16):
            for t in range(12):
                r = by[(sess, t)]
                ax.add_patch(plt.Rectangle((t, sess), 0.92, 0.88, color=WORKER[r["worker"]], linewidth=0))
                if t and r["frac"] < 0.5:
                    ax.plot(t + 0.46, sess + 0.44, marker="x", color="#ffffff" if r["worker"] == 0 else INK,
                            markersize=5, markeredgewidth=1.4)
                moves += t > 0 and r["worker"] != by[(sess, t - 1)]["worker"]
        ax.set_xlim(0, 12); ax.set_ylim(16, 0); ax.grid(False)
        ax.set_xticks([0.46, 5.46, 11.46], ["0", "5", "11"]); ax.set_xlabel("turn")
        ax.set_title(f"{LABEL[arm]}\n{moves} worker moves", fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("session"); axes[0].set_yticks([0.44, 5.44, 10.44, 15.44], ["0", "5", "10", "15"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=WORKER[0]), plt.Rectangle((0, 0), 1, 1, color=WORKER[1]),
               plt.Line2D([], [], marker="x", color=INK, linestyle="", markersize=6)]
    fig.legend(handles, ["served by worker 0", "served by worker 1", "resumed cache miss (<50% cached)"],
               frameon=False, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.04))
    fig.suptitle(f"Session-to-worker placement, rep {rep}", x=0.01, ha="left", fontweight="bold", y=1.04)
    misses = still = 0
    for arm in ARMS:
        by = {(r["session"], r["turn"]): r for r in runs[(rep, arm)]["rows"]}
        for (sess, t), r in by.items():
            if t and r["frac"] < 0.5:
                misses += 1; still += r["worker"] == by[(sess, t - 1)]["worker"]
    note = ("Every miss is on a turn where the session changed worker." if still == 0 else
            f"{misses - still} of {misses} misses are on turns where the session changed worker; "
            f"{still} occurred on the same worker.")
    finish(fig, out / "day2_session_placement.png", "Each cell is one request. " + note)


def kv_tokens(d: Path, i: int) -> str:
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", (d / f"worker{i}.log").read_text(errors="ignore"))
    return m.group(1) if m else "?"


def tables(runs, reps, out):
    lines = ["## Table A: paired comparison against R0 (per rep, then median [min, max])", "",
             "| Arm vs R0 | Recomputed prefill ratio | Resumed p95 ratio | Resumed median TTFT Δ (ms) | "
             "Resumed mean cached fraction Δ | Wall ratio | Goodput@3 s ratio |", "|---|---|---|---|---|---|---|"]
    def span(xs, fmt):
        return f"{fmt.format(s.median(xs))} [{fmt.format(min(xs))}, {fmt.format(max(xs))}]"
    mean_frac = lambda rep, arm: s.mean(r["frac"] for r in runs[(rep, arm)]["rows"] if r["turn"] > 0)
    for arm in [a for a in ARMS if a != "R0"]:
        g = lambda rep, k: runs[(rep, arm)]["summary"][k] / runs[(rep, "R0")]["summary"][k]
        d = lambda rep, k: runs[(rep, arm)]["summary"][k] - runs[(rep, "R0")]["summary"][k]
        lines.append(" | ".join([
            f"| {LABEL[arm]}",
            span([g(r, "prefill_computed_total") for r in reps], "{:.2f}×"),
            span([g(r, "resumed_ttft_p95_ms") for r in reps], "{:.2f}×"),
            span([d(r, "resumed_ttft_median_ms") for r in reps], "{:+.0f}"),
            span([mean_frac(r, arm) - mean_frac(r, "R0") for r in reps], "{:+.3f}"),
            span([g(r, "wall_s") for r in reps], "{:.2f}×"),
            span([g(r, "goodput_rps_at_slo") for r in reps], "{:.2f}×")]) + " |")
    lines += ["", "Resumed mean cached fraction for R0 itself (median of reps): "
              f"{s.median(mean_frac(r, 'R0') for r in reps):.3f}.", "",
              "## Table B: validity", "",
              "| Run | Requests ok | Placement known | Prompt digest | Preemptions (w0/w1) | Requests (w0/w1) | "
              "KV tokens (w0/w1) | Cached-token coverage | Mean queue time ms (w0/w1) | GPU util % (0/1) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for rep in reps:
        for arm in ARMS:
            x, d = runs[(rep, arm)]["summary"], runs[(rep, arm)]["dir"]
            q = []
            for i in (0, 1):
                tot = ana.delta(d, f"worker{i}", "vllm:request_queue_time_seconds_sum")
                cnt = ana.delta(d, f"worker{i}", "vllm:request_queue_time_seconds_count")
                q.append(f"{1000 * tot / cnt:.1f}" if tot is not None and cnt else "n/a")
            lines.append(f"| rep {rep} {arm} | {x['ok']}/{x['requests']} | {x['placement_known']}/{x['requests']} | "
                         f"{x['prompt_digest']} | {x['w0_preemptions']:.0f}/{x['w1_preemptions']:.0f} | "
                         f"{x['w0_requests']}/{x['w1_requests']} | {kv_tokens(d, 0)}/{kv_tokens(d, 1)} | "
                         f"{x['cached_coverage']} | {q[0]}/{q[1]} | {x['gpu0_util_mean']}/{x['gpu1_util_mean']} |")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global ARMS
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) > 3:  # optional comma-separated arm list, e.g. R0,R1,R1S4,R2
        ARMS = sys.argv[3].split(",")
    out.mkdir(parents=True, exist_ok=True)
    runs = load(root)
    reps = sorted({k[0] for k in runs})
    plot_reuse_by_turn(runs, reps, out)
    plot_ttft(runs, reps, out)
    plot_prefill(runs, reps, out)
    plot_placement(runs, 1, out)
    tables(runs, reps, out)
    print(f"wrote 4 figures and tables.md to {out}")


if __name__ == "__main__":
    main()
