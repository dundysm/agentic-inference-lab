"""Build docs/demo/kv-cliff/data.json from the committed A40 cache-residency runs.

Usage: python scripts/build_kv_cliff_demo.py [results/2026-10-12-runpod-a40] [docs/demo/kv-cliff/data.json]

Every number the demo shows comes from this file: per-request timings and cached-token counts
from requests.jsonl, KV pool size from kv_allocation.log. The only modelled values are the
`toy_lru` curve, which replays the real request order (4.5 GiB, main rep 1) through a pure-LRU block cache.
"""
import glob, json, re, statistics as s, subprocess, sys
from collections import OrderedDict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/2026-10-12-runpod-a40"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/demo/kv-cliff/data.json"
BLOCK = 16  # vLLM's default block size; every resumed request at 4.5 GiB cached exactly one block


def _glob(pattern):
    return sorted(p.replace("\\", "/") for p in glob.glob(pattern))


def toy_lru(order, cap_tokens):
    """Resumed cached fraction under pure LRU, one request at a time. A model, not a measurement."""
    cap, lru, hit, tot = cap_tokens // BLOCK, OrderedDict(), 0, 0
    for sess, turn, prompt in order:
        keys = [("shared", 0)] + [(sess, i) for i in range(1, prompt // BLOCK)]
        h = 0
        for k in keys:
            if k not in lru:
                break
            h += 1
        for k in keys:
            if k in lru:
                lru.move_to_end(k)
            else:
                lru[k] = 1
                while len(lru) > cap:
                    lru.popitem(last=False)
        if turn > 0:
            hit, tot = hit + h * BLOCK, tot + prompt
    return hit / tot


def run(d, cohort):
    parts = d.split("/")
    budget, rep, arm = int(parts[-3]), int(parts[-2].split("_")[1]), parts[-1]
    rows = sorted((json.loads(l) for l in open(d + "/requests.jsonl")), key=lambda r: r["start_unix"])
    assert all(r["success"] for r in rows) and len(rows) == 192, d
    kv = int(re.search(r"GPU KV cache size: ([\d,]+)", open(d + "/kv_allocation.log").read()).group(1).replace(",", ""))
    man = json.load(open(d + "/workload_manifest.json"))
    t0 = man["measured_start_unix"]
    res = [r for r in rows if r["turn"] > 0]
    return {
        "cohort": cohort, "budget_bytes": budget, "gib": budget / 2**30, "rep": rep, "arm": arm,
        "kv_tokens": kv,
        "wall_s": round(man["measured_end_unix"] - t0, 2),
        "resumed_cached_frac_median": s.median(r["cached_tokens"] / r["prompt_tokens"] for r in res),
        "resumed_ttft_median_ms": round(s.median(r["ttft_ms"] for r in res), 3),
        "prefill_computed_tokens": sum(r["prompt_tokens"] - r["cached_tokens"] for r in rows),
        # session, turn, start ms, end ms, ttft ms, prompt tokens, cached tokens
        "requests": [[r["session"], r["turn"], round((r["start_unix"] - t0) * 1000), round((r["end_unix"] - t0) * 1000),
                      round(r["ttft_ms"]), r["prompt_tokens"], r["cached_tokens"]] for r in rows],
    }


runs = [run(d, "pilot") for d in _glob(f"{ROOT}/pilot-scan/pilot*/*/rep_1/C")]
runs += [run(d, "main") for d in _glob(f"{ROOT}/main/*/rep_*/[CD]")]
runs.sort(key=lambda r: (r["budget_bytes"], r["cohort"], r["arm"], r["rep"]))

ref = next(r for r in runs if r["cohort"] == "main" and r["arm"] == "C" and r["rep"] == 1 and r["gib"] == 4.5)
working_set = sum(max(q[5] for q in ref["requests"] if q[0] == sess) for sess in range(16))
order = [(q[0], q[1], q[5]) for q in ref["requests"]]
caps = list(range(8000, 80001, 500)) + [r["kv_tokens"] for r in runs]
curve = sorted({c: toy_lru(order, c) for c in caps}.items())

sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
data = {
    "provenance": {
        "source": ROOT, "git": sha, "gpu": "1x NVIDIA A40 (RunPod)", "driver": "570.211.01",
        "engine": "vllm==0.31.0+cu129", "model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": json.load(open(f"{ROOT}/main/4831838208/rep_1/C/server_config.json"))["model_revision"],
        "workload": "16 sessions x 12 turns, deterministic scripted history, 0.5 s tool pause, 4 in flight, 32 output tokens, temperature 0; not a real agent",
        "block_tokens": BLOCK, "working_set_tokens": working_set,
    },
    "toy_lru": [[c, round(f, 4)] for c, f in curve],
    "runs": runs,
}
with open(OUT, "w") as f:
    json.dump(data, f, separators=(",", ":"))
print(f"wrote {OUT}: {len(runs)} runs, working set {working_set} tokens, git {sha}")
