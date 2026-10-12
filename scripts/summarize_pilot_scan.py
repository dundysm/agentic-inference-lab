"""Summarize the arm-C KV-budget scan. Usage: python scripts/summarize_pilot_scan.py results/<cohort-root>  (globs pilot*/<budget>/rep_1/C)."""
import glob, json, re, statistics as s, sys
from prometheus_client.parser import text_string_to_metric_families as parse

def metrics(path):
    v = {}
    for fam in parse(open(path).read()):
        for x in fam.samples:
            v[x.name] = v.get(x.name, 0) + x.value
    return v

root = sys.argv[1]
print("GiB  kv_tokens  ok/n     cachedFrac_med  >50%     hit_run  prefill_computed  preempt  ttft_cold  ttft_res_med/p95  user_ttft_res  completed")
dirs = sorted(glob.glob(f"{root}/pilot*/*/rep_1/C"), key=lambda p: int(p.split("/")[-3]))
for d in dirs:
    b = int(d.split("/")[-3])
    rows = [json.loads(l) for l in open(d + "/requests.jsonl")]
    kv = re.search(r"GPU KV cache size: ([\d,]+)", open(d + "/kv_allocation.log").read()).group(1)
    res = [r for r in rows if r["turn"] > 0 and r["success"]]
    cold = [r for r in rows if r["turn"] == 0 and r["success"]]
    fr = [r["cached_tokens"] / r["prompt_tokens"] for r in res]
    a, z = metrics(d + "/metrics_before.prom"), metrics(d + "/metrics_after.prom")
    dl = lambda k: z.get(k, 0) - a.get(k, 0)
    t = sorted(r["ttft_ms"] for r in res)
    p95 = t[int(0.95 * (len(t) - 1))]
    cfg = json.load(open(d + "/server_config.json"))
    hit = dl("vllm:prefix_cache_hits_total") / dl("vllm:prefix_cache_queries_total")
    print(f"{b/2**30:>4.1f} {kv:>10} {sum(r['success'] for r in rows):>4}/{len(rows):<4}"
          f" {s.median(fr):>10.3f} {sum(f > .5 for f in fr):>6}/{len(fr):<4}"
          f" {hit:>7.3f} {dl('vllm:request_prefill_kv_computed_tokens_sum'):>12.0f}"
          f" {dl('vllm:num_preemptions_total'):>7.0f} {s.median(r['ttft_ms'] for r in cold):>9.0f}"
          f" {s.median(t):>9.0f}/{p95:<6.0f} {s.median(r['user_ttft_ms'] for r in res):>10.0f}"
          f"     {cfg.get('run_completed_without_exception')}")

print("\nPer-turn median cached fraction (turns 1..11):")
for d in dirs:
    b = int(d.split("/")[-3])
    rows = [json.loads(l) for l in open(d + "/requests.jsonl")]
    line = []
    for turn in range(1, 12):
        f = [r["cached_tokens"] / r["prompt_tokens"] for r in rows if r["turn"] == turn and r["success"]]
        line.append(f"{s.median(f):.2f}")
    print(f"{b/2**30:>4.1f} GiB: " + " ".join(line))
