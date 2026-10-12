"""Summarize a one-rep A-D pilot with paired cache ON/OFF ratios. Usage: python scripts/summarize_ab_pilot.py results/<...>/ab-pilot"""
import glob, json, re, statistics as s, sys
from prometheus_client.parser import text_string_to_metric_families as parse


def _glob(pattern):
    # Normalize separators so path-part parsing works on Windows too.
    return [p.replace("\\", "/") for p in glob.glob(pattern)]

def metrics(path):
    v = {}
    for fam in parse(open(path).read()):
        for x in fam.samples:
            v[x.name] = v.get(x.name, 0) + x.value
    return v

root = sys.argv[1]
out = {}
print("GiB   arm cache sess  ok/n     cachedFrac  prefill_computed  preempt  ttft_cold  ttft_res_med/p95  user_ttft_res  wall_s  goodput@3s")
for d in sorted(_glob(f"{root}/*/rep_1/*"), key=lambda p: (int(p.split("/")[-3]), p[-1])):
    b, arm = int(d.split("/")[-3]), d[-1]
    rows = [json.loads(l) for l in open(d + "/requests.jsonl")]
    cfg = json.load(open(d + "/server_config.json"))
    res = [r for r in rows if r["turn"] > 0 and r["success"]]
    cold = [r for r in rows if r["turn"] == 0 and r["success"]]
    fr = [r["cached_tokens"] / r["prompt_tokens"] for r in res if isinstance(r.get("cached_tokens"), int)]
    a, z = metrics(d + "/metrics_before.prom"), metrics(d + "/metrics_after.prom")
    dl = lambda k: z.get(k, 0) - a.get(k, 0)
    t = sorted(r["ttft_ms"] for r in res)
    wall = max(r["end_unix"] for r in rows) - min(r["start_unix"] for r in rows)
    good = sum(r["ttft_ms"] <= 3000 for r in rows if r["success"]) / wall
    out[(b, arm)] = dict(res_med=s.median(t), prefill=dl("vllm:request_prefill_kv_computed_tokens_sum"), user=s.median(r["user_ttft_ms"] for r in res), wall=wall)
    print(f"{b/2**30:>4.1f}  {arm}   {'ON ' if cfg['prefix_cache_enabled'] else 'OFF'}  {cfg['sessions']:>3}  {sum(r['success'] for r in rows):>3}/{len(rows):<4}"
          f" {(f'{s.median(fr):.3f}' if fr else 'n/a'):>9} {dl('vllm:request_prefill_kv_computed_tokens_sum'):>14.0f}"
          f" {dl('vllm:num_preemptions_total'):>7.0f} {s.median(r['ttft_ms'] for r in cold):>9.0f}"
          f" {s.median(t):>9.0f}/{t[int(.95*(len(t)-1))]:<6.0f} {out[(b, arm)]['user']:>10.0f} {wall:>8.1f} {good:>8.2f}")

print("\nPaired cache ON minus OFF (same budget):")
for b in sorted({k[0] for k in out}):
    for on, off in (("B", "A"), ("C", "D")):
        if (b, on) in out and (b, off) in out:
            x, y = out[(b, on)], out[(b, off)]
            print(f"{b/2**30:>4.1f} GiB {on}-{off}: resumed TTFT {x['res_med']-y['res_med']:+7.0f} ms ({x['res_med']/y['res_med']:.2f}x), "
                  f"prefill computed {x['prefill']/y['prefill']:.2f}x, user TTFT {x['user']-y['user']:+7.0f} ms, wall {x['wall']:.1f}s vs {y['wall']:.1f}s")
