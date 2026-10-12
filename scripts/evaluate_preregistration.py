"""Evaluate the main cohort against docs/preregistration-main-2026-10-12.md (H1-H3). Usage: python scripts/evaluate_preregistration.py results/<...>/main"""
import glob, json, statistics as s, sys
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
R = {}
for d in _glob(f"{root}/*/rep_*/*"):
    if not d[-1] in "ABCD":
        continue
    b, rep, arm = int(d.split("/")[-3]), int(d.split("/")[-2][4:]), d[-1]
    rows = [json.loads(l) for l in open(d + "/requests.jsonl")]
    cfg = json.load(open(d + "/server_config.json"))
    res = [r for r in rows if r["turn"] > 0 and r["success"]]
    a, z = metrics(d + "/metrics_before.prom"), metrics(d + "/metrics_after.prom")
    dl = lambda k: z.get(k, 0) - a.get(k, 0)
    wall = max(r["end_unix"] for r in rows) - min(r["start_unix"] for r in rows)
    R[(b, arm, rep)] = dict(
        ok=sum(r["success"] for r in rows), n=len(rows), completed=cfg.get("run_completed_without_exception"),
        frac=s.median(r["cached_tokens"] / r["prompt_tokens"] for r in res),
        ttft=s.median(r["ttft_ms"] for r in res), user=s.median(r["user_ttft_ms"] for r in res),
        prefill=dl("vllm:request_prefill_kv_computed_tokens_sum"), preempt=dl("vllm:num_preemptions_total"),
        wall=wall, sha=rows[0]["prompt_set_sha256"], baseline=cfg["baseline_gpu_mib"])

budgets = sorted({k[0] for k in R}); reps = sorted({k[2] for k in R})
print(f"runs={len(R)}  all_ok={all(v['ok']==v['n'] for v in R.values())}  all_completed={all(v['completed'] for v in R.values())}"
      f"  total_preemptions={sum(v['preempt'] for v in R.values()):.0f}  failed_requests={sum(v['n']-v['ok'] for v in R.values())}")
for arm in "ABCD":
    print(f"prompt_set_sha256 arm {arm}: {sorted({v['sha'][:12] for k,v in R.items() if k[1]==arm})}")

def span(xs, f="{:.0f}"):
    return f"{f.format(s.median(xs))} [{f.format(min(xs))}, {f.format(max(xs))}]"

print("\nPer-arm, median [min, max] across 5 reps of the per-run medians:")
print("GiB  arm   cachedFrac              resumedTTFT_ms        userTTFT_ms              prefill_computed            wall_s")
for b in budgets:
    for arm in "ABCD":
        v = [R[(b, arm, r)] for r in reps]
        print(f"{b/2**30:>4.1f}  {arm}  {span([x['frac'] for x in v], '{:.3f}'):>22} {span([x['ttft'] for x in v]):>20} "
              f"{span([x['user'] for x in v]):>22} {span([x['prefill'] for x in v]):>28} {span([x['wall'] for x in v], '{:.1f}'):>20}")

print("\nPaired ON/OFF per repetition (ratio of resumed TTFT medians; wall ratio):")
for b in budgets:
    for on, off in (("B", "A"), ("C", "D")):
        rt = [R[(b, on, r)]["ttft"] / R[(b, off, r)]["ttft"] for r in reps]
        wt = [R[(b, on, r)]["wall"] / R[(b, off, r)]["wall"] for r in reps]
        pf = [R[(b, on, r)]["prefill"] / R[(b, off, r)]["prefill"] for r in reps]
        print(f"{b/2**30:>4.1f} GiB {on}/{off}: TTFT ratio {['%.2f' % x for x in rt]}  wall ratio {['%.2f' % x for x in wt]}  prefill ratio {['%.2f' % x for x in pf]}")

lo, hi = budgets
H1 = all(R[(b, "B", r)]["ttft"] < R[(b, "A", r)]["ttft"] and R[(b, "B", r)]["prefill"] < R[(b, "A", r)]["prefill"] for b in budgets for r in reps)
cdl = [R[(lo, "C", r)]["ttft"] / R[(lo, "D", r)]["ttft"] - 1 for r in reps]
H2 = all(R[(lo, "C", r)]["frac"] < 0.05 for r in reps) and abs(s.median(cdl)) <= 0.10
H3 = all(R[(hi, "C", r)]["frac"] > 0.9 for r in reps) and \
     s.median([R[(hi, "C", r)]["ttft"] / R[(hi, "D", r)]["ttft"] for r in reps]) < 0.5 and \
     all(R[(hi, "C", r)]["wall"] < R[(hi, "D", r)]["wall"] for r in reps)
print(f"\nH1 (B<A TTFT & prefill, both budgets, every rep): {'SUPPORTED' if H1 else 'NOT SUPPORTED'}")
print(f"H2 (4.5GiB: C frac<0.05 every rep, median C/D TTFT within +-10%): {'SUPPORTED' if H2 else 'NOT SUPPORTED'}  (median C/D-1 = {s.median(cdl):+.3f}, per rep {['%+.3f' % x for x in cdl]})")
print(f"H3 (10GiB: C frac>0.9 every rep, median C/D TTFT<0.5, C wall<D every rep): {'SUPPORTED' if H3 else 'NOT SUPPORTED'}")
