# agentic-inference-lab

**Question:** Agent sessions pause between turns while tools run. Does vLLM's automatic prefix cache keep their long contexts alive across those pauses, once the paused sessions add up to more than the GPU KV cache?

**Answer (measured, single GPU):** No, and the effect is all-or-nothing. With 16 paused sessions (~54K tokens of prefixes) on a 4.5 GiB KV budget (32,768 tokens), prefix caching gave **zero benefit**: cache ON and OFF matched on every metric across 5 repetitions. That held even though the cache could hold about 60% of the prefixes. With a 10 GiB budget, the same workload got about **8× faster resumed-turn TTFT** and finished in **42% of the wall time**. There were no preemptions anywhere, so `kv_cache_usage_perc` and `num_preemptions` do not show this failure. Per-request `cached_tokens` does.

![Resumed-turn prefix reuse per turn](results/2026-10-12-runpod-a40/main/analysis/resumed_cache_fraction_by_turn.png)

| 16 sessions, cache ON vs OFF | 4.5 GiB (tight) | 10 GiB (roomy) |
|---|---|---|
| Resumed-turn cached fraction | 0.005 | 0.989 |
| Resumed TTFT ratio (median of 5 reps) | 0.98× | 0.13× |
| Wall-time ratio | 1.00× | 0.42× |

The likely mechanism is LRU eviction under near-round-robin session access. Once the working set exceeds capacity, the block evicted next is the one needed soonest. Details, uncertainty and limitations are in **[docs/findings.md](docs/findings.md)**. The hypotheses were **[pre-registered](docs/preregistration-main-2026-10-12.md)** before the main cohort.

## What was measured
- **Hardware and software:** 1× NVIDIA A40 on RunPod, vLLM 0.31.0 (cu129 build), `Qwen/Qwen3-4B-Instruct-2507` at a pinned revision, BF16, max length 8192.
- **Workload:** deterministic scripted sessions with unique ~3K-token documents, 12 turns each, a 0.5 s tool pause, and an explicit KV budget set with `--kv-cache-memory-bytes`.
- **Four arms:** cache ON/OFF × 4/16 sessions, with a fresh server per run and the budget/arm order interleaved.
- **Pilot scan:** reuse is 0.005 at 2.5, 3, 4.5 and 6 GiB and 0.989 at 8 GiB, so the cliff lies between 43.7K and 58.2K tokens of capacity.
- **Main cohort:** 40 runs and 7,680 requests, with 0 failures and 0 preemptions.

This is a synthetic, closed-loop, single-GPU experiment. It does not claim anything about production agent traffic.

## Reproduce
- **Re-derive every table from the committed raw data** (CPU only):
  ```bash
  pip install -r requirements.txt
  python scripts/evaluate_preregistration.py results/2026-10-12-runpod-a40/main
  python scripts/summarize_pilot_scan.py results/2026-10-12-runpod-a40/pilot-scan
  ```
- **Rerun on a GPU:** follow [DAY_1_GUIDE.md](DAY_1_GUIDE.md). On a host whose driver tops out at CUDA 12.x, use [`env/runpod_setup.sh`](env/runpod_setup.sh) for the environment. Keep results and logs on local disk, not on a network volume.
- **Check the harness logic without a GPU:**
  ```bash
  python -m unittest discover -s tests
  DRY_RUN=1 REPS=2 bash scripts/run_baseline.sh
  ```

## Layout
```text
workloads/multi_turn.py    session replay client (TTFT, cached tokens, timestamps)
scripts/run_baseline.sh    per-arm vLLM restarts, interleaved budget x arm schedule
scripts/analyze.py         per-turn reuse, paired ON/OFF comparisons, plots
scripts/*summar*.py, evaluate_preregistration.py   tables in docs/findings.md
env/                       pod environment setup (CUDA 12.8-driver workarounds)
results/2026-10-12-runpod-a40/   raw per-run data, logs and analysis (no 1 s metric series)
docs/                      findings and pre-registration
```

## Next
Does a session-aware retention policy recover the gap at 4.5 GiB, where the cache had room for about 60% of prefixes but kept under 1%? After that comes multi-replica routing (NVIDIA Dynamo).
