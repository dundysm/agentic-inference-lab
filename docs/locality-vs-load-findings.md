# Locality versus load: findings

**Date:** 2026-10-12.

**Outcome:** **no saturation regime reached at the tested settings.** The pre-registered go/no-go rule (Amendment 2) was not met, so the 75-run main cohort was **not run**. No locality/load crossover is demonstrated or refuted.

- Protocol: [locality-vs-load-2026-10-12.md](protocols/locality-vs-load-2026-10-12.md), with Amendments 1 and 2.
- Study design: [locality-vs-load.md](locality-vs-load.md).
- Raw data: [`results/2026-10-12-runpod-2x-rtxpro4500se-locality-load/`](../results/2026-10-12-runpod-2x-rtxpro4500se-locality-load/).

## Question
For the 16-session × 12-turn agent-like workload, does a policy that can move a session to a different GPU become preferable to **hard session affinity** once affinity's preferred worker experiences real contention?

## What ran

| Phase | Settings | Runs | Valid |
|---|---|---:|---:|
| Original pilot | Stress profiles: 16 in flight, **unforced** 256-token cap | 15 | 15/15 |
| Pilot A2 (after Amendment 2) | Stress profiles: 16 in flight, **forced** 512-token outputs (`ignore_eos`) | 15 | 15/15 |
| Main cohort (75 runs) | — | **not run** (go/no-go not met) | — |

**Hardware and software:** one RunPod pod, 2× NVIDIA RTX PRO 4500 Blackwell Server Edition (32 GB each, separate NUMA nodes), driver 580.178.04, CUDA 13.0. No foreign GPU memory at baseline. Dynamo 1.5.1, vLLM 0.28.0, `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f…`.

**Fixed across all cells:**
- 4.5 GiB KV per worker (32,768 tokens), 64-token blocks.
- Seeded 0–8 s staggered starts and seeded tool pauses.
- One fresh cluster per run, with arm and profile order shuffled.

**Validity:** every run in both pilots passed the analyzer's fail-closed checks:
- 192/192 requests, with worker placement known for every one;
- full cached-token coverage;
- prompt and schedule digests identical across arms;
- queue-time and preemption counters present.

In pilot A2, every stress-profile response generated exactly **512** tokens.

## Results

### Go/no-go measure
R2 (hard affinity) contention in the stress profiles, by the Amendment 1 definition:

| Phase | Profile | R2 mean worker queue (max of 2) | R2 preemptions | Realized output (median) |
|---|---|---:|---:|---:|
| Original pilot | saturation | 0.01 ms | 0 | 62 tokens |
| Original pilot | hot-skew | 0.01 ms | 0 | 62 tokens |
| **Pilot A2** | **saturation** | **0.01 ms** | **0** | **512** |
| **Pilot A2** | **hot-skew** | **0.01 ms** | **0** | **512** |

The gate requires ≥5 ms or ≥1 preemption. It was not met in either pilot, even with outputs forced to 512 tokens, so the main cohort did not run.

### Pilot A2, all arms
One repetition per cell (exploratory). Full table: [`pilot-a2/summary.md`](../results/2026-10-12-runpod-2x-rtxpro4500se-locality-load/locality-load/pilot-a2/summary.md).

| Profile | Arm | Recomputed prefill vs R2 | Resumed p95 TTFT | Wall | Worker queue | Preemptions |
|---|---|---:|---:|---:|---:|---:|
| saturation | R0 round robin | 6.16× | 555 ms | 134.7 s | **5.60 ms** | 0 |
| saturation | R1 KV default | 2.42× | 303 ms | 128.0 s | 0.01 ms | 0 |
| saturation | R1S4 KV, prefill ×4 | 1.08× | 64 ms | 125.6 s | 0.01 ms | 0 |
| saturation | R2 hard affinity | 1.00× | 64 ms | 125.4 s | 0.01 ms | 0 |
| saturation | R3 soft affinity | 1.00× | 64 ms | 125.5 s | 0.01 ms | 0 |
| hot-skew | R0 | 5.81× | 319 ms | 149.0 s | 0.01 ms | 0 |
| hot-skew | R1 | 2.11× | 298 ms | 143.6 s | 0.17 ms | 0 |
| hot-skew | R1S4 | 1.00× | 63 ms | 141.6 s | 0.01 ms | 0 |
| hot-skew | R2 | 1.00× | 64 ms | 142.0 s | 0.01 ms | 0 |
| hot-skew | R3 | 1.00× | 65 ms | 142.2 s | 0.01 ms | 0 |
| reference | R0 / R1 / R1S4 / R2 / R3 | 4.99× / 3.61× / 1.00× / 1.00× / 1.00× | 290 / 280 / 51 / 50 / 49 ms | 27–35 s | ≤0.16 ms | 0 |

## Interpretation

These are single-repetition pilots, so they are descriptive. No crossover claim either way.

- **Hard affinity never created contention at these settings.** With 16 sessions bound 8 per worker, forced 512-token decodes and a 32,768-token KV pool per worker, both workers' scheduler queues stayed empty and nothing was preempted. At 16 sessions on two 4.5 GiB workers, session affinity had headroom. The scenario the study was built to test, *the affinity-preferred worker is overloaded*, did not occur.
- **The only measurable queueing came from round robin.** R0 averaged 5.6 ms in the saturation profile. Spreading each session's turns across both workers recomputes far more prefill (6.2× affinity), and that recomputation is itself load. Load-balancing by request lost locality and created the contention it was meant to avoid.
- **Consistent with the earlier routing cohorts:**
  - The default-weighted KV router recomputed 2.1–3.6× affinity.
  - With `--router-prefill-load-scale 4`, it was within 1.00–1.08× of affinity in every profile, with matching tail latency.
  - Soft affinity was indistinguishable from hard.
- **The realized-output lesson:** a `max_tokens` cap is not a load knob. The unforced 256-token cap realized only about 61 tokens. Forcing length (`ignore_eos`) was necessary but still not sufficient to saturate.

## What would be needed to test the crossover
A future, separately pre-registered study would need settings where affinity's preferred worker actually saturates. Candidates:
- more sessions than one worker's KV can hold while active (e.g., 24–32 sessions, or a smaller KV budget per worker);
- longer forced decodes;
- a hot-session skew strong enough that one worker's bound sessions exceed its capacity.

The go/no-go gate and the Amendment 1 decision rule could be reused unchanged.

## Cost and resources
- One 2× RTX PRO 4500 SE pod (`ewom212ycofeho`, $1.45/hr), about 1 h 17 min, roughly **$1.85**. Terminated after the go/no-go decision.
- Commits: Amendment 1 (`0bd3e24`) and Amendment 2 (`ff72551`) both landed before the runs they govern.
