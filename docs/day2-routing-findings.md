# Day 2 findings: two-replica request placement (Dynamo 1.5.1)

**Date:** 2026-10-12. **Status:** measured on two GPUs, with five pre-registered repetitions ([pre-registration](preregistration-day2-main-2026-10-12.md)). It is followed by one diagnostic run and two exploratory runs, which are not part of the cohort and are labelled as such.

## Question
On the **same two GPUs** with the **same 4.5 GiB KV pool per worker**, how much does the request router change prefix reuse, recomputation, latency and load balance for Day 1's 16-session agent-like workload?

## Answer
**Placement decides almost everything. KV-aware routing with default weights did not deliver it.**
- **Session affinity (R2 hard, R3 soft):**
  - Recomputes **0.26–0.36×** the prefill tokens of round robin in every repetition.
  - Cuts resumed p95 TTFT from about 280–310 ms to **46–50 ms**.
  - Finishes about 15–20% sooner, with perfectly balanced workers.
- **The KV-aware router (R1):** beat round robin on recompute in only **3 of 5** reps; median 0.88×, range 0.77–1.24×.
- **Where the misses come from:** **every one of the 397 resumed cache misses** in the non-affinity arms happened on a turn where the session had moved to the other worker.
- **Diagnosis, from debug router logs:**
  - The router's KV index was accurate. It still moved sessions *away* from the worker it correctly believed held their prefix, because its default cost weighting priced momentary load on that worker above the reuse.
  - Raising `--router-prefill-load-scale` from 1 to 4 made the KV router reproduce hard affinity token for token. This was one exploratory run.

## Setup
- **Hardware:** RunPod on-demand pod, 2× NVIDIA RTX PRO 4500 Blackwell Server Edition (32 GB, same NUMA node), driver 580.178.04, CUDA 13.0.
  - GPU 1 carried 5,691 MiB that did not belong to this experiment. It was invisible from the container and sat at 0% utilization beforehand. It was recorded as a baseline in every manifest, and utilization was sampled during each run.
- **Software:** Dynamo `ai-dynamo==1.5.1` (git `2093662a`), vLLM 0.28.0, torch 2.13.0+cu130.
  - Discovery is file-based, requests travel over TCP and KV events over ZMQ; there is no etcd or NATS.
- **Workers:** one per GPU, identical in every arm.
  - `--block-size 64 --max-model-len 8192 --kv-cache-memory-bytes 4831838208` (32,768 tokens each), prefix caching on, KV events on.
  - `PYTHONHASHSEED=0`, `--gpu-memory-utilization 0.70`. This only affects vLLM's startup free-memory check.
  - `VLLM_USE_FLASHINFER_SAMPLER=0`. The FlashInfer sampler cannot JIT-compile for sm_120 on this image. Decoding is greedy either way.
- **Workload:** Day 1's workload exactly. The prompt digest is `c9c7e42687f3…` in every run.
  - 16 sessions × 12 turns, about 3–3.4K-token prompts, a 0.5 s tool pause, 4 requests in flight, 32 output tokens.
  - A stable `X-Dynamo-Session-ID` is sent on every turn in all arms.
- **Placement:** recorded per request from Dynamo's `nvext.worker_id`, and mapped to GPUs from each worker's log.
  - It was corroborated by the per-worker vLLM counters.
  - It was known for **every one of the 3,840 main-cohort requests**.

| Arm | Frontend |
|---|---|
| R0 | `--router-mode round-robin` |
| R1 | `--router-mode kv` (default weights) |
| R2 | round robin plus hard session affinity (TTL 300 s) |
| R3 | KV plus soft session affinity (TTL 300 s) |

## Results (main cohort)
- **Validity:** 5 reps × 4 arms, all 20 runs valid. 192/192 requests per run, placement known 192/192, the same prompt digest everywhere, and **zero preemptions**. Each run used a fresh cluster in seeded order.

Per repetition, recomputed prefill relative to R0, then resumed p95 TTFT and wall time:

| Rep | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| 1 | 1.00× · 309 ms · 28.2 s | 0.88× · 276 ms · 26.7 s | 0.26× · 48 ms · 22.0 s | 0.26× · 49 ms · 21.9 s |
| 2 | 1.00× · 280 ms · 27.1 s | 0.77× · 276 ms · 25.1 s | 0.29× · 49 ms · 21.9 s | 0.29× · 50 ms · 21.9 s |
| 3 | 1.00× · 280 ms · 25.8 s | **1.17×** · 288 ms · 26.6 s | 0.36× · 49 ms · 21.9 s | 0.36× · 49 ms · 22.1 s |
| 4 | 1.00× · 287 ms · 26.0 s | **1.24×** · 290 ms · 27.3 s | 0.34× · 49 ms · 21.8 s | 0.34× · 49 ms · 22.0 s |
| 5 | 1.00× · 289 ms · 26.1 s | 0.86× · 283 ms · 25.0 s | 0.35× · 46 ms · 21.9 s | 0.35× · 47 ms · 21.8 s |

Pre-registered hypotheses (full output in [`hypotheses.txt`](../results/2026-10-12-runpod-2x-rtxpro4500/day2-main/hypotheses.txt)):

| | Claim | Result |
|---|---|---|
| H1 | Affinity: prefill and p95 below 0.5× R0 in every rep | **Supported** |
| H2 | KV router below R0 in ≥4/5 reps but above 2× R2 | **Not supported**: below R0 in only 3/5 reps (median R1/R2 = 3.25×) |
| H3 | ≥90% of misses on worker moves | **Supported**: 397/397 |
| H4 | Soft ≈ hard affinity (±10%), balanced split | **Supported**: R3/R2 = 0.999 |
| H5 | No balance penalty for affinity (R2 wall ≤ R0) | **Supported** in every rep |

**The median resumed cached fraction is misleading here.** It is about 0.97–0.98 in every arm, R0 included, because most resumed turns still hit. Recomputed prefill and p95 expose the roughly one-third of R0 turns that missed. That is why the pre-registration made recomputation and p95 primary.

## Mechanism

### Misses are moves
In every arm, a resumed turn served by the same worker as the previous turn hit the cache, with no exceptions. A move to the other worker almost always missed (pilot: mean cached fraction 0.12–0.18 on moves). Each worker's own cache was big enough for the sessions that stayed on it, so the cost came entirely from where requests were sent.

### Why the KV router moved sessions (diagnostic run, untimed)
One R1 run used `DYN_LOG=info,dynamo_kv_router=debug`, joined per request to the router's per-worker cost formula using the response id. [`scripts/day2_router_diagnosis.py`](../scripts/day2_router_diagnosis.py) does the join.
- **On the 135 stays,** the router believed 50 cached blocks on the serving worker, and 50 were actually cached.
- **On the 41 moves,** the router believed the *previous* worker still held a median of **49 of about 50 blocks**, and the destination 0. The destination indeed had 0.
  - Moves away from a worker the router knew held the prefix: **41/41**.
  - Stale-index mispredictions: **0/41**.
  - Moves caused by eviction from both workers: **0/41**.
- **The cost formula** is `logit = prefill_load_scale × max(0, raw_prefill − overlap_credit) + decode_blocks + active`.
  - The overlap credit offsets only the prefill term, so it is worth at most the request's own prefill (about 47 blocks).
  - A worker with one extra in-flight sequence (about 48 decode blocks), or another request's queued prefill (about 50 blocks), therefore ties or beats the reuse.

Example (session 0, turn 2):

| Worker | Prefill term | Decode term | Logit |
|---|---:|---:|---:|
| Worker with the prefix | 1.25 | 143 | 144.25 |
| Empty worker (chosen) | 47.25 | 95 | **142.25** |

With 32-token outputs, that decode load drains in tens of milliseconds, while recomputing a 3K-token prefix costs about 300 ms. The default weighting makes the wrong trade for this long-prefix, short-output, pause-heavy pattern.

### Causal check (exploratory, one run each, not pre-registered)

| R1 `--router-prefill-load-scale` | Moves | Sessions kept on one worker | Recomputed prefill | Resumed p95 | Wall |
|---|---:|---:|---:|---:|---:|
| 1 (default; pilot) | 50 | 2/16 | 187,772 | 287 ms | 26.5 s |
| 2 | 6 | 12/16 | 73,148 | 49 ms | 22.2 s |
| 4 | **0** | **16/16** | **59,196** | 50 ms | 21.9 s |
| *R2 hard affinity (pilot), for reference* | 0 | 16/16 | 59,196 | 48 ms | 21.8 s |

## Predictions that were wrong
- **R0 collapse.** I expected round robin to collapse to Day 1-style near-zero reuse. It did not: about two thirds of resumed turns still hit, because round robin scrambles the strict cyclic order that defeated LRU on Day 1. Its cost showed up as about 4× the recompute and about 6× the p95, not as a near-zero hit rate.
- **R1 cause.** I attributed R1's moves to eviction or a stale index before the diagnostic run. Both were ruled out; the cause is the cost weighting.

## Limitations
- **Scope:** one model, one workload pattern, one pod and host, and one Dynamo/vLLM build.
  - The workload is synthetic and client-throttled at 4 in flight.
  - Load is light enough that affinity never had to fight real imbalance. A heavier or skewed workload could reward the load term R1 is weighting.
- **Exploratory evidence:** the prefill-load-scale results are single exploratory runs on this workload. They show the mechanism, not a recommended production setting.
- **Host asymmetry:** GPU 1's foreign 5.7 GiB allocation is an unexplained asymmetry. Utilization during runs was similar on both GPUs, and per-worker results were symmetric.
- **TTFT definition:** client TTFT is measured to the first streamed content and includes the Dynamo frontend. Compare only within Day 2.
- **No rerun of Day 1:** there was no aligned single-worker rerun on this stack, so no one-GPU-vs-two-GPU claim is made.

## Reproduce
- **Launcher:** [`scripts/day2_cluster.sh`](../scripts/day2_cluster.sh) holds all worker and frontend flags.
- **Cohort:** [`scripts/day2_run_matrix.sh`](../scripts/day2_run_matrix.sh) runs it, for example `COHORT=main REPS=5`.
- **Analysis:**
  - [`scripts/day2_analyze.py`](../scripts/day2_analyze.py) and [`scripts/day2_evaluate.py`](../scripts/day2_evaluate.py) analyse and evaluate the cohort.
  - The diagnosis needs a run with `DYN_LOG=info,dynamo_kv_router=debug`.
- **Raw data:** [`results/2026-10-12-runpod-2x-rtxpro4500/`](../results/2026-10-12-runpod-2x-rtxpro4500/), containing `day2-pilot`, `day2-main`, `day2-diag-r1-debug` and `day2-explore-r1-prefill-scale-{2,4}`.

## Next
1. **Upstream issue.** Open a reproducible issue on Dynamo, with this workload, the debug-log join and the scale sweep, about the KV router trading cached-prefix reuse for transient decode load under default weights. Ask whether a different default, or an agent/session-aware load term, is warranted. This is related to [ai-dynamo/dynamo#13279](https://github.com/ai-dynamo/dynamo/issues/13279).
2. **Sensitivity cohort.** Before generalizing, run a separate, pre-registered cohort with staggered arrivals, variable pauses and 8 in flight, comparing R1 at scale 1 and 4 against R2.
