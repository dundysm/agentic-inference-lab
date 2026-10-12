# Day 1 findings: prefix-cache retention across tool pauses (single GPU)

**Date:** 2026-10-12. **Status:** measured on a real GPU, five independent repetitions, with hypotheses pre-registered in [preregistration-main-2026-10-12.md](preregistration-main-2026-10-12.md) before the main runs.

## Question
Does vLLM's automatic prefix caching keep long session prefixes alive across tool-like pauses when the paused sessions' working set is larger than the KV budget? And what does that do to recomputation, TTFT and throughput?

## Answer
**No. When the working set does not fit, reuse falls off a cliff.** At 4.5 GiB, about 0.6× the working set of the 16-session workload, the prefix cache gave **zero measurable benefit**. Cache ON and OFF were the same on every metric, and there were no preemptions.

With 10 GiB, about 1.35× the working set, the same workload got about **8× faster resumed-turn TTFT** and finished in **42% of the wall time**. The 4-session workload got a 4–6× benefit at both budgets.

## Environment
- **Pod:** RunPod on-demand `6sp3oabp0owevz`, 1× NVIDIA A40 (46 GB), driver 570.211.01, 9 vCPU. The same pod was used for every run.
- **Software:** vLLM `0.31.0+cu129`, torch `2.13.0+cu129`, Python 3.12. `torchcodec 0.17.0+cpu` is unused by this text-only workload.
- **Model:** `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554` (model and tokenizer), BF16, `--max-model-len 8192`.
- **KV capacity:** taken from the server logs, and it matches the 144 KiB/token estimate.

| Budget | GPU KV cache tokens |
|---|---:|
| 2.5 GiB | 18,192 |
| 3 GiB | 21,840 |
| 4.5 GiB | 32,768 |
| 6 GiB | 43,680 |
| 8 GiB | 58,240 |
| 10 GiB | 72,816 |

## Workload
- Each session uses a unique, deterministic incident log of 80 lines, giving prompts of **2,964–3,405 tokens**.
- Sessions run **12 turns** with append-only scripted history, a **0.5 s tool pause** outside the admission semaphore, `max_tokens=32` and `temperature=0`.
- A and B run 4 sessions with 2 in flight. C and D run 16 sessions with 4 in flight.
- Arm C's working set is about 16 × 3.4K ≈ **54K tokens**. The active footprint, 4 × 3.4K ≈ 14K, fits in every budget tested, so the C/D comparison isolates **idle-prefix eviction, not active preemption**.
- Each run uses a fresh server process, a disjoint warm-up namespace, and before/after Prometheus snapshots.

## Method
- **Arms:** A–D (cache OFF/ON × 4/16 sessions) at two KV budgets, with 5 repetitions. That is 40 runs and 7,680 requests.
- **Order:** within each repetition, all 8 (budget, arm) pairs were shuffled with a seed.
- **Choosing the budgets:** they were picked from the pilot scan (C only, at 2.5–8 GiB) before the main cohort.
- **Unit of analysis:** one repetition's median. Turns are not treated as independent.
- **Aborted cohort:** a first main cohort aborted after 4 runs because log writes on RunPod's network filesystem (`/workspace`) were lost. Following the pre-registration rule, it was discarded in full and rerun from local disk. Its partial data was not retained; it was lost when the pod was terminated.

## Results (main cohort)
All 40 runs completed. **0 failed requests and 0 preemptions** across all 7,680 requests.

Values are the median across reps of the per-run medians, with [min, max] across reps.

| Budget | Arm | Cache | Sessions | Resumed cached fraction | Resumed TTFT (ms) | TTFT incl. harness wait (ms) | Prefill tokens computed | Wall (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4.5 GiB | A | OFF | 4 | 0.000 | 331 [329, 347] | 965 | 153,216 | 25.8 |
| 4.5 GiB | B | ON | 4 | 0.989 | 56 [56, 75] | 168 | 13,664 | 15.2 |
| 4.5 GiB | C | ON | 16 | **0.005** | **581 [459, 599]** | 5,132 | **614,668** | **80.1** |
| 4.5 GiB | D | OFF | 16 | 0.000 | 591 [461, 611] | 5,119 | 617,724 | 80.3 |
| 10 GiB | A | OFF | 4 | 0.000 | 333 [331, 350] | 967 | 153,216 | 25.8 |
| 10 GiB | B | ON | 4 | 0.989 | 56 [56, 75] | 169 | 13,648 | 15.0 |
| 10 GiB | C | ON | 16 | **0.989** | **67 [64, 75]** | 1,608 | **54,956** | **33.7** |
| 10 GiB | D | OFF | 16 | 0.000 | 567 [458, 599] | 5,139 | 617,724 | 80.4 |

Paired ON/OFF ratios per repetition, as resumed-TTFT median ratio and wall-time ratio:

| Comparison | TTFT ratio | Wall ratio |
|---|---|---|
| 4.5 GiB, B/A | 0.22, 0.17, 0.17, 0.17, 0.17 | 0.59, 0.59, 0.59, 0.59, 0.58 |
| 4.5 GiB, C/D | 0.96, 0.82, 1.00, 0.98, 1.02 | 1.00 in all five |
| 10 GiB, B/A | 0.23, 0.16, 0.17, 0.17, 0.17 | 0.61, 0.57, 0.59, 0.58, 0.58 |
| 10 GiB, C/D | 0.13, 0.13, 0.11, 0.14, 0.15 | 0.43, 0.43, 0.42, 0.42, 0.42 |

**Hypotheses:** H1, H2 and H3 are **SUPPORTED** in every repetition. See [`hypotheses.txt`](../results/2026-10-12-runpod-a40/main/analysis/hypotheses.txt).

**Pilot scan (C only, one rep each):** the resumed cached fraction was 0.005 at 2.5, 3, 4.5 and 6 GiB, and 0.989 at 8 GiB. The cliff lies between 43.7K and 58.2K tokens of capacity, bracketing the ~54K working set.

## Mechanism
- **The prefixes are evicted while sessions are paused, not while they run.** Preemptions were zero everywhere, and the active set always fits. During its 0.5 s pause, a session's blocks have reference count 0 and are evicted to make room for other sessions' prefills.
- **Why it is a cliff and not a slope.** Sixteen sessions take turns in roughly round-robin order. With least-recently-used eviction and a working set larger than capacity, the block evicted next is always the one needed soonest. That is the textbook LRU cyclic-access pathology. Capacity for about 60% of the working set therefore yields about 0.5% reuse, not about 60%.
- **What the 0.5% is.** It is the shared system-prompt block. The deterministic recompute count (614,668 versus 617,724 with cache OFF, a difference of 3,056 tokens) shows that is all that survived.
- **Visibility.** `kv_cache_usage_perc` and `num_preemptions` would not show this failure. The per-request `cached_tokens` signal is what reveals it.

## Uncertainty and limitations
- **Scope:** one GPU type, one model, one vLLM build, and one pod and host.
- **Synthetic workload:** scripted, closed-loop sessions, not real agents. Arrival is client-throttled (at most 4 in flight), so server-side queueing is small and the "user" TTFT is dominated by harness wait.
- **Client-side TTFT:** this is time to the first SSE content chunk, which is not the server's internal TTFT.
- **Pressured-arm noise:** resumed TTFT in the pressured arms is bimodal across reps (about 460 ms vs about 590 ms). The cause was not investigated. One rep's C/D ratio at 4.5 GiB was 0.82, while the median across reps was −2.2%.
- **Untested factors:** the exact cliff location for other pause lengths, session counts, in-flight limits and block sizes was not tested. In the pilot, resumed TTFT was higher at 2.5–3 GiB than at 4.5–6 GiB despite identical recompute. This is plausibly admission limits near capacity, but it was not verified.

## Next question
At 4.5 GiB, the cache had room for about 60% of the paused sessions' prefixes but kept under 1%. How much of that gap does a session-aware retention policy recover, compared with LRU, on the same workload? Candidates are pinning prefixes during a known tool pause, or segmented or scan-resistant eviction. This is now a measured, reproducible target for a small vLLM experiment and an upstream discussion. Multi-replica routing (Dynamo) comes after.

## Reproduce
- **Harness:** `workloads/`, `scripts/run_baseline.sh` and `scripts/analyze.py`, unmodified from the version that was run (the first commit in this repo).
- **Environment:** [`env/runpod_setup.sh`](../env/runpod_setup.sh) and [`env/env.sh`](../env/env.sh). These record the cu129, torchcodec and local-disk workarounds the run needed.
- **Main cohort command:** [`results/2026-10-12-runpod-a40/main/analysis/main.sh`](../results/2026-10-12-runpod-a40/main/analysis/main.sh).
- **Re-derive the tables from the committed raw data** (no GPU needed):
  - `python scripts/evaluate_preregistration.py results/2026-10-12-runpod-a40/main`
  - `python scripts/summarize_pilot_scan.py results/2026-10-12-runpod-a40/pilot-scan`
  - `python scripts/summarize_ab_pilot.py results/2026-10-12-runpod-a40/ab-pilot`
- **Raw data per run:** `requests.jsonl`, the before/after `.prom` snapshots, `server.log`, `kv_allocation.log`, `environment.txt` and `server_config.json`. The 1-second Prometheus time series are not in git.
