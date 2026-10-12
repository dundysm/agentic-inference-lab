# Locality versus load: crossover study

**Status:** implementation and preregistered protocol prepared. **No GPU crossover results have been run or published.**

## Research question

Does preserving a multi-turn session's resident KV prefix remain optimal when the cache-holding worker is genuinely busy, or is there an offered-load/imbalance regime in which load-sensitive routing improves end-user tail latency or completed-session time?

This study extends the completed [distributed routing experiments](routing-findings.md), without changing their raw data. It intentionally tests for a regime where **hard session affinity loses**, including reporting a null result if one is not reached.

## Fixed system controls

- Two identical NVIDIA GPUs on one host, Dynamo 1.5.1 with the matching vLLM worker build, identical reserved **4.5 GiB GPU KV per worker**, 64-token KV blocks, 8,192-token model limit, identical prefix-caching and sampling configurations.
- `Qwen/Qwen3-4B-Instruct-2507` at pinned model/tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- 16 scripted sessions × 12 turns, identical session transcripts across policies for a given profile. No model-generated text feeds back into the next prompt.
- Fresh workers and cache for **each** arm/repetition, separate prompt warm-up namespace, per-worker metrics and response worker IDs.
- Five policies, using existing Dynamo flags: R0 round robin, R1 default KV-aware, R1S4 KV-aware with `--router-prefill-load-scale 4`, R2 hard session affinity, R3 KV-aware with soft session affinity.
- **Never compare absolute latency across GPUs, hosts, or different profile workloads**. Compare policies only within a paired profile/repetition on the same host.

## Workload profiles

| Profile | In flight | Output cap | Start staggering | Tool pauses | Session skew |
|---|---:|---:|---|---|---|
| `reference` | 4 | 32 | Uniformly seeded 0–8 seconds | Seeded 0.25–1.75 s | No |
| `saturation` | 16 | 512, forced (`ignore_eos`) | Same seed/window | Same seeded range | No |
| `hot-skew` | 16 | 512, forced (`ignore_eos`) | Same seed/window | Hot pauses ×0.1; cold ×3 | 8 deterministic even-ID hot sessions |

**Amendment 2:** the original pilot used an unforced 256-token cap, and the model stopped after a median of about 61 tokens, so the stress profiles generated no contention. The stress profiles now force exactly 512 output tokens, and a stress run is invalid unless every response reaches the cap. See [Amendment 2](protocols/locality-vs-load-2026-10-12.md#amendment-2-2026-10-12-after-the-original-pilot-before-any-main-run). The analyzer records realized output-token lengths. Likewise, setting concurrency to 16 does not prove saturation; inspect measured worker queueing, load imbalance, and harness admission delay.

Sessions and turns are constant, but the prompt fingerprints must differ between profiles with different output caps. Session transcripts themselves remain identical. For each profile/repetition, the exact full prompt digest and **entire timing schedule digest** must match across all routing policies.

The hot-session selection is intentionally even-indexed first to create a repeatable asymmetric pressure candidate, **not** a guarantee that the resulting worker placement will be skewed. Verify the observed hot-session-to-worker split.

## Experiment protocol

The historical routing scripts provide the backend and metrics path; the crossover controller simply chooses workload parameters and randomizes the full set of profile/policy cells **within each repetition**. All runs are saved to new directories, never into completed baselines.

```bash
# On a CPU machine: preview the full randomized schedule; no GPU is started.
DRY_RUN=1 STUDY_PHASE=pilot bash scripts/run_load_crossover.sh
DRY_RUN=1 STUDY_PHASE=main bash scripts/run_load_crossover.sh
```

On a compatible on-demand two-GPU pod:

```bash
# Clone the branch containing this study; use the pinned Dynamo runtime/environment.
git clone https://github.com/dundysm/inference-under-pressure.git
cd inference-under-pressure
# If not yet merged: git switch feat/locality-load-study-and-docs

# See env/runpod_routing_setup.sh for host-specific installation commands.
source /root/venv/bin/activate
export HF_HOME=/root/hf-cache HF_HUB_OFFLINE=1
export MODEL=Qwen/Qwen3-4B-Instruct-2507
export MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554

# Calibrate that the router loads and each scenario runs successfully (Amendment 2 pilot).
STUDY_PHASE=pilot-a2 RESULTS_ROOT=/root/results/inference-under-pressure \
  bash scripts/run_load_crossover.sh

# Only if the pilot-a2 go/no-go rule passes (Amendment 2): R2 contention in at least one stress profile.
STUDY_PHASE=main RESULTS_ROOT=/root/results/inference-under-pressure \
  bash scripts/run_load_crossover.sh

# Local post-processing; fails if expected data are missing.
python scripts/analyze_load_crossover.py \
  /root/results/inference-under-pressure/locality-load/main \
  --reps 5 --out /root/results/inference-under-pressure/locality-load/main/report.md
```

The main cohort requires **5 repetitions × 3 profiles × 5 routing arms = 75 clean restarts**. It can consume significant rental-GPU time. Pilot first. For a genuinely different experiment, write and commit a new protocol rather than silently changing this cohort.

## Validity checks

A comparison is **invalid** if any required per-worker prefill or queue-time counter is missing; any response failed; placement is unknown; the prompt/schedule digests differ across arms; the measured cache budget/worker count differs; GPU memory hasn't returned to baseline between runs; or a worker is not serving.

Inspect per-worker queue-time deltas (`vllm:request_queue_time_seconds_sum/count`), cached-token coverage, TTFT, **user-inclusive TTFT including harness admission**, per-session completion times, worker placement skew, realized generated-token count, preemptions and GPU utilization. A near-zero queue under the supposedly saturated profile means the crossover was **not demonstrated**, regardless of how large the configured output cap is.

## Comparison and interpretation

For each five-repetition profile, compute paired per-run differences/ratios relative to hard affinity (R2). Primary performance outcomes: resumed p95 TTFT and workload wall time, with client-inclusive p95 and p95 session duration as supporting measures. Secondary: recomputed prefill, measured queues, placement churn/skew, and hot-session placement. Keep individual run metrics intact.

A **strict crossover** requires an alternative policy (R0, R1, R1S4 or R3) in at least four of five repetitions to have:

1. Resumed p95 TTFT **at most 0.90× R2** (at least 10% better).
2. Workload wall time **at most 1.02× R2** (no more than 2% worse).
3. Real server-side contention under R2: at least one R2 worker's mean scheduler queue time is ≥5 ms, **or** an R2 worker preempted at least one request.

Rules 2 and 3 are as amended before any measurement; see [Amendment 1](protocols/locality-vs-load-2026-10-12.md#amendment-1-2026-10-12-before-any-pilot-or-main-run). The original rule 3 used request-placement skew ≥15 points, which cannot occur under hard affinity with 16 equal-length sessions. Placement skew and time-weighted in-flight imbalance are still reported as descriptive measures.

That is an operational decision rule, **not** a population-confidence claim. Predeclare any other success test in a separate, prospective protocol; do not move thresholds after measuring.

If no profile produces meaningful queueing or imbalance, report **"No saturation regime reached"**, not "affinity always wins." If the load-sensitive policy performs better, inspect actual router scoring logs to determine whether load relief is the mechanism.

## Supporting material

- [Frozen protocol and explicit thresholds](protocols/locality-vs-load-2026-10-12.md)
- [Recorded routing findings](routing-findings.md)
- [Engineering notes](engineering-notes.md)
- [NVIDIA Dynamo upstream issue #16194](https://github.com/ai-dynamo/dynamo/issues/16194)

**Limitations:** the traffic is synthetic, closed-loop and controlled by a semaphore. A true open-loop arrival-process follow-up and realistic variable output-length trace would be a separate study; do not claim this protocol covers production burst dynamics.
