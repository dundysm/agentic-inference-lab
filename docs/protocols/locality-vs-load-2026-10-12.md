# Preregistered protocol — locality versus load

**Prepared:** 2026-10-12. **State:** prospective, no live GPU runs have been performed for this study. This file must be committed before the first pilot/cohort is started.

## Research question and negative outcome

For a multi-turn 16-session deterministic workload, identify whether a routing policy that can move a session to a different GPU becomes preferable to **hard session affinity** once its preferred worker experiences actual contention. If this experiment never creates measurable queueing or imbalance, report no crossover demonstrated, not evidence that affinity is universally best.

## Environment and controls

The intended baseline is two equivalent GPUs on one host with Dynamo 1.5.1, matching vLLM integration, Qwen3-4B-Instruct-2507 model/tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554`; 4.5 GiB reserved KV per worker, 64-token blocks, max model length 8192, prefix cache and KV events enabled.

Record actual GPU model, driver, vLLM and Dynamo commits, CPU load, other processes' GPU memory baseline and worker placement. One host and GPU pair must be used within a cohort.

## Fixed profiles and policies

Five routing arms: `R0` round robin; `R1` KV default; `R1S4` KV with prefill load scale 4; `R2` round robin with hard affinity TTL 300 s; `R3` KV with soft affinity TTL 300 s.

Three profiles, 16 sessions × 12 turns each and identical scripted transcript:

- `reference`: 4 max in flight, max output 32, starts jittered 0–8 s, pauses 0.25–1.75 s.
- `saturation`: 16 max in flight, max output 256, same starts and pauses.
- `hot-skew`: 16 max in flight, max output 256, same starts; 8 even-local-ID hot sessions have pause duration ×0.1, cold ones ×3.

For repetition `rep`, timing seed `crossover-<profile>-rep-<rep>`. Session-hot selection is deterministic. The controller shuffles all 15 arm/profile cells by `locality-load-crossover-v1-rep-<rep>` and starts a fresh cluster per cell. Warmup session namespace is disjoint and reaches both workers.

Pilot data are exploratory and strictly excluded from the main cohort. Main: **5 repetitions, 75 runs**, all retained. Any invalid or missing cell invalidates the full affected main cohort: do not cherry-pick runs.

## Primary outcomes

Pair policies within profile and repetition. Primary:

1. Resumed-turn client-observed first-content-token latency p95.
2. Whole-workload completion time.

Secondary: p95 user-inclusive TTFT including admission waiting; session-duration p95; recomputed prefill token count; prefix reuse; average worker queue time; placement churn and worker skew; realized output-token lengths, preemptions, failures and GPU utilization.

Use one repetition, not individual turns, as comparison unit. Historical routing cohorts use different GPU types and must not be pooled.

## Primary crossover decision rule

An alternative policy must beat hard affinity `R2` on resumed p95 TTFT by **at least 10%** and have **non-worse wall time**, in at least **4 of the 5 paired repetitions**, *and* measured R2 worker average queue ≥5 ms **or** R2 request-placement imbalance ≥15 percentage points from 50/50. No conclusion is allowed without all required per-worker counters and full session/worker placement.

If none qualifies, report **no crossover demonstrated in the tested profiles**, distinguishing genuinely observed saturation from merely requested concurrency.

## Failure modes and constraints

- Prompt and schedule digests must match across arms in each profile/rep; the output cap may cause different prompt digests between profiles.
- The output cap is **not** a guarantee that 256 tokens are generated. Report realized output-length median.
- Historical metrics missingness must trigger invalid results, not be interpreted as zero.
- A seeded hot-session pause schedule does not guarantee GPU imbalance; measure actual hot-session placement.
- This is closed-loop client-throttled traffic, **not** an open-loop production arrival trace.
- No high-load routing implementation will be changed as a result of these measurements until the observed tradeoffs have been examined.

No outcome is guaranteed or implied by preparing this protocol.
