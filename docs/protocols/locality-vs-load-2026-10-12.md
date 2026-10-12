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

---

## Amendment 1 (2026-10-12, before any pilot or main run)

**Status when written:** no pilot or main run of this study had started, and no study data existed. The original text above is kept unchanged so the change is auditable. This amendment replaces two parts of the primary crossover decision rule.

**1. Contention gate: the placement-skew alternative is replaced.**
- **Problem:** the rule's alternative gate, "R2 request-placement imbalance ≥15 percentage points from 50/50", is structurally unreachable.
  - Under hard affinity, every session sends exactly 12 requests to its bound worker, and round-robin first placement binds sessions 8/8. A 15-point skew would require ≤5 of 16 sessions on one worker.
  - Every historical R2 run had a skew of exactly 0.0 (96/96 requests).
  - The hot-skew profile creates imbalance **over time** (hot sessions burst), which total request counts cannot show.
- **Alternative considered and rejected:** a time-weighted in-flight imbalance, measured as the mean of |share of in-flight requests on worker 0 − 0.5| over busy time.
  - Calibration on the committed historical cohorts showed affinity is burstier than round robin **even without contention**. R2 exceeded R0 by 0.06–0.10 (primary cohort) and 0.07–0.11 (sensitivity cohort), while worker queue time stayed ≤0.1 ms.
  - A gate on imbalance would therefore open without any real contention.
- **Amended gate:** a repetition counts toward a crossover only if, in that (profile, repetition), **at least one R2 worker's mean scheduler queue time is ≥5 ms** (unchanged) **or at least one R2 worker records ≥1 preemption** (`vllm:num_preemptions_total` delta).
  - Both are direct server-side contention signals.
  - Missing queue-time or preemption counters invalidate the run, as before.
  - Request-placement skew and time-weighted in-flight imbalance are still computed and reported, but descriptively only.

**2. "Non-worse wall time" becomes "at most 1.02× R2".**
- **Problem:** a strict `wall ≤ R2 wall` turns near-ties into coin flips. Wall time varied by about ±0.3 s across repetitions of the same arm in the historical cohorts, about 1–1.5% of a 20–27 s run.
- **Amended criterion:** the candidate's wall time must be at most 1.02× R2's in the same (profile, repetition).

**Unchanged:**
- The p95 criterion (≤0.90× R2).
- The 4-of-5 repetition requirement.
- All profiles, arms, seeds, validity rules and the "no crossover demonstrated" outcome.

**Implementation:** `scripts/analyze_load_crossover.py` (`contention_observed`, `crossover_win`, `WALL_TOLERANCE`, `inflight_imbalance`), with unit tests in `tests/test_logic.py`.
