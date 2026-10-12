# Distributed KV-aware request placement

**Status: run on 2026-10-12.** Results, diagnosis and limitations are in [day2-routing-findings.md](day2-routing-findings.md). The design below is the plan as written. Deviations found during bring-up are recorded in the findings and in `scripts/day2_cluster.sh`. In particular, Dynamo 1.5.1 runs with file discovery, a TCP request plane and ZMQ events, so no etcd or NATS is needed.

The measured single-worker baseline shows a cache-residency problem under a deterministic, heavily reused multi-turn workload. This study asks whether **request placement across replicas** recovers prefix reuse before modifying engine-side retention policy.

![Proposed routing topology](assets/serving-architecture.svg)

## Research question

With the **same two GPUs, identical per-worker KV budgets and identical agent session workload**, how much do routing decisions change cache reuse, recomputation, latency and load balance?

Placement and retention are different control surfaces: a router chooses a worker; the engine chooses what KV remains resident. This experiment intentionally isolates routing first.

## Reference workload and resource invariants

Start with the same model and synthetic 16-session × 12-turn transcripts. Use two equivalent GPUs, one vLLM worker per GPU, with a 4.5 GiB KV pool per worker (**9 GiB total**). Keep the model revision, context limit, block size, tool pauses, concurrency cap and output cap fixed across *all routing arms*.

If the release-compatible Dynamo worker uses a different vLLM build or block size from the single-worker benchmark, **do not make direct speedup claims relative to that older measurement**. Rerun an aligned single-worker control if that comparison becomes important. Compare router policies on the *same two GPUs* as the primary result.

## Planned routing arms

| Arm | Policy | What it tests |
|---|---|---|
| R0 | Load-only / per-request distribution | Base distributed placement without cache locality |
| R1 | Prefix-overlap and load-aware | Whether inferred resident prefixes improve placement |
| R2 | Hard session affinity | Maximum easy-to-implement session locality; can imbalance workers |
| R3 | Soft / saturation-aware affinity | Preserves locality while letting overloaded workers shed requests |

An arm is valid only when its actual **session → worker mapping** and active configuration are observed. A stable session header alone does not prove affinity. Verify the pinned Dynamo frontend flags instead of copying untested examples from another release.

## Architecture and deployment

Use a pinned NVIDIA Dynamo release/runtime with its supported vLLM integration. The integration may require etcd/NATS and version-specific frontend, worker and KV event configuration. Start from that release's official multi-worker example and first confirm both workers register, receive traffic, and expose metrics.

Do not run the stock example as a matched benchmark: demonstration models, prefix block sizes, GPU budgets and scheduling flags may differ from this study.

Each request should carry a stable session identifier across turns. Instrument actual worker assignment using per-worker request logs/headers; a routing policy is not verified merely because it is configured.

## Quality gates before measurements

1. Two workers are available and handle requests; neither silently monopolizes the workload.
2. Both GPUs load the same model/tokenizer revision with the same KV and scheduler settings.
3. Fresh processes/caches are used between routing arms; the measured warm-up namespace is disjoint.
4. The router's KV event index is populated if the cache-aware scorer requires it.
5. Session identity persists across turns and worker assignment is recorded.
6. Per-worker `cached_tokens`, recomputation, request count, queueing, and preemptions are observable—or gaps are explicitly documented.
7. A fixed workload fingerprint matches all routing arms, and a pilot succeeds without failed requests.

## Analysis

For each repetition and routing arm, report:

- Median/p95 resumed first-content-token latency, plus harness admission waiting.
- Per-worker and aggregate recomputed prefill tokens; resumed cached-token fraction.
- Worker-placement churn per session, load skew, and queue delay.
- Completed sessions/workload time and latency-SLO goodput.
- Actual per-GPU KV memory used and any preemptions.

Keep repeated runs as the unit of comparison. First run the fixed deterministic schedule, then a separate sensitivity cohort with staggered arrivals, variable tool pauses and an increased in-flight limit. A strict improvement requires meaningful performance or SLO gains, not just more cache hits.

## Decision criteria

If a strong routing arm preserves most prefixes, identify which locality or load mechanism explains the effect. If routing does not help, test whether the observed bottleneck is cache capacity, worker assignment, queueing or cache-index staleness. **Only then** decide whether an engine-side retention or offload policy is worth developing.

Useful upstream context: [Dynamo roadmap](https://github.com/ai-dynamo/dynamo/issues/14124), [session-aware prefix indexer](https://github.com/ai-dynamo/dynamo/issues/13279), and [vLLM programmable KV RFC](https://github.com/vllm-project/vllm/issues/57103). These are context and proposals, not evidence of results achieved here.

For baseline evidence see [single-worker findings](findings.md); for existing workload-generation and metric definitions see [benchmark methodology](benchmark-methodology.md).
