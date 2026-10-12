# Agentic Inference Lab — Day 1 findings (fill after GPU measurement)

> **Status: NOT MEASURED.** This is a template, not benchmark results.

## 1. Question and precommitted hypothesis

Do resumed turns retain prefix reuse when paused session working set exceeds KV HBM, under bounded active concurrency? Hypothesis: at fixed GPU memory pressure, cache ON should save computed prefill tokens and usually improve TTFT *if* prefixes survive; under eviction, benefits diminish. Do not assume an effect.

## 2. Environment / fixed artifacts

- RunPod type, region, GPU, CPU, image tag:
- GPU driver, CUDA, vLLM, PyTorch, model and tokenizer SHA:
- vLLM startup cache capacity (blocks/tokens):
- Source commit SHA, workload SHA, seeds and timestamps:
- Actual KV budget(s), max context, GPU idle baseline:

## 3. Workload

- Actual prompt-token distribution (not document-line estimates):
- Four arms, session count, in-flight limit, pauses, turns, outputs:
- Warm-up namespace: 9000 and system tag `[warmup-not-measured]`:
- Client-throttled, closed-loop behavior and cold first-turn burst:

## 4. Method and validity

- Pilot budgets and how final budgets were chosen:
- Predeclared latency SLO:
- Repetition schedule and failures:
- Distinguish sample/harness waiting vs measured service TTFT:
- Were `cached_tokens` available in the streamed final usage chunk?
- Did GPU memory return to idle before every new server process?
- Did metrics counters reset and remain monotonic within a run?

## 5. Results

Link `summary.csv`, `paired_comparisons.csv`, `resumed_cache_fraction_by_turn.png`, `resumed_ttft.png`, `cache_reuse.png`. Describe the **distribution of per-repetition effect sizes**, not a false significance test on pooled turns.

## 6. Mechanism and counter-explanations

What changed: cached prompt fraction? Newly computed prefill tokens? Resumed TTFT? Client harness wait? Server queue or preemption? Is there enough evidence to attribute results to idle-prefix eviction?

## 7. Limitations

Synthetic deterministic history, single GPU/model, fixed pauses, client-side semaphore, metric availability, RunPod noise, no cross-replica routing, no source-code modifications, possible cache block rounding.

## 8. Decision

- **GO** to distributed Dynamo routing if the benchmark is reliable and reproduces a relevant bottleneck.
- **FIX** if cache usage fields are absent, capacity estimates wrong, errors or resource interference occur.
- **STOP/RETHINK** if no meaningful cache-pressure effect appears even under controlled pilots; report that outcome honestly.
