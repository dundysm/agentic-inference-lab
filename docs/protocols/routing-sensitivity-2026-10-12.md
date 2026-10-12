# Preregistered protocol: routing sensitivity and capacity controls

**Written:** 2026-10-12, after the Day 2 main cohort, the diagnostic run and the two exploratory runs ([findings](../routing-findings.md)). It was written **before** any follow-up measurement and before the pod for these cohorts was deployed.

These cohorts finish the remaining GPU items in the Day 2 plan:
- the guide's section 8.3 sensitivity cohort;
- the guide's original R3 variant (KV routing with hard affinity);
- the guide's section 0 single-worker control on the Day 2 software stack.

They are separate cohorts. **Results are compared only within each cohort**, never spliced with the main cohort.

## Common configuration
Everything is as frozen in [the main-cohort pre-registration](routing-primary-2026-10-12.md) unless stated here:
- **Pod and GPUs:** a new RunPod on-demand 2× RTX PRO 4500 pod, the same type as the main cohort. If only a different GPU is available, the hypotheses still apply but the type is recorded and flagged.
- **Software:** Dynamo 1.5.1, vLLM 0.28.0, the same model revision, 64-token blocks, 8192 context and 4.5 GiB KV per worker unless stated otherwise.
- **Arms:** defined in `scripts/routing_cluster.sh` at the commit adding this file.
  - R0–R3 launch byte-identically to the main cohort; this was checked by diffing the generated argv.
  - New arms are `R1S4` (KV router with `--router-prefill-load-scale 4`) and `R3H` (KV router with hard affinity, TTL 300 s).
  - Single-worker arms are `S45` (one worker on GPU 0 with 4.5 GiB KV) and `S9` (one worker on GPU 0 with 9 GiB KV, equal to the two-worker total).
- **Workload:** the same prompts as before; the prompt digest must be `c9c7e42687f3…`. The order of arms within a rep is seeded (`agentic-lab-day2-{rep}`), and each run gets a fresh cluster.
- **Recorded per run:** placement (`nvext.worker_id`), per-worker metrics and GPU utilization.
- **Primary metrics:** as before — recomputed prefill (sum of worker counter deltas), resumed p95 TTFT and wall time. The unit of analysis is one rep.

## Cohort S: sensitivity (`COHORT=sens`)
- **Arms:** R0, R1, R1S4, R2. R3 (soft) is omitted because it matched R2 to within 0.1% in the main cohort.
- **Reps:** 5, giving 20 runs.
- **Timing:** `STAGGER_S=8`, `PAUSE=0.25`, `PAUSE_MAX=1.75`, `MAX_INFLIGHT=8`, `SCHEDULE_SEED=day2-sensitivity`.
  - Each session starts after a seeded uniform delay of 0–8 s.
  - Each tool pause is a seeded uniform 0.25–1.75 s (mean 1.0 s).
  - Up to 8 requests are in flight instead of 4.
  - The schedule is identical across arms and reps (its `schedule_sha256` is recorded), so comparisons are paired.

Hypotheses:
- **S-H1 (affinity survives):** R2's recomputed prefill is below 0.5× R0's in every rep.
- **S-H2 (default KV router still ≈ round robin):** the median across reps of R1/R0 recomputed prefill is at least 0.75.
- **S-H3 (prefill weighting recovers locality):** the median across reps of R1S4/R2 recomputed prefill is at most 1.2.
- **S-H4 (no tail penalty for the reweighting):** the median across reps of R1S4/R2 resumed p95 TTFT is at most 1.2.
- **S-H5 (misses come from moves):** in R0 and R1, pooled across reps, at least 90% of resumed misses (cached fraction below 0.5) occur on turns where the session changed worker.
- **S-H6 (affinity stays balanced):** R2 serves 40–60% of requests on each worker in every rep.

## Cohort X: supplementary, fixed workload (`COHORT=supp`)
- **Arms:** R2, R3H, S45, S9.
- **Reps:** 3, giving 12 runs.
- **Workload:** the main cohort's workload and timing (no stagger, 0.5 s pause, 4 in flight).

Hypotheses:
- **X-H1 (guide R3 ≈ R2):** the median across reps of R3H/R2 recomputed prefill is within ±10% of 1.0.
- **X-H2 (the Day 1 collapse reproduces on the Day 2 stack):** S45's resumed mean cached fraction is below 0.2 in every rep. One worker with 4.5 GiB holding 16 sessions is Day 1's thrashing configuration, now with Dynamo, vLLM 0.28 and 64-token blocks.
- **X-H3 (pooling the memory on one GPU also fits the working set):** S9's resumed mean cached fraction is above 0.9 in every rep, and its recomputed prefill is at most 1.2× R2's (median across reps).
- **X-H4 (two GPUs finish sooner):** R2's wall time is below S9's in every rep. This is a capacity comparison, not a routing claim.

## Rules
- These are the same as for the main cohort:
  - Every run is kept.
  - Failed, partial, unknown-placement or digest-mismatched runs are reported as invalid.
  - An aborted run means the whole affected cohort is rerun.
- Where a hypothesis fails, the result is reported as it is; its threshold is not adjusted after seeing the data.
