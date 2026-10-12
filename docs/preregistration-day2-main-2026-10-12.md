# Day 2 main cohort pre-registration: two-replica routing

**Written:** 2026-10-12, after the four-arm pilot (`results/2026-10-12-runpod-2x-rtxpro4500/day2-pilot/`) and **before** any main-cohort run.

## Frozen configuration
- **Hardware:** RunPod on-demand pod `t08nb1xd591jrv` (US-KS-2, Secure Cloud), 2× NVIDIA RTX PRO 4500 Blackwell Server Edition (32 GB each, same NUMA node), driver 580.178.04, CUDA 13.0. The same pod is used for every run.
- **Known host asymmetry:** GPU 1 carries 5,691 MiB that is not ours and is not visible from inside the container. It sat at 0% utilization before the runs. It is recorded as a per-GPU baseline in every manifest. Per-GPU utilization is sampled during every run.
- **Software:** NVIDIA Dynamo `ai-dynamo==1.5.1` (git `2093662a4d4e5fd318732935213dde08dc7be818`), vLLM `0.28.0`, torch `2.13.0+cu130`.
- **Discovery and transport:** file discovery, TCP request plane, ZMQ event plane.
- **Model:** `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- **Workers:** identical in every arm, one per GPU.
  - Flags: `--block-size 64 --max-model-len 8192 --kv-cache-memory-bytes 4831838208` (32,768 KV tokens each), `--gpu-memory-utilization 0.70`, prefix caching on, KV events on.
  - Environment: `PYTHONHASHSEED=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`.
  - The exact argv is recorded per run in `launch_argv.txt`.
- **Arms:** frontend flags per `scripts/day2_cluster.sh` at the commit that adds this file.

| Arm | Frontend routing |
|---|---|
| R0 | `--router-mode round-robin` |
| R1 | `--router-mode kv` |
| R2 | `--router-mode round-robin`, hard session affinity, TTL 300 s |
| R3 | `--router-mode kv`, soft session affinity, TTL 300 s |

- **Workload:** Day 1's 16 sessions × 12 turns, 80 lines, 0.5 s pause, 4 in flight, `max_tokens=32`, `temperature=0`.
  - A stable `X-Dynamo-Session-ID` is sent on every turn in all arms.
  - Placement is recorded via `nvext.extra_fields=["worker_id"]`.
  - The prompt fingerprint must equal `c9c7e42687f3…` in every run.
- **Cohort:** `REPS=5`, arm order seeded per rep (`agentic-lab-day2-{rep}`), a fresh cluster per run, results in `results/day2/main`.
- **TTFT SLO for goodput:** 3,000 ms (illustrative).

## Primary metrics
These are per run, compared as paired per-repetition ratios against R0:
1. Total recomputed prefill tokens: the sum of both workers' `vllm:request_prefill_kv_computed_tokens_sum` deltas.
2. Resumed-turn p95 TTFT (176 resumed requests per run).
3. Workload wall time.

**The median resumed cached fraction is reported but is not primary.** The pilot showed it hides misses: the R0 median was 0.975 while about a third of resumed turns missed.

## Hypotheses
- **H1 (affinity helps):** in every rep, R2 and R3 each recompute less than 0.5× R0's prefill tokens and have resumed p95 TTFT below 0.5× R0's.
- **H2 (KV-aware routing is partial):** R1's recomputed prefill is below R0's in at least 4 of 5 reps, but its median across reps is above 2× R2's.
- **H3 (misses come from moves):** in R0 and R1, pooled across reps, at least 90% of resumed misses (cached fraction below 0.5) occur on turns where the session changed worker since its previous turn.
- **H4 (soft ≈ hard here):** the median across reps of R3/R2 recomputed prefill is within ±10% of 1.0, and R2 splits requests within 45–55% per worker in every rep.
- **H5 (no balance penalty):** R2's wall time is at or below R0's in every rep.

## Rules
- Every run is kept. A run with failed requests, unknown placement, a fingerprint mismatch or an aborted cluster is reported as invalid, not dropped silently.
- Any aborted run means the full cohort is rerun.
- The unit of analysis is one repetition. Turns are not independent.
- A separate, **untimed** diagnostic R1 run with debug router logging may follow the main cohort to explain R1's moves. It is not part of the cohort.
