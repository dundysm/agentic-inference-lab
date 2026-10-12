# Inference Under Pressure

**Measuring how KV-cache residency, request placement, and serving load affect multi-turn LLM inference.**

A real-GPU investigation using **vLLM** and **NVIDIA Dynamo**. Includes deterministic workloads, pinned environments, experimental controls, raw traces, router-decision diagnosis, and independently repeatable analysis. The goal is to understand when cache locality helps—and when a router should trade locality for load balance.

[Results](#measured-results) · [Routing findings](docs/routing-findings.md) · [Cache residency](docs/cache-residency-findings.md) · [Methodology](docs/routing-methodology.md) · [Engineering notes](docs/engineering-notes.md) · [Load crossover study](docs/locality-vs-load.md)

![Measured single-worker and distributed routing experiment topology](docs/assets/serving-architecture.svg)

## The systems question

Multi-turn LLM applications alternate inference with tool calls. During idle periods, session-prefix KV blocks compete for finite accelerator memory. With more than one serving replica, an additional decision matters: **which worker should receive the resumed turn?**

This repository separates three factors that are often conflated: (1) GPU cache capacity, (2) session-to-worker placement, and (3) momentary serving load.

## Measured results

### Cache residency under pressure

**One NVIDIA A40 · vLLM 0.31.0+cu129 · Qwen3-4B-Instruct-2507 · 16 synthetic sessions × 12 turns**

| Metric, prefix caching enabled | 4.5 GiB KV | 10 GiB KV |
|:--|--:|--:|
| Resumed prompt tokens cached (median) | **0.5%** | **98.9%** |
| Resumed first-content-token latency (median) | **581 ms** | **67 ms** |
| Computed prefill tokens per run | **614,668** | **54,956** |
| Workload completion time | **80.1 s** | **33.7 s** |

This sharply varying reuse pattern is specific to the tested cyclic workload, not a universal threshold. Five repetitions per controlled arm/budget produced **40 valid runs and 4,800 successful requests**. [Full controls and uncertainty →](docs/cache-residency-findings.md)

![Measured KV-budget sensitivity](docs/assets/cache-reuse-cliff.svg)

### Distributed KV-aware routing

**Two RTX PRO 4500 GPUs · Dynamo 1.5.1 / vLLM 0.28.0 · identical 4.5 GiB KV pools per worker**

| Routing mode | Recomputed prefill vs round robin | Resumed p95 first-content-token latency |
|:--|--:|--:|
| Round robin | 1.00× | ~289 ms |
| KV-aware, default weights | 0.88× median paired ratio, inconsistent | ~283 ms |
| Hard session affinity | **0.34×** | **~49 ms** |
| KV-aware + soft affinity | **0.34×** | **~49 ms** |

The main routing cohort contains **20 valid runs / 3,840 successful requests**. A debug-trace join showed the KV router selecting a worker *without* the reusable prefix despite recognizing that another worker held it; its load terms outweighed the locality credit in this workload.

A separate **two-L40S sensitivity cohort** with staggered arrivals, variable pauses, and eight in-flight requests changed the answer: default KV routing improved to **0.51×** round robin's recomputed prefill, while a 4× prefill weight reached **0.26×** and hard affinity reached **0.21×**. These are within-cohort comparisons; different GPU types are **never** treated as directly comparable latency baselines.

A supplementary control showed that one 4.5 GiB worker loses reusable prefixes, while one 9 GiB worker preserves them about as well as two affinity-routed 4.5 GiB workers. [Full routing evidence, methods and limitations →](docs/routing-findings.md)

![Measured routing-policy prefill breakdown](results/2026-10-12-runpod-2x-rtxpro4500/day2-main/analysis/day2_prefill_per_worker.png)

## Current investigation: locality versus load

The remaining question is **where session affinity becomes harmful because the preferred worker is genuinely overloaded**. The [load-crossover protocol](docs/locality-vs-load.md) specifies reference, high-concurrency/long-decode, and hot-session-skew profiles, comparing five routing policies on the same GPUs. It includes a shuffled experiment runner, workload fingerprints, strict metric checks and a preregistered crossover criterion.

**Status (2026-10-12): no saturation regime reached at the tested settings.** Two valid 15-run pilots, one with stress outputs forced to 512 tokens, never produced contention under hard affinity: R2 worker queue 0.01 ms, 0 preemptions. Per the pre-registered go/no-go rule, the main cohort was not run, and no crossover is demonstrated or refuted. The only measurable queueing came from round robin, whose lost locality recomputed 6.2× the prefill. [Findings →](docs/locality-vs-load-findings.md)

## Run and verify

Inspect existing evidence on a CPU machine:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/evaluate_preregistration.py results/2026-10-12-runpod-a40/main
python scripts/routing_evaluate.py results/2026-10-12-runpod-2x-rtxpro4500/day2-main
python scripts/routing_followup_evaluate.py sens results/2026-10-12-runpod-2x-l40s/day2-sens
python scripts/routing_followup_evaluate.py supp results/2026-10-12-runpod-2x-l40s/day2-supp
```

Preview the proposed GPU crossover run **without launching a server**:

```bash
DRY_RUN=1 STUDY_PHASE=pilot bash scripts/run_load_crossover.sh
DRY_RUN=1 STUDY_PHASE=main bash scripts/run_load_crossover.sh
```

For live GPU setup, use the [recorded routing environment](env/runpod_routing_setup.sh) as a **host-specific reference** and follow the [load-crossover runbook](docs/locality-vs-load.md). Every real run requires a compatible GPU driver/runtime and a pinned model revision; don't assume the setup works on arbitrary pods.

## Project layout

| Area | Contents |
|---|---|
| [`workloads/`](workloads/) | Deterministic multi-turn prompt, session and timing generator |
| [`scripts/`](scripts/) | Cluster lifecycle, routing matrix, strict analysis, figures and experimental controls |
| [`docs/cache-residency-findings.md`](docs/cache-residency-findings.md) | Single-worker measurements and caveats |
| [`docs/routing-findings.md`](docs/routing-findings.md) | Multi-replica policy comparison, router diagnosis and capacity controls |
| [`docs/engineering-notes.md`](docs/engineering-notes.md) | Reproducibility notes, integration problems and provenance |
| [`docs/protocols/`](docs/protocols/) | Dated, frozen hypotheses kept as a scientific audit trail |
| [`results/`](results/) | Untouched historical request logs, manifests, server logs, Prometheus snapshots and plots |

Historical raw-result directories intentionally retain their original recorded names and commands. That preserves reproducibility rather than rewriting what happened.

## Upstream and limitations

**NVIDIA Dynamo:** [Issue #16194 — KV-router load/locality tradeoff](https://github.com/ai-dynamo/dynamo/issues/16194). The report includes the failure reproduction, internal cost calculations and sensitivity evidence. It is a measured *workload-dependent tradeoff*, not proof of a universally incorrect router policy.

All reported traffic is synthetic and closed-loop; there are no production agent traces, cluster-wide load tests, multi-node KV transfers, or engine-level retention changes. Client-observed TTFT includes frontend overhead, and the five repetitions are repeated schedules on the same pod—not independent traffic populations. Do not interpret them as statistical generalization beyond this setup.

Feedback, alternative traces and reproducible counterexamples are welcome in [Issues](https://github.com/dundysm/inference-under-pressure/issues).
