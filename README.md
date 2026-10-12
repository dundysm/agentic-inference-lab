# Agentic Inference Lab — Day 1

**Question:** How do predictable multi-turn agent-like conversations behave when the idle-prefix working set exceeds available GPU KV cache?

This is a **measurement harness**, not a new cache policy and not yet a claim of performance improvement.

- RunPod, **one GPU**, one vLLM worker, Qwen3-4B-Instruct-2507.
- Four arms: prefix cache ON/OFF at low/high session pressure.
- Explicit GPU KV budgets and clean restarts before every arm.
- Pre/post Prometheus snapshots, request-level cached tokens and timing, per-repetition plots.
- Working-set pressure sweep before committing to a main A–D comparison.

**Start here:** [DAY_1_GUIDE.md](DAY_1_GUIDE.md). The file includes the exact preparation, smoke, pilot, benchmark, and analysis steps.

**No GPU local test:** `DRY_RUN=1 REPS=2 bash scripts/run_baseline.sh` lists deterministic interleaved run order.

Source-controlled scripts have syntax and local non-GPU unit checks. **They have not been run against a live RunPod/vLLM GPU**, and no results are included in this package. Keep published results separate from local model caches or tokens.
