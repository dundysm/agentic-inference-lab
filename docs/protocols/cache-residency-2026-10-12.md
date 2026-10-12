# Main cohort pre-registration

**Written:** 2026-10-12, after the arm-C budget scan and the A–D pilot, **before** any main-cohort run.

## Frozen configuration
- **Hardware:** RunPod on-demand pod `6sp3oabp0owevz`, 1× NVIDIA A40 (46 GB), driver 570.211.01. The same pod is used for every run.
- **Software:** vLLM `0.31.0+cu129`, torch `2.13.0+cu129`, torchcodec `0.17.0+cpu`. The full venv freeze is recorded in each run's `environment.txt`.
- **Model:** `Qwen/Qwen3-4B-Instruct-2507`, model and tokenizer revision `cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, `--max-model-len 8192`.
- **Harness:** v3 bundle, unmodified. Run with `REPS=5`, `ARMS=ABCD`, `LINES=80`, `TURNS=12`, `PAUSE=0.5`, `max_tokens=32`.
- **KV budgets:** `4831838208` (4.5 GiB, 32,768 tokens; tight) and `10737418240` (10 GiB, 72,816 tokens; roomy). They were chosen from the pilot because they sit clearly on each side of arm C's ~54K-token working set.
- **TTFT SLO for goodput:** 3,000 ms (illustrative).
- **Results directory:** `/workspace/results/main`.

## Hypotheses
Each one is evaluated per repetition as a paired ON−OFF difference within the same budget.

- **H1:** At both budgets, B beats A: the median resumed-turn TTFT is lower and computed prefill tokens are lower.
- **H2:** At 4.5 GiB, the median resumed cached fraction in C is below 0.05, and the median resumed TTFT for C−D is within ±10% of zero. This is idle-prefix eviction with no benefit, and it should appear with zero preemptions.
- **H3:** At 10 GiB, the median resumed cached fraction in C is above 0.9, the median resumed TTFT for C is below 0.5× that of D, and the wall time for C is below that of D.

## Rules
- Every run is kept.
- A run with failures is reported, not dropped.
- If any run aborts, the whole cohort is rerun rather than patched.
- Turns within a session are not treated as independent replicates. The unit is one repetition.

---

## Note added after the runs (not part of the pre-registration)
- The first attempt at this cohort aborted after 4 runs when log writes on `/workspace` (a network filesystem) were lost.
- Per the rule above, it was discarded, and the full cohort was rerun with results written to local disk (`/root/results/main`) and copied off the pod afterwards.
- The configuration, budgets and hypotheses were otherwise unchanged.
