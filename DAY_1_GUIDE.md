# Agentic Inference Lab — Day 1, v3

**Revised October 12, 2026.** Step-by-step guide for a **single-GPU RunPod pilot**, followed by a reproducible prefix-cache experiment. It applies the v2 peer feedback. All expected GPU effects are hypotheses, **not measured results**.

## 0. The question and what you are building

> Does vLLM retain useful long prefixes across tool-like pauses when the total paused-session working set is larger than its GPU KV budget? What happens to reused prompt tokens, prefill recomputation, TTFT and goodput?

You will build a deterministic, *scripted* multi-turn client and benchmark vLLM with cache ON/OFF under low and high idle-session pressure. This is **not** an agent benchmark, not a new retention policy, and not yet a distributed routing experiment.

**Today's stopping point:** verified workload and metrics, an initial memory-budget scan, and (only if the scan works) one complete A–D pilot. Do not immediately spend money on 5 repetitions. Real GPU integration is still untested in this package.

**Files**

```text
agentic-inference-lab/
├── README.md
├── DAY_1_GUIDE.md                 ← this guide
├── requirements.txt
├── .gitignore
├── workloads/multi_turn.py       ← session replay + request-level metrics
├── scripts/run_baseline.sh       ← vLLM process group / budget-arm scheduling
├── scripts/scrape_metrics.sh     ← 1-second metric snapshots for context
├── scripts/analyze.py            ← per-turn reuse + paired results
├── tests/test_logic.py           ← CPU-only logic tests
└── docs/findings.md              ← fill only after real GPU runs
```

## 1. Before renting a GPU

Select **one RunPod on-demand GPU pod**, NVIDIA GPU with **at least 24 GB VRAM**. Prefer persistent `/workspace` storage and the same host throughout all comparison arms. Avoid spot interruption. Choose Secure Cloud if repeatability is the priority; Community Cloud is usable if you record the host configuration and accept additional variability. Do not change providers, GPU or CUDA/PyTorch environment mid-cohort.

For this first experiment use `Qwen/Qwen3-4B-Instruct-2507`, BF16/default KV dtype, one vLLM process, `--max-model-len 8192`. Estimated KV bytes/token for model architecture is `2 × 36 layers × 8 KV heads × 128 head dim × 2 B ≈ 147,456 B = 144 KiB`; these estimates are **not** observed block allocations. Measure actual prompt lengths and cache capacity on the pod.

Why `8192` instead of v2's `16384`? A smaller maximum context keeps the minimum single-sequence KV requirement low enough to test tighter budgets. Keep it identical across all arms.

### A. Install and pin the serving environment

In the **pod shell**, not local Windows:

```bash
cd /workspace
unzip agentic_inference_lab_day1_v3.zip -d .
cd agentic-inference-lab
export HF_HOME=/workspace/hf-cache
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Only install vLLM into a CUDA/PyTorch environment that supports it.
python -m pip install 'vllm==0.31.0'

nvidia-smi
vllm --version
vllm serve --help | grep -E 'kv-cache-memory-bytes|enable-prefix-caching|enable-prompt-tokens-details|tokenizer-revision|max-model-len'
python -m unittest discover -s tests -v
```

vLLM 0.31.0 exists (released October 5, 2026). Version pinning does **not** guarantee binary compatibility with your selected RunPod image. If the image comes with another working vLLM build, freeze that version instead, confirm every required CLI flag, and document the difference. Do not overwrite a working CUDA stack blindly.

Useful docs: [vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/), [vLLM v0.31.0 metrics](https://docs.vllm.ai/en/v0.31.0/design/metrics/), [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [vLLM 0.31.0 release](https://pypi.org/project/vllm/0.31.0/).

### B. Version-lock the model once, before the sweep

The runner automatically resolves the current Hugging Face `main` to a **40-character commit SHA** once before a sweep, pre-downloads that exact revision, then supplies **both** `--revision` and `--tokenizer-revision` to every server launch. It records the SHA in `model-lock.txt` and every `server_config.json`.

For strongest repeatability, pin a known SHA yourself (example **placeholder**, not a real commit):

```bash
export MODEL=Qwen/Qwen3-4B-Instruct-2507
# MODEL_REVISION must be a real 40-character HF commit SHA.
# Optional: resolve once, then re-use the value for the full experiment.
export MODEL_REVISION=$(python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("Qwen/Qwen3-4B-Instruct-2507").sha)
PY
)
echo "$MODEL_REVISION"
hf download "$MODEL" --revision "$MODEL_REVISION" >/dev/null
```

Do not time model download or CUDA-graph capture as part of the workload. Keep `MODEL_REVISION` the same for pilot and main. The runner refuses invalid/missing revision resolution; it never silently runs a moving `main`.

## 2. Check the runner before spending GPU time

On a local Linux shell or in the pod, this prints the full deterministic schedule **without** CUDA or Hugging Face access:

```bash
DRY_RUN=1 REPS=2 bash scripts/run_baseline.sh
```

Each repetition randomizes all budget–arm pairs together. That controls time drift better than v2's “all tight, then all generous” order. Every completed arm has **its own freshly started vLLM process**, disjoint warmup and before/after metrics.

The cleanup routine kills the process group **started by this runner only**, and waits until GPU memory returns near the original idle baseline. It does not search for and kill arbitrary vLLM processes. If a child EngineCore still owns memory or another process starts competing for VRAM, it **aborts instead of claiming a clean comparison**. This assumes a single-GPU pod; inspect actual usage if it aborts.

Do not run a second vLLM server on port 8000. Do not benchmark through `*.proxy.runpod.net`—the client must be inside the pod talking to `127.0.0.1`.

## 3. Run a manual smoke test

In a pod terminal:

```bash
vllm serve "$MODEL" --revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION" \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 \
  --kv-cache-memory-bytes 3221225472 \
  --enable-prefix-caching --enable-prompt-tokens-details \
  2>&1 | tee /workspace/manual-vllm.log
```

In a second pod terminal, with the venv activated:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/metrics | grep -E 'prefix_cache|prefill_kv_computed|kv_cache_usage' | head -30
python workloads/multi_turn.py --scenario smoke --sessions 2 --turns 3 \
  --lines 40 --pause 0.3 --max-inflight 2 --out-dir results/manual-smoke
```

Examine six rows in `results/manual-smoke/requests.jsonl` and the two Prometheus snapshots. Check that the final SSE usage actually supplies `prompt_tokens_details.cached_tokens`. vLLM's support for this field has changed across releases, so **verify actual outputs**. Zero means a measured zero; missing is *missing*, not zero. If `vllm:request_prefill_kv_computed_tokens_sum` is absent, the analyzer will leave that column blank.

**Manually stop the server**, then confirm GPU usage returns to its previous idle state before running the orchestration script. The runner will refuse to take over an existing healthy server on port 8000.

## 4. Understand the four arms

| Arm | Cache | Sessions | In-flight | Turns/session | Main control |
|---|---|---:|---:|---:|---|
| **A** | OFF | 4 | 2 | 12 | Low session pressure without reuse |
| **B** | ON | 4 | 2 | 12 | Benefit when reusable prefixes fit |
| **C** | ON | 16 | 4 | 12 | Paused-session working set stresses cache |
| **D** | OFF | 16 | 4 | 12 | High-session control without reuse |

`--lines 80`, `--pause 0.5 s` and `max_tokens=32` are fixed unless the pilot shows an invalid setup. Each session is sequential. Requests from different sessions share a semaphore limiting concurrent *active* requests. A “tool pause” occurs outside that semaphore, so most completed-session blocks are idle and reclaimable.

**Important experimental limitation:** This uses client throttling. In C/D, 16 sessions can produce a wait of their turn at the 4-slot harness semaphore, but the server sees at most 4 simultaneous requests and may have little internal queueing. The harness reports two separate clocks:

- `ttft_ms`: from semaphore admission to first SSE **content** token (excludes harness wait).
- `user_ttft_ms`: harness wait **plus** `ttft_ms` (not a real end-user RTT over the internet).

The first turns of all sessions start close to simultaneously—an initial cold burst. Low and high arms also change both session count and concurrency. This is a **screening experiment**, not a complete isolation of independent arrival rate, context length and concurrency effects.

Warmup uses `--session-offset 9000 --system-tag '[warmup-not-measured]'`. Its prompts differ from the measured session IDs 0–15, including the system message and document body. The tiny shared common prefix is harmless but may still share token blocks; it is not reused session 0's long incident document as v2 accidentally did.

The workload SHA-256 now covers *all intended message transcripts across all turns*, model and generation cap—not just the first document. It is stored on every request and in `workload_manifest.json`.

## 5. Budget scan: do this BEFORE the full A–D pilot

A nominal 3 GiB cache holds about 21.8k KV tokens, 6 GiB about 43.7k and 8 GiB about 58.3k, **if** the estimated 144 KiB/token is accurate. Workloads may have ~45–50k tokens in distinct idle prefixes at 16 sessions, but **do not assume** line counts represent exact token lengths.

Run **only arm C** at 4 exploratory budgets. This is a pilot selecting the memory-pressure range, not a publishable proof:

```bash
RESULTS_DIR=results/pilot \
ARMS=C REPS=1 \
KV_BUDGETS='2684354560 3221225472 4831838208 6442450944' \
MODEL_REVISION="$MODEL_REVISION" \
bash scripts/run_baseline.sh

python scripts/analyze.py --results results/pilot --slo-ms 3000
```

These are 2.5, 3, 4.5 and 6 GiB, respectively. If a budget cannot hold **one maximum-length context** or exceeds the pod's free VRAM, adjust `MAX_MODEL_LEN` or budgets **before** the main run, without changing only some arms. The default context length in the runner is 8192, but no capacity formula supersedes the live startup logs.

Examine:

```bash
find results/pilot -name requests.jsonl -o -name server.log
find results/pilot -name kv_allocation.log -exec sh -c 'echo "$1"; cat "$1"' _ {} \;
python - <<'PY'
import glob, json
for fn in sorted(glob.glob('results/pilot/*/rep_*/*/requests.jsonl')):
  rows=[json.loads(line) for line in open(fn)]
  resumed=[r for r in rows if r.get('turn',0)>0 and r.get('success')]
  valid=[r for r in resumed if isinstance(r.get('cached_tokens'),int) and isinstance(r.get('prompt_tokens'),int) and r['prompt_tokens']>0]
  f=sum(r['cached_tokens']/r['prompt_tokens'] for r in valid)/len(valid) if valid else None
  print(fn,'resumed',len(resumed),'observed cache fraction',f,'coverage',f'{len(valid)}/{len(resumed)}')
PY
```

**Checkpoint:** At a sufficiently generous budget, cache-on resumed turns should show reuse. At a tighter budget, the median resumed cache fraction may drop while recomputed prefill tokens increase. But a collapse is not guaranteed. Do **not** infer eviction from low GPU utilization, missing preemption counters or a single bad p95. If effects don't materialize, inspect prompt sizes, cache capacity, scheduling and tool timing; modify only the **pilot** to locate a genuine pressure regime.

Select **two feasible budgets** for the main sweep—one with sustained reuse, one with materially lower reuse if such a regime exists. Two are enough for Day 1; a third around the crossover is worthwhile if time/compute permits. Predeclare the choice, the 3-second illustrative TTFT SLO (or your own value), and hypotheses in `docs/findings.md` **before seeing main results**.

## 6. Full A–D pilot, then repeat if valid

Use **new directories** for each cohort. The runner will refuse to overwrite existing `requests.jsonl`.

```bash
# Replace examples with two budgets selected from the scan.
RESULTS_DIR=results/ab-pilot ARMS=ABCD REPS=1 \
KV_BUDGETS='3221225472 6442450944' MODEL_REVISION="$MODEL_REVISION" \
bash scripts/run_baseline.sh

python scripts/analyze.py --results results/ab-pilot --slo-ms 3000
```

Inspect the generated plots and CSVs. If the runs have errors, missing cache data, inconsistent capacity or contamination, fix the harness and **redo the full cohort**, not selective arms. The output should contain warmup and `server.log` in every run directory, and `server_config.json` even if the workload fails. Check `environment.txt` for version consistency and `kv_allocation.log` for startup capacity evidence.

Only after validating the pilot:

```bash
# Five independent restarts per arm/budget = 40 launches; requires substantial GPU time.
RESULTS_DIR=results/main ARMS=ABCD REPS=5 \
KV_BUDGETS='3221225472 6442450944' MODEL_REVISION="$MODEL_REVISION" \
bash scripts/run_baseline.sh
python scripts/analyze.py --results results/main --slo-ms 3000
```

Never silently mix different model SHAs, GPU types, prompts, code revisions or budgets within one analysis cohort. The directory list and `model-lock.txt` support auditing this.

## 7. Interpreting the output

`python scripts/analyze.py --results results/main --slo-ms 3000` creates:

| Artifact | What it establishes |
|---|---|
| `analysis/summary.csv` | **One row per independent run**: TTFT, cached fractions, computed prefill, failure status and goodput |
| `analysis/paired_comparisons.csv` | Paired **cache ON minus cache OFF** differences for B–A and C–D within each budget/repetition |
| `analysis/resumed_cache_fraction_by_turn.png` | **Primary evidence:** per-turn cached fraction; point = within-repetition median, line = median across rep medians |
| `analysis/resumed_ttft.png` | Resumed TTFT with individual-turn context and **repetition median** markers |
| `analysis/cache_reuse.png` | Secondary server-level hit ratio by budget |

For `paired_comparisons.csv`, a *negative* `ttft_median_delta_ms` or `computed_prefill_tokens_delta` means cache ON was lower than cache OFF at that budget; a *positive* `goodput_rps_delta` means cache ON met more requests/sec under the predeclared TTFT threshold. These are **descriptive paired differences**, not a claim of statistical significance. A run with errors is excluded from the pair file.

**Goodput caveat:** `goodput_rps_at_slo` uses client first-content TTFT from admission; `user_goodput_rps_at_slo` includes harness wait. Neither is a real operator-defined end-to-end workflow completion SLO. P95 is suppressed for fewer than 50 resumed requests **within a run**. Avoid treating multiple turns of the same session as independent experimental replications.

Cached-token fraction is missing if the server did not report `prompt_tokens_details.cached_tokens`; the analyzer does **not** convert it to zero. The same applies to `request_prefill_kv_computed_tokens_sum` if absent. The aggregate prefix hit ratio includes cold turns; therefore never substitute it for the resumed-turn plot.

The workload is prefill-dominated (`max_tokens=32`), so output-token throughput is not the main result. Document kernel/queue contention if present. Pressured cache blocks of finished requests may count as free, so `kv_cache_usage_perc` and `num_preemptions` cannot by themselves confirm retention or eviction.

## 8. Publishing standards

Fill in [docs/findings.md](docs/findings.md), linking raw data, exact model and code SHAs, vLLM/cuda/gpu versions, scenario configs and all failure cases. Include negative or inconclusive results. Do not put private Hugging Face credentials, pod secret tokens, environment variables, or sensitive prompts in GitHub. Keep the full sampled Prometheus time series out of Git until curated.

**Don't claim** this experiment reproduces real Claude Code/AgentX traffic, proves MORI vs Continuum, or shows anything about multi-replica routing. This synthetic single-GPU experiment is foundation work.

If repeated results support a real idle-prefix eviction or contention mechanism, Day 2 is two vLLM replicas behind NVIDIA Dynamo, comparing load-only and KV/session-aware routing. Only then select a concrete upstream performance improvement based on observations.

## 9. Final go/no-go checklist

- [ ] Pod, GPU and serving runtime are stable and recorded (RunPod on-demand, no proxy path).
- [ ] vLLM CLI options and streamed usage `cached_tokens` verified on the **actual pod**.
- [ ] Model and tokenizer SHA pinned, pre-downloaded and recorded.
- [ ] Warmup has a distinct system tag and session ID namespace.
- [ ] Every restart returns to the original GPU idle-memory baseline before continuing.
- [ ] KV budgets interleaved with arms, not clustered by budget.
- [ ] Actual prompt token lengths, KV allocation capacity and active footprint inspected.
- [ ] Per-turn reuse plot and paired comparisons exist and have non-missing data where expected.
- [ ] Failures and missing metrics never converted to valid zero-valued observations.
- [ ] Main cohort uses immutable choices made after **pilot** but before main measurements.
- [ ] `docs/findings.md` explains what was established, what wasn't, and limitations.

### Verification disclosure

The v3 code has CPU-side syntax and logic checks and a deterministic schedule dry-run. **It has not been executed against a live vLLM server or NVIDIA GPU.** Any RunPod compatibility issues, vLLM metric changes or actual capacity limits must be resolved and documented in the first pilot. No benchmark result is implied by this guide.
