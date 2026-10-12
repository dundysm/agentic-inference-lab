# Agentic Inference Lab — Day 2
## Two-replica KV-aware routing: a measured inference-systems experiment

**Status:** Implementation and measurement guide — not yet validated on a two-GPU pod  
**Date:** October 12, 2026  
**Prerequisite:** Completed Day 1 v3 and the saved Day 1 RunPod results  
**Recommended environment:** One RunPod on-demand pod with **2 × NVIDIA A40** (or two identical NVIDIA GPUs), NVIDIA Dynamo + vLLM, and a shell within the Dynamo-compatible runtime  
**Estimated scope:** One controlled routing experiment; no new caching algorithm, no simulator, no disaggregated prefill/decode, no P2P KV transfer

---

## 0. Today's objective

**Question:** With the *same two GPUs and the same aggregate 9 GiB KV budget*, how much does the choice of request router affect the ability to reuse an agent's KV cache across multi-turn tool pauses?

Day 1 already showed that automatic prefix caching alone is not enough when many idle session prefixes compete for a small cache. Day 2 asks whether **placing successive turns on the correct worker** changes that outcome.

**Definition of done:** A two-worker Dynamo-vLLM deployment, a deterministic agent-like workload that carries stable session IDs, four routing configurations, per-worker evidence of routing/cache behavior, three or more valid repetitions, and an honest written conclusion. **A speedup is not required.**

Do *not* confuse “two GPUs are faster than one” with “this routing policy is better.” **Every routing comparison uses the same two GPUs**, same model, same memory per worker, same workload, and same client throttling.

### Actual Day 1 reference point

Source: the supplied `agentic-inference-lab-results.zip`, especially `main-2026-10-12/main/analysis/summary.csv`, `findings.md`, and the run manifests.

| Setting | Day 1 value |
|---|---|
| Hardware | 1 × NVIDIA A40, 46 GiB reported GPU memory, RunPod |
| vLLM | `0.31.0+cu129` |
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Model and tokenizer revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Precision and context | BF16, `--max-model-len 8192` |
| Agent-like workload | **16 sessions × 12 turns = 192 requests/run** |
| Prompt | Unique deterministic 80-line incident log per session; scripted append-only turns |
| Tool pause / generation | 0.5 seconds between turns; max 32 output tokens; temperature 0 |
| Admission | At most 4 in-flight requests, client-side semaphore |
| Cache budget that thrashed | 4.5 GiB (`4,831,838,208` bytes); resumed cached fraction ~0.5% |
| Cache budget that retained prefixes | 10 GiB (`10,737,418,240` bytes); resumed cached fraction ~98.9% |
| Approximate median resumed TTFT, cache ON | ~581 ms at 4.5 GiB versus ~67 ms at 10 GiB |
| Main-cohort sample | 40 runs × (48 or 192 requests according to arm) = **4,800 requests** |

These are **one-host, deterministic-workload** results, not a population estimate for all agent traffic. The Day 1 finding also concerns idle-prefix *residency*, not necessarily active-request preemption; zero preemptions was recorded.

**Day 2 primary resource choice:** **2 × 4.5 GiB KV pools = 9 GiB total**. This allows the same 16 distinct sessions to be distributed across workers. It is an *experimental hypothesis*, not a guarantee, that about eight sessions per worker will be easier to cache than 16 on one worker.

> **Important comparability caveat:** Dynamo's stock aggregated KV-routing example uses a **64-token block size** and a small demonstration model. Day 1 used the native vLLM configuration, so **Day 1 and Day 2 are not automatically a strictly matched apples-to-apples latency comparison**. Keep the KV block size, model revision, runtime version, and other worker settings *identical within Day 2*. Treat Day 1 numbers as motivation and a historical reference. To make a rigorous one-versus-two-GPU comparison, re-run the single-worker configuration using matching Day 2 software and block settings.

---

## 1. Hypotheses — record before the first run

- **H1 — Locality:** With the same two workers and total KV memory, some form of session-aware or KV-aware placement may preserve more resumed-turn prefix cache than request-level round robin.
- **H2 — Performance mechanism:** If a routing strategy preserves more prefixes, the same workload should show fewer computed prefill tokens and lower resumed-turn server TTFT. If it does not, investigate tokenizer/hash mismatches, stale router state, and scheduling contention.
- **H3 — Load-balancing cost:** Hard affinity may improve locality yet cause load imbalance, worse queueing, or longer full-workload completion time.
- **H4 — KV-aware versus hard affinity:** A KV-aware cost model may approximate the benefit of explicit session affinity; hard affinity is not automatically superior.

**Falsification is allowed:** If every router behaves similarly, the honest finding is that this simple workload does not differentiate routing policies. If a mode improves cache hits but harms completion time, report that tradeoff.

Before the main cohort, write the chosen SLO and the hypothesis wording into `docs/day2_findings.md` and commit the file. Example exploratory **TTFT SLO: 3,000 ms**, carried over from Day 1; do not pretend it is a production SLA.

---

## 2. Hardware, software, and version lock

### 2.1 Provision one two-GPU pod

- **Recommended:** one RunPod *on-demand* pod with 2 × A40s, if available. Same GPU type, one host, two independent workers, **one GPU per worker**. Do not use tensor parallelism (`--tensor-parallel-size 2`): that is one model sharded across two GPUs, **not two replicas**.
- Keep the model/HF cache, repository, raw results, and environment manifests on persistent `/workspace` storage.
- Execute the load generator **inside the same pod**, targeting `http://127.0.0.1:8000`; do not benchmark through the RunPod public HTTP proxy.
- Do not expose etcd, NATS, worker metrics, or the frontend publicly. If the pod template exposes services by default, restrict access appropriately.
- Check both GPUs with `nvidia-smi -L`; explicitly log the PCIe topology with `nvidia-smi topo -m`. Inter-GPU transfer is **not** enabled in this experiment.

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv
```

**Hard stop:** If there aren't two equivalent usable GPUs, do not silently substitute one tensor-parallel model or two workers contending for one GPU.

### 2.2 Use a pinned NVIDIA Dynamo runtime

Dynamo is not simply `pip install dynamo` into an arbitrary Day 1 vLLM image. It needs its matching frontend/worker build and its service-discovery dependencies. **Start from the NVIDIA-supported Dynamo runtime/container instructions for the release you actually deploy**.

- Use **a released/pinned Dynamo tag or immutable Git commit**, not floating `main`.
- The stable v1.5.1 docs, available at the time this guide was written, cover the two-worker `agg_router.sh` path. The `dev` documentation may describe features that are not available in your pinned runtime.
- Use the **vLLM version bundled with that Dynamo runtime** unless you have independently verified integration compatibility. Do **not** force `vllm==0.31.0` into a mismatched Dynamo image merely to match Day 1.
- Record Dynamo Git SHA, image digest, vLLM version, Python/CUDA/driver versions, model revision, CUDA visibility, and relevant runtime flags for every arm.
- Pre-download the model once to prevent download or compilation variance between experimental arms. Exact commands depend on the container image and HF tooling.

At the end of setup:

```bash
# From a Dynamo runtime shell with a checkout of the matching Dynamo source:
export DYNAMO_HOME=/workspace/dynamo  # adjust to actual checkout root
cd "$DYNAMO_HOME"
git rev-parse HEAD
python -m dynamo.frontend --help | grep -E \
  'router-mode|session-affinity|router-policy-config' || true
python -m dynamo.vllm --help | grep -E \
  'model|revision|kv-cache-memory-bytes|block-size|kv-events-config' || true
python -m vllm --version || vllm --version
```

This is a **capability check**; the guide cannot certify which options your selected container actually accepts. If a required option is missing or renamed, use the release's documented equivalent and record the adjustment. **Do not launch a partial comparison with different worker budgets.**

### 2.3 Required worker invariants

Set these to the same values on **both** replicas **and across every routing arm**:

```bash
export MODEL='Qwen/Qwen3-4B-Instruct-2507'
export MODEL_REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'
export KV_BYTES='4831838208'  # 4.5 GiB PER WORKER
export MAX_MODEL_LEN='8192'
export BLOCK_SIZE='64'        # match Dynamo router + vLLM workers
export PYTHONHASHSEED='0'     # consistent KV block hashing
```

Use `--revision "$MODEL_REVISION"` and `--tokenizer-revision "$MODEL_REVISION"` **if supported by the deployed worker**. Use BF16 if both workers/runtime accept it. Disable CPU/disk offload, P2P KV transfer, and speculative decoding; do not introduce policies you are not testing. Enable prefix caching, prompt-token details, and KV event publishing if supported.

**KV budget is in bytes; total across two workers is 9 GiB.** KV block count/capacity must be verified in each server's startup log; the nominal byte budget alone is insufficient.

**Hashing requirement:** Dynamo's documented KV router requires consistent token/block hashing across workers; keep `PYTHONHASHSEED=0` everywhere, with the same tokenizer, cache key configuration, prompt formatting, and **64-token block size** in both workers and router. A zero-overlap router paired with a healthy worker-local cache is a *setup failure to debug*, not evidence that KV-aware routing is bad.

---

## 3. Bring up the official two-worker path FIRST

This is an **integration smoke test**, not a benchmark. NVIDIA provides a two-vLLM-worker aggregated KV-routing launch example. Verify the version-pinned source before changing it.

**Official references:**

- [Dynamo v1.5.1: KV-aware routing local guide](https://docs.nvidia.com/dynamo/v1.5.1/cli/kv-aware-routing/overview)
- [Dynamo v1.5.1: vLLM backend overview](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/modular-components/backends/vllm/overview)
- [Official `agg_router.sh` reference](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_router.sh) — **main is unpinned, read only as an example**
- [Router configuration and session affinity](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/modular-components/router/configuration-and-tuning)

Inside a supported Dynamo environment, from the checkout root:

```bash
cd "$DYNAMO_HOME"
docker compose -f dev/docker-compose.yml up -d
# In the Dynamo runtime shell (the repo's release-specific example uses these deps):
cd "$DYNAMO_HOME/examples/backends/vllm"
bash launch/agg_router.sh
```

The upstream **stock** script currently demonstrates 2 GPUs, a small Qwen model, a KV-aware frontend, worker KV events and deterministic hashing. It is useful to confirm discovery and communication, **not** suitable unchanged for the actual experiment. In particular, `--enforce-eager` in an example launcher can change performance; keep CUDA-graph policy consistent across the eventual measured workers.

Inspect its startup logs:

1. Exactly **two** vLLM workers are registered, each assigned to a different GPU.
2. The frontend accepts OpenAI-compatible streaming chat completions.
3. Both workers publish KV events; the router consumes them rather than merely guessing cache state.
4. Both workers advertise the *same model deployment*, tokenizer and block size.
5. The complete toolchain can shut down cleanly with no orphaned vLLM `EngineCore` or GPU allocations.

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/metrics | head -25
```

The frontend metrics and each worker's metrics **may be exposed on different ports**. The example uses distinct `DYN_SYSTEM_PORT` values for worker services; determine the actual Prometheus URLs **from the live process/help/logs** and test them. Do not assume the Dynamo frontend's `/metrics` is the sum of two worker-local vLLM metric sets.

**Checkpoint 1:** the official two-worker smoke test succeeds. Then stop it and verify that both GPUs return to their observed idle memory baselines.

---

## 4. Build an experiment launcher — do not benchmark the stock example

Create `scripts/day2_launch.sh` or a small Python process supervisor. Copy the pinned release's supported worker command shape and maintain one source of truth for worker flags.

The **reference shape** from Dynamo's aggregated two-worker example is:

```bash
# Reference commands; adapt to the pinned Dynamo runtime and validate --help.
# Run inside the correct runtime, after etcd/NATS have been started.
export PYTHONHASHSEED=0

# Frontend — select ROUTER_MODE and optional AFFINITY_ARGS per arm.
python -m dynamo.frontend \
  --http-port 8000 \
  --router-mode "$ROUTER_MODE" "${AFFINITY_ARGS[@]}" \
  >"$RUN_DIR/frontend.log" 2>&1 &

# Worker 0, GPU 0. Must also receive identical model/runtime flags below.
DYN_SYSTEM_PORT=8081 CUDA_VISIBLE_DEVICES=0 \
  python -m dynamo.vllm \
  --model "$MODEL" --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --block-size "$BLOCK_SIZE" --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' \
  >"$RUN_DIR/worker0.log" 2>&1 &

# Worker 1, GPU 1. Use different service, event, and side-channel ports.
DYN_SYSTEM_PORT=8082 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=1 \
  python -m dynamo.vllm \
  --model "$MODEL" --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --block-size "$BLOCK_SIZE" --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' \
  >"$RUN_DIR/worker1.log" 2>&1 &
```

**This block is a launch template, not a claim of verified execution.** Some Dynamo releases add wrapper parameters or reject a forwarded vLLM option. Follow the pinned `agg_router.sh` for required env vars, backend ports, dependency startup, and worker registration; use the flags above only after verifying compatibility. Production-grade shell supervision should use process groups, PID files, health checks, and cleanup traps—not these bare `&` launches alone.

### Process supervision checklist

- Restart **frontend and both workers for each measured run** so no previous KV cache or affinity binding survives into the next arm.
- Prefer a single process group/container-level supervisor; terminate children and wait for both GPUs to return to the pre-run idle level.
- Fail fast if either worker's startup KV allocation differs or if the frontend has fewer than two ready workers.
- Persist `frontend.log`, `worker0.log`, `worker1.log`, `manifest.json`, `environment.txt`, and optional metrics sampling even after failures.
- Record exact launch argv **before** sending benchmark traffic.
- Use a consistent, disjoint warm-up workload on **both** workers. One global warm-up request can leave only one worker initialized; explicitly verify initialization separately without polluting measured prefixes.

**Checkpoint 2:** a single validated worker configuration runs on each GPU and the frontend can send requests to both.

---

## 5. The main routing matrix

No extra retention policy or memory tier is introduced today. Every row has:

- 2 × A40 workers
- 4.5 GiB KV per worker (9 GiB total)
- same model revision, vLLM/Dynamo versions, context, block size, cached-prefix policy, and tool-pause workload
- same number of active requests admitted by the client

| Arm | Frontend routing | Session affinity | Why run it |
|---|---|---|---|
| **R0** | `--router-mode round-robin` | disabled | Naive request-level placement baseline |
| **R1** | `--router-mode kv` | disabled | Router uses actual KV events + load cost |
| **R2** | `--router-mode round-robin` | **hard**; `--router-session-affinity-ttl-secs 300` | Session stickiness independent of overlap scoring |
| **R3** | `--router-mode kv` | **hard**; `--router-session-affinity-ttl-secs 300` | Interaction: smart first placement, then sticky session |

On the pinned runtime, verify `python -m dynamo.frontend --help` recognizes each mode/TTL combination. **The header alone does not enable affinity.** The affinity TTL is idle time *since the request lease completes*, not the same thing as a vLLM KV retention TTL. Set 300 s here to comfortably cover 0.5-second tool pauses and scheduling gaps; log the choice. With LoRA enabled, some mode+affinity combinations are explicitly prohibited, so keep LoRA disabled. See the [stable routing docs](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/modular-components/router/configuration-and-tuning).

**Why include both R2 and R3?** Hard affinity may override subsequent KV-aware re-ranking once a session is bound. Thus the two may be similar after turn 0. If they are, that is a useful result—not a reason to remove the control.

**Optional R4 — soft affinity:** Only after checking that your exact binary exposes `--router-session-affinity-mode soft` and documenting its behavior. Some `dev` documentation describes this mode while v1.5.1 stable documentation does not; **do not assume feature availability from a later branch**. If you change versions to enable it, collect a new full cohort rather than comparing with old R0–R3 runs.

### Executable frontend configuration examples

```bash
# R0
python -m dynamo.frontend --http-port 8000 --router-mode round-robin

# R1
python -m dynamo.frontend --http-port 8000 --router-mode kv

# R2 (only if accepted by your pinned version)
python -m dynamo.frontend --http-port 8000 --router-mode round-robin \
  --router-session-affinity-ttl-secs 300

# R3
python -m dynamo.frontend --http-port 8000 --router-mode kv \
  --router-session-affinity-ttl-secs 300
```

The worker processes stay configured identically in every mode. `--router-mode kv` is not sufficient unless workers actually emit KV cache events and the frontend receives them.

---

## 6. Update the Day 1 workload client

**Reuse** the already tested `workloads/multi_turn.py`. Do not write a separate generator that accidentally changes prompt prefixes or timing. Make the smallest possible additions in a Day 2 branch.

### 6.1 Send a stable session header

The OpenAI-compatible request must carry:

```python
headers = {"X-Dynamo-Session-ID": f"ailab-session-{sid:04d}"}
```

Pass `sid` through `query()` from `run_session()` and send the header on **every turn**, including turn 0. Keep `sid` stable across the 12 turns; different sessions have different IDs. Use IDs unique to the experiment run, or restart the frontend and workers between arms (recommended). Add the request's `session_id` and `route_arm` to its JSONL row.

Minimal conceptual patch:

```python
# before (Day 1)
await client.stream("POST", url, json=body)

# after (Day 2) — illustrative; retain the existing async context manager
async with client.stream(
    "POST", url, json=body,
    headers={"X-Dynamo-Session-ID": session_id},
) as response:
    ...
```

**Don't insert the session ID into the textual prompt**; that would change token prefixes and could affect cache behavior. Use the transport header only.

### 6.2 Keep the existing timing definitions

Preserve the meanings of:

- `request_ready_unix` — when a request was ready before the client semaphore
- `start_unix` — after client-side admission
- `harness_wait_ms` — ready-to-admitted wait
- `ttft_ms` — admitted-to-first streamed *content* token, including frontend/network-on-localhost/server processing
- `user_ttft_ms` — client wait plus `ttft_ms`
- `end_unix`, `e2e_ms`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `success`

**Important:** routed Day 2 TTFT includes Dynamo frontend overhead; it is no longer exactly the single-worker Day 1 latency path. Compare **only against other Day 2 arms** for your primary routing conclusion.

### 6.3 Record actual worker placement

This is non-negotiable. Without it you might publish a routing analysis without knowing where requests were routed.

Try, in order:

1. A **documented request-to-worker identifier** emitted by your pinned Dynamo frontend's logs/tracing, correlated by request ID and session. Prefer structured tracing/OTel if available.
2. Backend logs with request IDs, joinable to the client's request IDs. Confirm IDs are propagated; don't just assume headers appear in backend logs.
3. For initial *sequential diagnostics only*, compare per-worker request-counter deltas before and after one request. This identifies placement with no concurrent traffic; it is **not** a general high-concurrency tracing solution.
4. If none is available, add a **clearly labeled debug-only correlation hook** at the router or worker and validate it. Turn off expensive per-request debug logging for timed main runs, but retain a lightweight unambiguous placement field.

Each JSONL row should contain `session_id`, `route_arm`, and `worker_id` **when proven**, not a guessed worker assignment. If worker assignment isn't observable, stop at an exploratory routing experiment; don't claim that session stickiness worked.

### 6.4 Collect per-worker metrics

The *frontend* metrics are useful for dispatch/queue behavior; the two *worker* metrics are essential to show cached-prefix reuse and computed prefill on each GPU.

Use `/metrics` endpoints verified from the live deployment and write:

```text
frontend_metrics_before.prom
frontend_metrics_after.prom
worker0_metrics_before.prom
worker0_metrics_after.prom
worker1_metrics_before.prom
worker1_metrics_after.prom
```

As in Day 1, calculate counter **deltas** only inside a single server lifetime. Check exact metric names using `prometheus_client.parser` or the live exposition; **do not** silently assume a Day 1 vLLM metric has the same name in the Dynamo-integrated build. Prometheus histograms' `_sum` is not a p95. Exclude the separate warm-up snapshots from measured totals.

`prompt_tokens_details.cached_tokens` on the final SSE usage record is the best per-request signal *if it survives Dynamo's proxy path*. If missing, record coverage explicitly and use validated worker-side counters/trace instrumentation; don't replace missing values with zero.

**Checkpoint 3:** One known session repeatedly hits the same worker with R2/R3, and both workers actually receive requests in at least one non-affinity arm.

---

## 7. Integration checks before spending money on repetitions

Do the following short tests before the 192-request pilot.

### Test A — two workers are really active

- Send a small number of unrelated requests with R0.
- Verify two distinct workers are selected by a trustworthy placement signal.
- Confirm each worker has its own GPU and independent KV allocation.
- Verify identical model revision, block size, tokenizer and prefix caching settings.

### Test B — hard affinity actually binds a session

- Start clean with R2; send three turns of one session ID, with a tool pause.
- Inspect route decisions; all should target the same worker as long as the worker is healthy and the TTL has not expired.
- Use a second independent session ID to confirm the router can bind another session separately.

### Test C — KV event path is alive

- Start R1 on fresh workers; send repeated long-prefix turns without a worker restart.
- Check that Dynamo ingests `BlockStored`/cache lifecycle events, that the router computes nonzero overlap where appropriate, and that the chosen worker shows corresponding worker-side cache usage/hits.
- If affinity and KV both appear ineffective, check `PYTHONHASHSEED`, tokenizer revision, block-size alignment, event transport, and the frontend's effective routing mode *before* interpreting timing numbers.

### Test D — cached tokens survive the frontend

- Check the last SSE usage frame for `prompt_tokens_details.cached_tokens`, or choose your validated fallback.
- Confirm request success and total token accounting.
- Confirm all 192 intended requests can be replayed without changing the prompt digest.

### Test E — cleanup and isolation

- Kill the run, check both GPU allocations drop to the measured idle baseline, and confirm no old EngineCore or frontend remains.
- Verify the next run starts with a fresh cache and fresh affinity mappings.

**Go/no-go:** If any of A–E fails, **fix integration first**. Do not turn unexplained frontend problems into benchmark results.

---

## 8. Run the main experiment

### 8.1 Pilot — one run per routing arm

For each R0–R3:

1. Write `manifest.json` containing arm, model revision, Dynamo/vLLM SHA/version, worker KV bytes, block size, process command lines, code revision and a shared prompt digest.
2. Shut down old workers/frontend and verify both GPUs are idle.
3. Start etcd/NATS if needed; start **both workers plus frontend** with the same worker configuration.
4. Wait for two registered workers and successful `/v1/chat/completions` health smoke.
5. Warm up with **disjoint prompt namespaces**, preserving Day 1's `--session-offset 9000 --system-tag '[warmup-not-measured]'` idea. Ensure both workers have been initialized. Do not prefill a measured session's unique incident document.
6. Capture separate frontend + worker pre-run metrics.
7. Run **16 sessions × 12 turns**, `lines=80`, `pause=0.5`, `max-inflight=4`, `max_tokens=32`, `temperature=0`, using the same generated session transcripts and stable session headers.
8. Capture after-run metrics, log/trace outputs and worker placement.
9. Verify **192/192 successful requests**; mark failed or partial runs invalid.
10. Stop the cluster and release resources.

Illustrative workload invocation **after** adding `--route-arm` and passing session headers to the client:

```bash
python workloads/multi_turn.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --scenario day2-r1 \
  --route-arm R1 \
  --sessions 16 --turns 12 --lines 80 \
  --pause 0.5 --max-inflight 4 \
  --out-dir results/day2/pilot/rep_1/R1
```

`--route-arm` **does not exist in Day 1 v3**: add it before using that exact command. Likewise, the whole project requires a new Day 2 launcher; the stock Day 1 `run_baseline.sh` starts one plain vLLM server and cannot be used unchanged.

### 8.2 Main cohort — repeat the comparison

Once all four pilot arms pass the acceptance checks:

- Run **at least three**, ideally five, full repetitions of all four arms. Each arm has **192 requests**, so five repetitions give **3,840 measured requests** total.
- For each repetition, randomly shuffle all four arms with a **recorded deterministic seed**. Preserve the common prompt digest; each arm must see the same logical workload.
- Keep the same two GPUs and host for the entire cohort. If you move pods or install a new Dynamo/vLLM build, start a *new* cohort; don't splice results together.
- Scrape real worker metrics before and after each measured interval. Disjoint warm-up and restart rules are mandatory.
- Save failed/inconclusive runs, but don't silently average them with successful ones.

Example schedule generator:

```bash
python - <<'PY'
import random
for rep in range(1, 6):
    arms = ['R0', 'R1', 'R2', 'R3']
    random.Random(f'agentic-lab-day2-{rep}').shuffle(arms)
    print(f'rep={rep}:', ', '.join(arms))
PY
```

**Do not introduce new budgets during the primary routing comparison.** The point is to compare policy, not simultaneously change KV capacity.

### 8.3 One realism/stress follow-up (optional, highly recommended)

Once the fixed arrival-order experiment is sound, run a **separate cohort** in which session start times and tool pauses vary under a fixed, recorded random seed. Consider a second sensitivity test with `max-inflight=8`, because the Day 1 client's global four-request semaphore may underutilize two GPUs.

Do **not** mix these results with the fixed 0.5-second/4-inflight cohort. Report separate workload regimes. This matters: synchronized deterministic sessions can create artificially strong or weak affinity advantages.

---

## 9. What to measure (and how to avoid fooling yourself)

| Priority | Metric | Measurement | Why it matters |
|---|---|---|---|
| **Primary** | Resumed-turn cached-token fraction | Per request, turn ≥1, `cached_tokens / prompt_tokens` | Did the routed request actually reuse its prefix? |
| **Primary** | Resumed-turn TTFT | Client timestamps, by run and session | Was reuse visible in latency? |
| **Primary** | Total computed prefill tokens | Per-worker Prometheus counter deltas, summed | Did the system avoid recomputation? |
| **Primary** | Workload completion time | First request ready to final response; one value per run | Did the distributed system finish the job sooner? |
| **Primary** | Session-to-worker transitions | Correlated router/worker records | Did affinity work? Did R0 switch unnecessarily? |
| Secondary | Cache-reuse coverage | Fraction of requests with valid cached-token data | Guards against missing-as-zero errors |
| Secondary | Worker balance | Per-worker request count, active time, queue depth | Did stickiness overload one replica? |
| Secondary | TTFT p95 | Per run, on the 176 resumed requests, with repeated runs | Characterizes tail; don't pool turns as independent reps |
| Secondary | SLO-goodput | Valid requests meeting preregistered TTFT SLO / total elapsed | Operationally interpretable result |
| Secondary | Harness wait / user TTFT | Existing client timestamps | Detects client throttling as an artificial limit |
| Diagnostic | Cache hit ratio, KV usage, preemptions, KV-event lag | Per-worker/router metrics | Helps diagnose mechanism; not a substitute for primary metrics |

**Avoid these mistakes:**

- **No two-GPU versus one-GPU “routing speedup.”** Compute capacity changed. Routing effect is R1/R2/R3 versus R0 on the same two GPUs.
- **No pooled-turn p-values.** One session's turns are correlated. Summarize each run first, then compare paired run summaries across repetitions.
- **No ignoring cold turns.** Show them separately, but make resumed turns the main reuse analysis. Turn 0 determines the initial placement in affinity arms.
- **No missing-as-zero metrics.** If final SSE usage or metrics disappear behind Dynamo, the result is incomplete until instrumentation is validated.
- **No inference from a router setting alone.** A flag does not prove that workers received events or that their prompts hashed identically.
- **No selective per-arm debugging.** Repeating only the losing arm on a fresh machine and comparing with old numbers is invalid.
- **No claim that hard affinity means KV protection.** It only affects placement; the selected worker can still evict paused-session blocks.

### Minimum output files

```text
agentic-inference-lab/
├── DAY_1_GUIDE.md
├── DAY_2_GUIDE.md                 # this file
├── workloads/
│   └── multi_turn.py             # add session headers + route-arm metadata
├── scripts/
│   ├── day2_launch.sh            # create: two workers + frontend + cleanup
│   ├── day2_run_matrix.sh        # create: shuffled arms and repetitions
│   └── day2_analyze.py           # create: per-worker + per-rep analysis
├── configs/
│   └── day2_lock.json            # exact model / runtime / GPU / KV settings
├── docs/
│   └── day2_findings.md          # preregistered before main runs
└── results/day2/
    ├── pilot/rep_1/{R0,R1,R2,R3}/
    │   ├── manifest.json
    │   ├── requests.jsonl
    │   ├── workload_manifest.json
    │   ├── frontend.log
    │   ├── worker0.log
    │   ├── worker1.log
    │   ├── placement.jsonl
    │   ├── frontend_metrics_before.prom
    │   ├── frontend_metrics_after.prom
    │   ├── worker0_metrics_before.prom
    │   ├── worker0_metrics_after.prom
    │   ├── worker1_metrics_before.prom
    │   └── worker1_metrics_after.prom
    └── main/rep_{1..5}/{R0,R1,R2,R3}/
```

`placement.jsonl` is a **desired artifact**; it should only be created if placement was truly observed. If you cannot correlate routes, write a visible limitation in the findings, not invented IDs.

---

## 10. Analysis: the four plots and two tables to produce

**Plot 1 — Primary:** per-turn *median cached-token fraction*, split by R0/R1/R2/R3. The unit of variability should be the repetition (show within-run medians plus individual-request faint points, not an error bar pretending 176 turns are independent).

**Plot 2:** resumed TTFT by routing arm. Show a dot for each repetition median plus distribution context for the requests. Cold-turn TTFT separate.

**Plot 3:** per-worker computed prefill tokens (stacked or side-by-side), with total printed above each arm. Include whether workers were evenly utilized.

**Plot 4:** session-to-worker assignment timeline or transition matrix; also show the number of distinct workers a session visited. This is the strongest visual proof that routing behavior actually changed.

**Table A — comparison:** For each repetition and each R1/R2/R3 versus R0, compute **paired** changes in resumed TTFT median, computed prefill tokens, resumed cached fraction, completion time and goodput. Use median paired effects and ranges/intervals across repetitions; with only three to five runs, do not oversell precise confidence.

**Table B — validity:** requests succeeded, worker-count coverage, per-worker KV allocations, cache-metric coverage, model/tooling SHA consistency, process restarts, hash configuration and any anomalous runs.

### Diagnostic interpretations

| Observation | Likely next check |
|---|---|
| R0 low reuse; R2 high reuse and lower prefill | Session placement makes a genuine difference in this workload |
| R1 ≈ R2, both improve | KV-overlap routing can recover much of affinity's benefit here |
| R2 high hits but worse completion | Uneven worker load / queueing may overwhelm cache savings |
| R1 low hits despite correct engine-local reuse | Hash/event/indexer/score configuration may be wrong |
| All arms low hits | Aggregate working set may still exceed per-worker budgets, or routing may split sessions |
| All arms high hits | Chosen pressure regime may be too generous or the workload too easy |
| R2/R3 send all 16 sessions to one worker | Initial placement likely imbalanced; measure rather than dismiss |
| Per-request `cached_tokens` missing | Fix instrumentation/worker metrics before claiming cache effects |
| TTFT decreases but prefill does not | Investigate scheduling/front-end overhead and whether the same work was served |

**Interpretation discipline:** Unlike the Day 1 single-worker cache budget sweep, Day 2 introduces the router and may modify software/runtime versions. If the cache threshold shifts, it is not automatically proof that routing caused it; compare routing arms at constant Day 2 configuration.

---

## 11. Validation / stop-go checkpoints

- [ ] Two identical NVIDIA GPUs available; two **independent** serving workers, no tensor-parallel substitution
- [ ] Runtime and model SHA pinned; release-specific command-line options verified
- [ ] Both workers use 4.5 GiB KV, 8192 context, same BF16 mode, same 64-token blocks, prefix cache ON
- [ ] `PYTHONHASHSEED=0` for relevant workers and exact same tokenizer / prompt rendering
- [ ] Frontend sees exactly two ready workers; KV event path validated
- [ ] R2/R3 session affinity verified by **actual worker placement**, not just a header or flag
- [ ] Session ID stable across 12 turns; distinct per session; prompts match Day 1 digest for equivalent workload
- [ ] All four pilot arms complete **192/192** requests with meaningful cache and route evidence
- [ ] Fresh cache and affinity state between every measured run; pre/post metrics per worker
- [ ] Three to five valid randomized repetitions for R0–R3; no mixing hosts or runtime builds
- [ ] Figures and paired comparison table generated; load imbalance, failed runs and measurement limits reported
- [ ] `docs/day2_findings.md` updated with a mechanism-based conclusion, even if negative or inconclusive

**If the pilot shows real routing/cache differences:** move into repeated comparisons and analysis.  
**If the router integration is wrong:** fix the integration; do not create a new KV policy.  
**If all routes tie under this deterministic workload:** perturb arrival order / pause distributions as a separate, preregistered sensitivity cohort before concluding routing is redundant.

---

## 12. Day 3 decision (do not start today)

Only after Day 2 results identify a specific failure should you pick an implementation contribution:

1. **Stale or incomplete cache events?** Investigate event freshness/indexing and reproducible routing misses.
2. **Hard affinity hurts due to concentration?** Study load-aware or soft affinity selection, after pinning a runtime with support.
3. **Same-worker resumptions still lose KV under pressure?** Now consider bounded engine-local retention or a tier policy.
4. **No material routing failure?** Do not invent one. Share the reproducible workload as a benchmark case, or choose an active upstream bug with an end-to-end GPU reproduction.

The goal of this project is **a real, measured inference-system improvement and ideally a strong upstream contribution**, not simply a positive benchmark chart.

---

## 13. Reference documentation

* [NVIDIA Dynamo v1.5.1 — Local KV-aware routing](https://docs.nvidia.com/dynamo/v1.5.1/cli/kv-aware-routing/overview)
* [NVIDIA Dynamo v1.5.1 — Router configuration and session affinity](https://docs.nvidia.com/dynamo/v1.5.1/knowledge-base/modular-components/router/configuration-and-tuning)
* [NVIDIA Dynamo — Worker/client routing guide](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/router/router-guide) (**dev** docs, do not assume all features exist in pinned build)
* [Dynamo aggregated vLLM two-worker example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/agg_router.sh) (unpinned upstream reference)
* [Dynamo router example benchmarks](https://github.com/ai-dynamo/dynamo/blob/main/benchmarks/router/README.md)
* [Dynamo session-aware prefix indexer proposal](https://github.com/ai-dynamo/dynamo/issues/13279) (proposed engineering direction, not proof of a finished feature)
* [vLLM serving CLI](https://docs.vllm.ai/en/latest/cli/serve/) (confirm flags in your exact deployed build)

---

### Explicit boundaries

This guide specifies a plausible, documentation-grounded deployment and reproducibility protocol. **It has not been run on a two-GPU RunPod instance as part of preparing the guide.** Release-specific frontend/worker flags, metric endpoint discovery, and exact end-to-end compatibility must be validated by the prescribed smoke tests. Day 1 numeric observations above are grounded in the supplied results archive; Day 2 performance outcomes are hypotheses, not measurements.
