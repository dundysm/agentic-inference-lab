# Engineering notes: distributed inference

**Date:** 2026-10-12. **Status:** complete. All times below are UTC.

This is the *process* record: what was planned, what was built, what broke, what was decided and why, and where every artifact lives. For results and interpretation, read [routing-cache-residency-findings.md](routing-cache-residency-findings.md).

---

## 1. Goal

**Question:** on the **same two GPUs** with the **same KV budget per worker** (4.5 GiB, 32,768 tokens each), how much does the request router change prefix reuse, recomputation, latency and load balance for the cache-residency baseline's agent-like workload?

The workload is 16 sessions × 12 turns, about 3K-token prefixes and tool pauses.

**Motivation from the cache-residency baseline:** a single worker at 4.5 GiB showed total prefix-reuse collapse once 16 paused sessions (about 54K tokens) exceeded the cache, with zero preemptions. the distributed routing study asks whether **placing turns on the right worker** recovers that reuse before touching any engine-side retention policy.

**Definition of done** (from the the distributed routing study guide): a two-worker Dynamo + vLLM deployment, a deterministic workload with stable session IDs, four routing arms, per-worker evidence of placement and cache behaviour, at least 3 valid repetitions, and an honest written conclusion. A speedup was not required.

---

## 2. Timeline

| Time (UTC) | Step |
|---|---|
| 12:47 | the distributed routing study guide reviewed against the real Dynamo v1.5.1 source and committed (`ee1a465`); corrections in §3 |
| ~13:50 | 2× A40 sold out; 2× RTX PRO 4500 chosen with approval. The first pod sat in "Initializing" with no logs for about 10 min and was terminated (stuck host) |
| ~14:07 | Second pod `t08nb1xd591jrv` came up (2× RTX PRO 4500, US-KS-2). Lab SSH key added via the Jupyter terminal |
| 14:12–14:20 | Dynamo 1.5.1 + vLLM 0.28.0 installed with uv; capability check of every frontend and worker flag in the pinned build |
| 14:19–14:26 | Cluster bring-up: four failures diagnosed and fixed (§4) |
| 14:26–14:28 | Integration tests A–E passed; launcher and client changes committed (`8c72776`) |
| 14:29 | R3 changed to *soft* affinity to match `docs/distributed-routing.md` (`756cb9e`) |
| 14:30–14:34 | Four-arm pilot (one rep): all gates passed, two predictions wrong (§7) |
| 14:36–14:37 | Pilot data, **main-cohort pre-registration** (`1750685`) and evaluator (`824437c`) committed **before** the main cohort |
| 14:35–14:57 | **Main cohort**: 5 reps × R0–R3, 20 runs, all valid |
| 14:58–15:01 | Response-id capture added (`1321307`); untimed **diagnostic** R1 run with debug router logging |
| 15:01–15:04 | **Exploratory** R1 runs at `--router-prefill-load-scale` 2 and 4 |
| 15:05 | Main cohort, diagnosis and findings committed (`c85b6dc`) |
| ~15:27 | Pod `t08nb1xd591jrv` stopped and terminated at your request |
| 15:38 | Four section 10 figures and Tables A/B added (`0f07fcb`) |
| ~15:45 | Upstream issue drafted; no duplicate found in ai-dynamo/dynamo |
| 16:42 | Follow-up code (sensitivity timing, R1S4/R3H arms, single-worker S45/S9) and the **follow-up pre-registration** committed (`a7ec8fe`), **before** any follow-up pod existed |
| ~16:42 | 2× RTX PRO 4500 sold out; **2× L40S** pod `pa78lsi4duh10l` deployed with approval |
| 16:46 | Follow-up evaluator committed before any data (`87c2352`). The committed setup script ran unmodified on the new pod |
| 16:48–16:52 | Smoke runs (outside the cohorts): S9, R1S4 and sensitivity timing verified |
| 16:52–17:15 | **Sensitivity cohort**: 5 reps × R0, R1, R1S4, R2, 20 runs, all valid |
| 17:15–17:29 | **Supplementary cohort**: 3 reps × R2, R3H, S45, S9, 12 runs, all valid |
| ~17:31 | Follow-up pod stopped and terminated |
| 17:38 | Follow-up results, figures and complete findings committed (`ad4fab1`) |
| ~17:40 | Upstream issue filed from your account at your request: **[ai-dynamo/dynamo#16194](https://github.com/ai-dynamo/dynamo/issues/16194)**; findings link it (`7644734`) |

---

## 3. Guide review: what was corrected before spending

The guide was checked against the pinned Dynamo v1.5.1 source on GitHub before any pod was rented.

| Guide said | Reality in v1.5.1 | Action |
|---|---|---|
| Start etcd/NATS with `docker compose` | RunPod pods have no docker. v1.5.1 supports `--discovery-backend file`, `--request-plane tcp` and `--event-plane zmq` | No etcd, NATS or docker; a fresh file-discovery directory per run |
| Use the pinned runtime's vLLM | Dynamo 1.5.1 pins `vllm==0.28.0` and `nixl[cu13]` (a CUDA 13 build) | Required a **CUDA 13 host** (driver ≥580); the cache-residency baseline's 570 driver could not run it |
| Put results on `/workspace` | the cache-residency baseline lost a cohort to silent write loss on RunPod's network filesystem | Everything on local disk, copied home after each cohort |
| Soft affinity may not exist in stable | `--router-session-affinity-mode {hard,soft}` exists in v1.5.1 | Used |
| Placement must be observed | `nvext.extra_fields=["worker_id"]` returns the serving worker per request (documented) | Used; corroborated by per-worker vLLM counters |
| Session header | `x-dynamo-session-id` confirmed in source (`HEADER_DYNAMO_SESSION_ID`) | Sent on every turn in every arm |
| Stock `agg_router.sh` uses `--enforce-eager` and a 0.6B model | Fine for a demo, not for measurement | Own launcher (`scripts/day2_cluster.sh`) with no `--enforce-eager` |
| Arm R3 = KV + hard affinity | Your `docs/distributed-routing.md` defines R3 as **soft** affinity | Main-cohort R3 = soft. The guide's version was later run as R3H in the supplementary cohort |

---

## 4. Bring-up problems and fixes (main-cohort pod)

| Symptom | Root cause | Fix (in `scripts/day2_cluster.sh`) |
|---|---|---|
| Worker 0: `Address already in use` on 8081 | RunPod's nginx inside the pod owns port 8081 | Worker status ports moved to **9081/9082** |
| Worker 1: free memory 25.6 GiB < 0.92 × 31.4 GiB | **5,691 MiB on GPU 1 belonged to something outside our container** (likely the stuck pod's leftover). vLLM's startup check uses `--gpu-memory-utilization` even when the KV size is explicit | `--gpu-memory-utilization 0.70` on **both** workers. This affects only the startup check; KV stays exactly 32,768 tokens. The asymmetry was recorded as a per-GPU baseline, and utilization was sampled every run |
| `FlashInfer requires GPUs with sm75 or higher` (sm_120 GPU) | FlashInfer JIT used `/usr/local/cuda` 12.8 from the base image, which can't target Blackwell | Tried pointing `CUDA_HOME` at pip's CUDA 13 toolkit, which failed: nvcc 13.4 doesn't match the 13.0 headers. Reverted |
| Same sampler JIT | Only vLLM's **sampler** uses that JIT path | `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch-native sampler). Decoding is greedy at temperature 0; attention, prefill and KV are unaffected |
| SSH command killed itself | `pkill -f` matched its own command line | Kill by process group only |
| Orphaned `VLLM::EngineCore` after killing its parent | EngineCore renames itself and survives a by-name kill | `stop_cluster` kills whole `setsid` process groups, then waits for both GPUs to return to baseline |
| `git push` rejected mid-session | You had merged PRs #1/#2 (rebrand, new docs, guides removed) and renamed the repo to `inference-under-pressure` | Rebased onto your changes; remotes updated |

---

## 5. What was built

### Code (all in the repo)

| File | Purpose |
|---|---|
| `workloads/multi_turn.py` | All new options are opt-in, so the cache-residency baseline behaviour and the prompt digest are unchanged (tested):<br>• `--session-header`: stable `X-Dynamo-Session-ID`<br>• `--request-worker-id`: placement per request<br>• `--route-arm`<br>• `--metrics NAME=URL`: per-endpoint snapshots<br>• response-id capture<br>• `--stagger-s`, `--pause-max`, `--schedule-seed`: seeded sensitivity timing |
| `scripts/day2_cluster.sh` | Single source of all frontend and worker flags; arms R0, R1, R2, R3, R1S4, R3H, S45, S9; process-group lifecycle; GPU-baseline checks |
| `scripts/day2_run_matrix.sh` | Seeded arm order per rep and a fresh cluster per run. Records manifests and the instance-id→GPU map, requires warm-up to reach every worker, samples GPU utilization, and snapshots frontend and per-worker metrics |
| `scripts/day2_analyze.py` | Per-run summary: placement, moves, per-worker reuse and recomputation, latency, balance, utilization |
| `scripts/day2_evaluate.py`, `scripts/day2_followup_evaluate.py` | Pre-registered hypothesis checks, committed before their data |
| `scripts/day2_router_diagnosis.py` | Joins debug router cost lines to requests: router belief vs actual reuse per decision |
| `scripts/day2_plots.py` | Four figures (reuse by turn, resumed TTFT, per-worker prefill, placement grid) and Tables A/B; captions computed from the data; palette validated for colour-vision deficiency |
| `env/runpod_day2_setup.sh` | the distributed routing study environment install. It ran unmodified on the second pod |
| `tests/test_logic.py` | New tests: worker-id parsing, digest independence from routing flags, metric targets, schedule determinism and bounds |

### Documents
- [`routing-primary-2026-10-12.md`](routing-primary-2026-10-12.md): H1–H5, committed before the main cohort.
- [`routing-sensitivity-2026-10-12.md`](routing-sensitivity-2026-10-12.md): S-H1–S-H6 and X-H1–X-H4, committed before the follow-up pod existed.
- [`routing-cache-residency-findings.md`](routing-cache-residency-findings.md): results, mechanism, wrong predictions, limitations, reproduction.
- Status lines updated in `README.md` and `docs/distributed-routing.md` (your text otherwise untouched).

### Data
- **[`results/2026-10-12-runpod-2x-rtxpro4500/`](../results/2026-10-12-runpod-2x-rtxpro4500/):**
  - `day2-pilot` (4 runs) and `day2-main` (20 runs, with `hypotheses.txt`, `summary.csv`, `tables.md` and figures);
  - `day2-diag-r1-debug`;
  - `day2-explore-r1-prefill-scale-{2,4}`.
- **[`results/2026-10-12-runpod-2x-l40s/`](../results/2026-10-12-runpod-2x-l40s/):** `day2-smoke-*`, `day2-sens` (20 runs), `day2-supp` (12 runs), plus setup and runner logs.
- **Each run directory holds:**
  - `requests.jsonl` (per request: timing, cached tokens, worker id, response id);
  - frontend and per-worker metrics before and after;
  - `frontend.log` and `worker*.log`;
  - `launch_argv.txt`, `manifest.json`, `worker_map.json`, `environment.txt`, `gpu_util.csv`;
  - the warm-up subfolder.

---

## 6. Experimental design

| Arm | Frontend | Cohorts |
|---|---|---|
| R0 | `--router-mode round-robin` | main, sensitivity |
| R1 | `--router-mode kv` (default weights) | main, sensitivity |
| R2 | round robin + hard affinity, TTL 300 s | main, sensitivity, supplementary |
| R3 | KV + **soft** affinity, TTL 300 s | main |
| R1S4 | KV + `--router-prefill-load-scale 4` | sensitivity |
| R3H | KV + **hard** affinity (the guide's R3) | supplementary |
| S45 / S9 | one worker, 4.5 / 9 GiB, on GPU 0 | supplementary |

**Held constant everywhere:**
- Model `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f…` (model and tokenizer).
- 64-token blocks, 8192 context, prefix caching, KV events, `PYTHONHASHSEED=0`.
- Prompt digest `c9c7e42687f3…`, identical to the cache-residency baseline's 16-session arms.

**Timing:**
- **Main and supplementary:** synchronized starts, 0.5 s pauses, 4 in flight.
- **Sensitivity:** seeded 0–8 s staggered starts, 0.25–1.75 s pauses, 8 in flight. The schedule is identical across arms and reps.

**Protocol:** a fresh cluster per run, a disjoint warm-up that must reach every worker, seeded arm order per rep, and the repetition as the unit of analysis. Any invalid run is reported, not dropped.

---

## 7. Results in brief

Details, figures and tables are in the [findings](routing-cache-residency-findings.md).

**Pilot (one rep):** all gates passed. Two predictions were wrong:
- Round robin did **not** collapse to near-zero reuse.
- The KV router kept only 2 of 16 sessions in place.

**Main cohort** (2× RTX PRO 4500, 5 reps, 20/20 valid):

| | R0 round robin | R1 KV default | R2 hard | R3 soft |
|---|---:|---:|---:|---:|
| Recomputed prefill vs R0 | 1.00× | 0.88× (0.77–1.24) | 0.26–0.36× | 0.26–0.36× |
| Resumed p95 TTFT | ~290 ms | ~285 ms | ~48 ms | ~49 ms |

- **H1** (affinity helps): supported.
- **H2** (KV router partial but better than round robin): **not supported**; it beat round robin in only 3/5 reps.
- **H3** (misses come from moves): 397/397.
- **H4** (soft ≈ hard): supported.
- **H5** (no balance penalty): supported.

**Diagnostic run:** 41/41 KV-router moves left a worker the router *correctly* believed held about 49 of 50 prefix blocks. There were 0 stale-index errors and 0 eviction-driven moves. Cause: the default cost function's overlap credit only offsets the prefill term, so one extra in-flight sequence's decode blocks outweigh the reuse.

**Exploratory runs:** `--router-prefill-load-scale` 2 → 6 moves; 4 → 0 moves and recompute identical to hard affinity.

**Sensitivity cohort** (2× L40S, 5 reps, 20/20 valid):

| Arm | R0 | R1 | R1S4 | R2 |
|---|---:|---:|---:|---:|
| Recomputed prefill vs R0 | 1.00× | **0.51×** | **0.26×** | **0.21×** |

- **S-H2: not supported.** The default KV router is not round-robin-like under staggered traffic.
- **S-H3: not supported.** Prefill-scale 4 reached 1.30× affinity, not ≤1.2×.
- **S-H1, S-H4, S-H5 (524/533), S-H6:** supported.

**Supplementary cohort** (2× L40S, 3 reps, 12/12 valid): all four hypotheses supported.
- The guide's R3 equals R2.
- One 4.5 GiB worker reproduces the cache-residency baseline's collapse on the the distributed routing study stack (0.000 cached, 10.4× recompute).
- One 9 GiB worker matches two affinity-routed workers exactly, 13% slower in wall time.

**Bottom line:** session affinity was the best placement in every regime. The default KV router gives up all locality on synchronized sessions and about half on staggered ones, by design of its cost weighting. Reweighting recovers most but not all of the gap.

---

## 8. Integrity practices used

- Pre-registration and evaluators committed **before** each cohort's data; thresholds never adjusted afterwards. Failed hypotheses (H2, S-H2, S-H3) are reported as failed.
- Every run validated: 192/192 requests, placement known for every request, prompt digest match, `completed` flag, GPUs back to baseline.
- Code changes for new cohorts verified not to alter existing arms: R0–R3 argv diffed byte-identical, and main-cohort figures regenerated byte-identical.
- Diagnostic and exploratory runs labelled as outside the cohorts.
- Cohorts on different GPU types never compared on absolute latency.
- Secret and IP scans before every push. Tokens, keys and pod IPs stayed out of the repo.
- Numbers in the findings re-derived from raw data before publishing. One wrong range in the issue draft (moves per run) was caught and corrected before filing.

---

## 9. Resources and cost

| Pod | GPUs | Use | Approx. cost |
|---|---|---|---:|
| (stuck, terminated after ~10 min) | 2× RTX PRO 4500 | never started its container | ~$0.25 |
| `t08nb1xd591jrv` | 2× RTX PRO 4500 Blackwell SE, $1.45/hr | bring-up, pilot, main cohort, diagnostic, exploratory | ~$2.00 |
| `pa78lsi4duh10l` | 2× L40S, $2.19/hr | smoke, sensitivity, supplementary | ~$1.80 |
| **the distributed routing study total** | | | **~$4** |

All pods were terminated, and their SSH aliases were removed from `~/.ssh/config`. The lab key `~/.ssh/id_ed25519_runpod_lab` is kept for future pods.

---

## 10. Deviations, limitations and open items

**Deviations:**
- Pip install instead of the NVIDIA container.
- File discovery instead of etcd/NATS.
- Status ports 9081/9082.
- Native sampler.
- `gpu-memory-utilization 0.70`.
- R3 = soft affinity in the main cohort.
- Follow-up cohorts on L40S, because the original GPU pair was sold out.
- No separate `configs/day2_lock.json` or `placement.jsonl`; the same information is in the manifests and request logs.
- The stock `agg_router.sh` smoke test was skipped; the launcher's own checks covered it.

**Limitations:**
- One model and one synthetic prompt set, with two timing patterns.
- Client-throttled load (4 or 8 in flight), light enough that worker queues stayed near empty.
- Single-host pods.
- Client TTFT includes the Dynamo frontend.
- Unexplained foreign 5.7 GiB on GPU 1 of the main-cohort pod.

**Open:**
- Maintainer responses on **ai-dynamo/dynamo#16194**.
- Possible follow-ups, not started: heavier or skewed load where the load term could legitimately win; a second model; an engine-side retention study (the Day 3 decision in the guide).
