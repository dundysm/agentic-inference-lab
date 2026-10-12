#!/usr/bin/env bash
# Routing: run routing arms (R0-R3, follow-up R1S4/R3H/S45/S9), one fresh cluster per run, seeded order per rep.
#   COHORT=pilot REPS=1 bash scripts/routing_run_matrix.sh
# Results go to local disk (RESULTS_ROOT); copy them off the pod after each cohort.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${COHORT:?set COHORT (e.g. pilot, main)}"
: "${REPS:=1}"
: "${REP_START:=1}"
: "${ARMS:=R0 R1 R2 R3}"
: "${RESULTS_ROOT:=/root/results/day2}"
: "${SESSIONS:=16}"
: "${TURNS:=12}"
: "${LINES:=80}"
: "${PAUSE:=0.5}"
: "${MAX_INFLIGHT:=4}"
: "${STAGGER_S:=0}"         # follow-up sensitivity timing; 0/0 = fixed schedule
: "${PAUSE_MAX:=0}"
: "${SCHEDULE_SEED:=day2-sensitivity}"
: "${MAX_OUTPUT_TOKENS:=32}"
: "${HOT_SESSION_COUNT:=0}"
: "${HOT_PAUSE_SCALE:=1}"
: "${COLD_PAUSE_SCALE:=1}"
: "${MODEL:?}" "${MODEL_REVISION:?}"
source scripts/routing_cluster.sh

schedule() {
  python - "$1" $ARMS <<'PY'
import random, sys
rep, arms = sys.argv[1], sys.argv[2:]
random.Random(f"agentic-lab-day2-{rep}").shuffle(arms)
print(" ".join(arms))
PY
}
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for ((rep=REP_START; rep<=REPS; rep++)); do echo "rep=$rep $(schedule "$rep")"; done; exit 0
fi

curl -fs --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { echo "port 8000 already serving; stop it first" >&2; exit 1; }
record_gpu_baseline
echo "gpu baseline MiB: ${GPU_BASELINE[*]}"
trap 'stop_cluster || true; kill "${SAMPLER_PID:-}" 2>/dev/null || true' EXIT

for ((rep=REP_START; rep<=REPS; rep++)); do
  for arm in $(schedule "$rep"); do
    dir="$RESULTS_ROOT/$COHORT/rep_$rep/$arm"
    [[ -e "$dir/requests.jsonl" ]] && { echo "existing data in $dir; refusing to overwrite" >&2; exit 1; }
    mkdir -p "$dir"
    { date -u; nvidia-smi; nvidia-smi topo -m; python --version; python -m pip freeze
      echo "lab_git=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
      echo "dynamo_git=$(git -C /root/dynamo rev-parse HEAD 2>/dev/null)"; } > "$dir/environment.txt" 2>&1
    python - "$dir/manifest.json" <<PY
import json, sys
json.dump({"cohort": "$COHORT", "rep": $rep, "arm": "$arm", "model": "$MODEL",
  "model_revision": "$MODEL_REVISION", "kv_cache_memory_bytes_per_worker": int("$KV_BYTES"),
  "block_size": int("$BLOCK_SIZE"), "max_model_len": int("$MAX_MODEL_LEN"),
  "gpu_memory_utilization": float("$GPU_MEM_UTIL"), "affinity_ttl_s": int("$AFFINITY_TTL"),
  "sessions": $SESSIONS, "turns": $TURNS, "lines": $LINES, "pause_s": $PAUSE,
  "max_inflight": $MAX_INFLIGHT, "max_output_tokens": int("$MAX_OUTPUT_TOKENS"),
  "hot_session_count": int("$HOT_SESSION_COUNT"), "hot_pause_scale": float("$HOT_PAUSE_SCALE"),
  "cold_pause_scale": float("$COLD_PAUSE_SCALE"), "stagger_s": float("$STAGGER_S"), "pause_max_s": float("$PAUSE_MAX"),
  "schedule_seed": "$SCHEDULE_SEED", "workers": $(arm_workers "$arm"), "kv_cache_memory_bytes_actual": $(arm_kv_bytes "$arm"),
  "gpu_baseline_mib": [int(x) for x in "${GPU_BASELINE[*]}".split()],
  "vllm_use_flashinfer_sampler": "$VLLM_USE_FLASHINFER_SAMPLER", "pythonhashseed": "$PYTHONHASHSEED",
  "completed": False}, open(sys.argv[1], "w"), indent=2)
PY
    echo "LAUNCH $COHORT rep=$rep arm=$arm"
    start_cluster "$arm" "$dir"
    wait_cluster_ready "$dir"
    python - "$dir" "$CLUSTER_WORKERS" <<'PY'
import json, re, sys, pathlib
d = pathlib.Path(sys.argv[1]); m = {}
n = int(sys.argv[2])
for i in range(n):
    ids = set(re.findall(r"Initializing KvEventPublisher for worker (\d+)", (d / f"worker{i}.log").read_text()))
    if len(ids) != 1: raise SystemExit(f"worker{i}: expected one instance id, found {ids}")
    m[ids.pop()] = i
json.dump(m, open(d / "worker_map.json", "w"), indent=2); print("worker_map", m)
PY
    # Disjoint warm-up that must reach both workers (verified by reported worker ids).
    python workloads/multi_turn.py --model "$MODEL" --scenario warmup --sessions 4 --turns 2 --lines 12 \
      --pause 0 --max-inflight 2 --session-offset 9000 --system-tag '[warmup-not-measured]' \
      --request-worker-id --out-dir "$dir/warmup" > "$dir/warmup.log"
    python - "$dir" "$CLUSTER_WORKERS" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); m = json.load(open(d / "worker_map.json"))
seen = {m.get(str(json.loads(l).get("worker_id"))) for l in open(d / "warmup/requests.jsonl")}
if seen != set(range(int(sys.argv[2]))): raise SystemExit(f"warm-up reached workers {seen}, not all")
print("warmup reached all workers:", sorted(seen))
PY
    WORKER_METRICS=()
    for ((i = 0; i < CLUSTER_WORKERS; i++)); do WORKER_METRICS+=(--metrics "worker$i=http://127.0.0.1:$((SYSTEM_PORT_BASE + i))/metrics"); done
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -lms 500 > "$dir/gpu_util.csv" &
    SAMPLER_PID=$!
    python workloads/multi_turn.py --model "$MODEL" --scenario "routing-$arm" --route-arm "$arm" \
      --sessions "$SESSIONS" --turns "$TURNS" --lines "$LINES" --pause "$PAUSE" --max-inflight "$MAX_INFLIGHT" \
      --session-header --request-worker-id \
      --stagger-s "$STAGGER_S" --pause-max "$PAUSE_MAX" --schedule-seed "$SCHEDULE_SEED" \
      --max-output-tokens "$MAX_OUTPUT_TOKENS" --hot-session-count "$HOT_SESSION_COUNT" \
      --hot-pause-scale "$HOT_PAUSE_SCALE" --cold-pause-scale "$COLD_PAUSE_SCALE" \
      --metrics frontend=http://127.0.0.1:8000/metrics "${WORKER_METRICS[@]}" \
      --out-dir "$dir" > "$dir/workload.log"
    kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""
    stop_cluster
    python - "$dir/manifest.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d["completed"] = True; json.dump(d, open(p, "w"), indent=2)
PY
    echo "COMPLETED $COHORT rep=$rep arm=$arm"
  done
done
echo "COHORT_DONE $COHORT"
