#!/usr/bin/env bash
# Controlled two-factor runs. No performance claim without real GPU validation.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${REPS:=1}"
: "${ARMS:=ABCD}"
: "${KV_BUDGETS:=3221225472 6442450944}" # 3 / 6 GiB; reselect after pilot
: "${MAX_MODEL_LEN:=8192}"
: "${LINES:=80}"
: "${TURNS:=12}"
: "${PAUSE:=0.5}"
: "${RESULTS_DIR:=results}"
: "${SAMPLE_METRICS:=1}"
: "${PREFETCH_MODEL:=1}"
: "${GPU_IDLE_TOLERANCE_MIB:=350}"
: "${STARTUP_WAIT_S:=1200}"

# Safe deterministic schedule inspection on laptops with no CUDA installation.
build_order() {
  python - "$1" "$ARMS" $KV_BUDGETS <<'PY'
import random, sys
rep, arms, *budgets = sys.argv[1:]
if not budgets or len(set(budgets)) != len(budgets):
    raise SystemExit("KV_BUDGETS must contain distinct positive byte integers")
if len(set(arms)) != len(arms) or not set(arms) <= set("ABCD"):
    raise SystemExit("ARMS must be unique letters drawn from ABCD")
for b in budgets:
    if not b.isdecimal() or int(b) <= 0:
        raise SystemExit("KV_BUDGETS must be positive integer bytes")
pairs = [f"{budget}:{arm}" for budget in budgets for arm in arms]
random.Random(f"agentic-inference-day1-{rep}").shuffle(pairs)
print(" ".join(pairs))
PY
}
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for ((rep=1; rep<=REPS; rep++)); do
    printf 'rep=%s %s\n' "$rep" "$(build_order "$rep")"
  done
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi missing: run this on an NVIDIA GPU pod.' >&2; exit 1; }
command -v setsid >/dev/null || { echo 'setsid required to manage vLLM process group' >&2; exit 1; }
command -v vllm >/dev/null || { echo 'vllm not installed in this environment' >&2; exit 1; }
command -v hf >/dev/null || { echo 'hf CLI missing; install huggingface_hub' >&2; exit 1; }
mkdir -p "$RESULTS_DIR"

# Resolve moving main exactly ONCE and freeze both weights and tokenizer to this SHA.
if [[ -z "${MODEL_REVISION:-}" ]]; then
  MODEL_REVISION=$(python - "$MODEL" <<'PY'
import sys
from huggingface_hub import HfApi
print(HfApi().model_info(sys.argv[1]).sha)
PY
)
fi
if [[ ! "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "MODEL_REVISION must be a 40-character HF git commit SHA (got '$MODEL_REVISION')" >&2
  exit 1
fi
printf 'model=%s\nrevision=%s\n' "$MODEL" "$MODEL_REVISION" | tee "$RESULTS_DIR/model-lock.txt"
if [[ "$PREFETCH_MODEL" == 1 ]]; then
  hf download "$MODEL" --revision "$MODEL_REVISION" >/dev/null
fi

# Single-GPU pod assumption; do not concurrently run unrelated GPU jobs.
gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r'
}
BASELINE_GPU_MIB=$(gpu_used_mib)
[[ "$BASELINE_GPU_MIB" =~ ^[0-9]+$ ]] || { echo 'Cannot read GPU memory usage' >&2; exit 1; }
if curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo 'An inference server already owns port 8000. Stop it manually.' >&2; exit 1
fi

server_pid=""
sampler_pid=""
stop_processes() {
  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sampler_pid=""
  fi
  if [[ -n "$server_pid" ]]; then
    # Kill our own process group only; never kill unrelated vLLM/RunPod processes.
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 -- "-$server_pid" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 -- "-$server_pid" 2>/dev/null; then
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
wait_gpu_idle() {
  local used
  for _ in {1..90}; do
    used=$(gpu_used_mib)
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= BASELINE_GPU_MIB + GPU_IDLE_TOLERANCE_MIB )); then
      if ! curl --max-time 1 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then return 0; fi
    fi
    sleep 2
  done
  echo "GPU not idle: initial ${BASELINE_GPU_MIB}MiB, current $(gpu_used_mib)MiB. Inspect orphaned EngineCore / other processes; refusing to continue." >&2
  return 1
}
trap 'stop_processes' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((rep=1; rep<=REPS; rep++)); do
  for pair in $(build_order "$rep"); do
    budget=${pair%:*}; arm=${pair#*:}
    stop_processes
    wait_gpu_idle
    case "$arm" in
      A) cache_enabled=false; sessions=4; inflight=2;;
      B) cache_enabled=true; sessions=4; inflight=2;;
      C) cache_enabled=true; sessions=16; inflight=4;;
      D) cache_enabled=false; sessions=16; inflight=4;;
    esac
    if [[ "$cache_enabled" == true ]]; then cache_flag=--enable-prefix-caching; else cache_flag=--no-enable-prefix-caching; fi
    dir="$RESULTS_DIR/$budget/rep_$rep/$arm"
    if [[ -e "$dir/requests.jsonl" ]]; then echo "Existing data in $dir; use new RESULTS_DIR (don't overwrite)." >&2; exit 1; fi
    mkdir -p "$dir"
    { date -u; nvidia-smi; vllm --version; python --version; python -m pip freeze; lscpu; } > "$dir/environment.txt" 2>&1
    # Persist configuration BEFORE server/workload; survives failed runs.
    python - "$dir/server_config.json" "$MODEL" "$MODEL_REVISION" "$budget" "$cache_enabled" "$MAX_MODEL_LEN" "$sessions" "$inflight" "$TURNS" "$LINES" "$PAUSE" "$rep" "$BASELINE_GPU_MIB" <<'PY'
import json, sys
(path, model, revision, budget, cache, max_len, sessions, inflight, turns,
 lines, pause, rep, baseline) = sys.argv[1:]
payload = {"model": model, "model_revision": revision, "tokenizer_revision": revision,
 "kv_cache_memory_bytes": int(budget), "prefix_cache_enabled": cache == "true",
 "max_model_len": int(max_len), "sessions": int(sessions), "max_inflight": int(inflight),
 "turns": int(turns), "lines": int(lines), "pause_seconds": float(pause),
 "rep": int(rep), "baseline_gpu_mib": int(baseline), "run_validated_on_gpu": False}
with open(path, "w") as f: json.dump(payload, f, indent=2)
PY
    echo "LAUNCH $budget rep=$rep arm=$arm dir=$dir"
    setsid vllm serve "$MODEL" --host 127.0.0.1 --port 8000 \
      --revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION" \
      --max-model-len "$MAX_MODEL_LEN" --kv-cache-memory-bytes "$budget" \
      --enable-prompt-tokens-details "$cache_flag" > "$dir/server.log" 2>&1 &
    server_pid=$!
    ready=0
    deadline=$((SECONDS + STARTUP_WAIT_S))
    while (( SECONDS < deadline )); do
      if curl --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ready=1; break; fi
      if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
      sleep 3
    done
    if (( ! ready )); then echo "vLLM failed; inspect $dir/server.log" >&2; exit 1; fi
    # Record informative startup capacity lines (may be absent/reformatted in other releases).
    grep -Eim 5 'GPU KV cache size|KV cache.*(blocks|tokens|size)|GPU blocks' "$dir/server.log" > "$dir/kv_allocation.log" || true
    # Different system prompt AND document namespace ensure warm-up is disjoint.
    python workloads/multi_turn.py --model "$MODEL" --scenario warmup \
      --sessions 1 --turns 2 --lines 12 --pause 0 --max-inflight 1 \
      --session-offset 9000 --system-tag '[warmup-not-measured]' --out-dir "$dir/warmup"
    if [[ "$SAMPLE_METRICS" == 1 ]]; then
      bash scripts/scrape_metrics.sh "$dir/metric_series" & sampler_pid=$!
    fi
    python workloads/multi_turn.py --model "$MODEL" --scenario "$arm" \
      --sessions "$sessions" --turns "$TURNS" --lines "$LINES" --pause "$PAUSE" \
      --max-inflight "$inflight" --out-dir "$dir"
    if [[ -n "$sampler_pid" ]]; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; sampler_pid=""; fi
    python - "$dir/server_config.json" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p));d['run_completed_without_exception']=True
with open(p,'w') as f: json.dump(d,f,indent=2)
PY
    echo "COMPLETED $budget rep=$rep arm=$arm"
  done
done
