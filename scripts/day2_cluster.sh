#!/usr/bin/env bash
# Day 2: one Dynamo frontend + two dynamo.vllm workers (one per GPU). Source this file.
# Single source of truth for worker/frontend flags; workers are identical in every routing arm.
#
#   start_cluster ARM RUN_DIR   # ARM in R0 R1 R2 R3 R1S4 R3H S45 S9
#   wait_cluster_ready RUN_DIR  # frontend serves the model and every worker system port is healthy
#   stop_cluster                # kill our process groups only, wait for GPUs to return to baseline
#
# Requires: MODEL, MODEL_REVISION, HF_HOME. Optional: KV_BYTES, BLOCK_SIZE, MAX_MODEL_LEN, AFFINITY_TTL.

: "${KV_BYTES:=4831838208}"      # 4.5 GiB per worker (9 GiB total)
: "${BLOCK_SIZE:=64}"
: "${MAX_MODEL_LEN:=8192}"
: "${AFFINITY_TTL:=300}"
: "${STARTUP_WAIT_S:=900}"
: "${GPU_IDLE_TOLERANCE_MIB:=350}"
: "${GPU_MEM_UTIL:=0.70}"      # vLLM's startup free-memory check only; KV size is fixed by KV_BYTES
: "${SYSTEM_PORT_BASE:=9081}"  # worker status/metrics ports; 8081 is taken by RunPod's nginx
export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
# vLLM's FlashInfer sampler JIT-compiles a kernel at startup. On this pod it cannot: the base
# image's /usr/local/cuda is 12.8 (too old for sm_120), and pip's CUDA 13 nvcc (13.4) does not
# match its 13.0 headers. Use vLLM's PyTorch-native sampler instead; with temperature=0 decoding
# is greedy either way, and attention, prefill and KV caching are unaffected.
export VLLM_USE_FLASHINFER_SAMPLER=0

CLUSTER_PIDS=()

frontend_args() {
  local arm=$1
  local args=(--http-port 8000 --discovery-backend file --request-plane tcp --event-plane zmq
              --kv-cache-block-size "$BLOCK_SIZE")
  case "$arm" in
    R0) args+=(--router-mode round-robin) ;;
    R1) args+=(--router-mode kv) ;;
    R2) args+=(--router-mode round-robin --router-session-affinity-ttl-secs "$AFFINITY_TTL"
               --router-session-affinity-mode hard) ;;
    # R3 per docs/distributed-routing.md: soft (policy-visible) affinity, so the KV/load scorer
    # can still move a session off a saturated worker. Soft needs a scoring router, hence kv.
    R3) args+=(--router-mode kv --router-session-affinity-ttl-secs "$AFFINITY_TTL"
               --router-session-affinity-mode soft) ;;
    # Follow-up arms (docs/preregistration-day2-followup-2026-10-12.md).
    R1S4) args+=(--router-mode kv --router-prefill-load-scale 4) ;;          # KV router, prefill weighted 4x
    R3H) args+=(--router-mode kv --router-session-affinity-ttl-secs "$AFFINITY_TTL"
                --router-session-affinity-mode hard) ;;                         # the Day 2 guide's original R3
    S45|S9) args+=(--router-mode round-robin) ;;                                # single worker: nothing to route
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  printf '%s\n' "${args[@]}"
}

# Single-worker controls run one worker on GPU 0; S9 gives it the two-worker arms' total KV budget.
arm_workers() { case "$1" in S45|S9) echo 1 ;; *) echo 2 ;; esac; }
arm_kv_bytes() { case "$1" in S9) echo $((2 * KV_BYTES)) ;; *) echo "$KV_BYTES" ;; esac; }

worker_args() {
  local idx=$1 kv=${2:-$KV_BYTES}
  printf '%s\n' --model "$MODEL" --revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION" \
    --block-size "$BLOCK_SIZE" --max-model-len "$MAX_MODEL_LEN" --kv-cache-memory-bytes "$kv" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --enable-prefix-caching --discovery-backend file --request-plane tcp --event-plane zmq \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((20080 + idx))\",\"enable_kv_cache_events\":true}"
}

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r'; }

record_gpu_baseline() { mapfile -t GPU_BASELINE < <(gpu_used_mib); }

start_cluster() {
  local arm=$1 dir=$2
  mkdir -p "$dir"
  export DYN_FILE_KV="$dir/discovery"      # fresh discovery store per run: no stale registrations
  rm -rf "$DYN_FILE_KV"; mkdir -p "$DYN_FILE_KV"
  mapfile -t FARGS < <(frontend_args "$arm") || return 1
  CLUSTER_WORKERS=$(arm_workers "$arm")
  local kv; kv=$(arm_kv_bytes "$arm")
  {
    echo "frontend: python -m dynamo.frontend ${FARGS[*]}"
    for ((i = 0; i < CLUSTER_WORKERS; i++)); do mapfile -t W < <(worker_args "$i" "$kv")
      echo "worker$i: DYN_SYSTEM_PORT=$((SYSTEM_PORT_BASE + i)) CUDA_VISIBLE_DEVICES=$i python -m dynamo.vllm ${W[*]}"; done
  } > "$dir/launch_argv.txt"
  for ((i = 0; i < CLUSTER_WORKERS; i++)); do
    mapfile -t W < <(worker_args "$i" "$kv")
    DYN_SYSTEM_PORT=$((SYSTEM_PORT_BASE + i)) VLLM_NIXL_SIDE_CHANNEL_PORT=$((20096 + i)) CUDA_VISIBLE_DEVICES=$i \
      setsid python -m dynamo.vllm "${W[@]}" > "$dir/worker$i.log" 2>&1 &
    CLUSTER_PIDS+=($!)
  done
  setsid python -m dynamo.frontend "${FARGS[@]}" > "$dir/frontend.log" 2>&1 &
  CLUSTER_PIDS+=($!)
}

wait_cluster_ready() {
  local dir=$1 deadline=$((SECONDS + STARTUP_WAIT_S))
  while (( SECONDS < deadline )); do
    for pid in "${CLUSTER_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || { echo "cluster process $pid exited; see $dir/*.log" >&2; return 1; }
    done
    local healthy=1
    for ((i = 0; i < ${CLUSTER_WORKERS:-2}; i++)); do
      curl -fsS --max-time 2 "http://127.0.0.1:$((SYSTEM_PORT_BASE + i))/health" >/dev/null 2>&1 || healthy=0
    done
    if (( healthy )) && curl -fsS --max-time 2 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "$MODEL"; then
      return 0
    fi
    sleep 3
  done
  echo "cluster not ready after ${STARTUP_WAIT_S}s" >&2; return 1
}

stop_cluster() {
  for pid in "${CLUSTER_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for _ in {1..30}; do
    local alive=0
    for pid in "${CLUSTER_PIDS[@]}"; do kill -0 -- "-$pid" 2>/dev/null && alive=1; done
    (( alive )) || break; sleep 1
  done
  for pid in "${CLUSTER_PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
  CLUSTER_PIDS=()
  [[ ${#GPU_BASELINE[@]} -gt 0 ]] || return 0
  for _ in {1..90}; do
    mapfile -t now < <(gpu_used_mib); local ok=1
    for i in "${!GPU_BASELINE[@]}"; do (( now[i] <= GPU_BASELINE[i] + GPU_IDLE_TOLERANCE_MIB )) || ok=0; done
    (( ok )) && ! curl -fs --max-time 1 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "GPUs not back to baseline (${GPU_BASELINE[*]} MiB): now $(gpu_used_mib | tr '\n' ' ')" >&2; return 1
}
