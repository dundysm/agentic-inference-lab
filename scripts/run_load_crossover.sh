#!/usr/bin/env bash
# Deterministic locality-versus-load crossover matrix. GPU execution required unless DRY_RUN=1.
# Study design: docs/locality-vs-load.md. Historical experiment data are not modified.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${STUDY_PHASE:=pilot}"                   # pilot or main; NEVER pool the cohorts
case "$STUDY_PHASE" in pilot|main) ;; *) echo "STUDY_PHASE must be pilot or main" >&2; exit 2 ;; esac
: "${REPS:=$([[ $STUDY_PHASE == pilot ]] && echo 1 || echo 5)}"
: "${RESULTS_ROOT:=/root/results/inference-under-pressure}"
: "${ARMS:=R0 R1 R1S4 R2 R3}"
: "${PROFILES:=reference saturation hot-skew}"
: "${SESSIONS:=16}"
: "${TURNS:=12}"
: "${LINES:=80}"
if [[ "$STUDY_PHASE" == main && "$REPS" -ne 5 ]]; then
  echo "Preregistered main cohort requires REPS=5. Use pilot for exploratory runs." >&2; exit 2
fi
for arm in $ARMS; do
  case "$arm" in R0|R1|R1S4|R2|R3) ;; *) echo "Invalid routing arm: $arm" >&2; exit 2 ;; esac
done
for profile in $PROFILES; do
  case "$profile" in reference|saturation|hot-skew) ;; *) echo "Invalid profile: $profile" >&2; exit 2 ;; esac
done

# Each repetition shuffles the full profile x policy matrix, avoiding whole-profile time drift.
matrix_for_rep() {
  python - "$1" $PROFILES -- $ARMS <<'PY'
import random, sys
rep = int(sys.argv[1])
sep = sys.argv.index("--")
profiles, arms = sys.argv[2:sep], sys.argv[sep+1:]
cells = [(p, a) for p in profiles for a in arms]
random.Random(f"locality-load-crossover-v1-rep-{rep}").shuffle(cells)
for p, a in cells:
    print(f"{p}|{a}")
PY
}

profile_vars() {
  STAGGER_S=8 PAUSE=0.25 PAUSE_MAX=1.75
  MAX_INFLIGHT=4 MAX_OUTPUT_TOKENS=32
  HOT_SESSION_COUNT=0 HOT_PAUSE_SCALE=1 COLD_PAUSE_SCALE=1
  case "$1" in
    reference) ;;
    saturation) MAX_INFLIGHT=16 MAX_OUTPUT_TOKENS=256 ;;
    hot-skew)
      MAX_INFLIGHT=16 MAX_OUTPUT_TOKENS=256
      HOT_SESSION_COUNT=8 HOT_PAUSE_SCALE=0.1 COLD_PAUSE_SCALE=3
      ;;
  esac
}
for ((rep=1; rep<=REPS; rep++)); do
  while IFS='|' read -r profile arm; do
    profile_vars "$profile"
    seed="crossover-${profile}-rep-${rep}"
    printf 'rep=%d profile=%s arm=%s inflight=%s max_output=%s hot=%s seed=%s\n' \
      "$rep" "$profile" "$arm" "$MAX_INFLIGHT" "$MAX_OUTPUT_TOKENS" "$HOT_SESSION_COUNT" "$seed"
    if [[ "${DRY_RUN:-0}" == 1 ]]; then continue; fi
    # The underlying launcher starts a fresh cluster, records worker placement,
    # captures metrics and refuses to overwrite historical run files.
    COHORT="locality-load/${STUDY_PHASE}/${profile}" \
    RESULTS_ROOT="$RESULTS_ROOT" REP_START="$rep" REPS="$rep" ARMS="$arm" \
    SESSIONS="$SESSIONS" TURNS="$TURNS" LINES="$LINES" \
    MAX_INFLIGHT="$MAX_INFLIGHT" MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" \
    STAGGER_S="$STAGGER_S" PAUSE="$PAUSE" PAUSE_MAX="$PAUSE_MAX" \
    HOT_SESSION_COUNT="$HOT_SESSION_COUNT" HOT_PAUSE_SCALE="$HOT_PAUSE_SCALE" \
    COLD_PAUSE_SCALE="$COLD_PAUSE_SCALE" SCHEDULE_SEED="$seed" \
      bash scripts/routing_run_matrix.sh
  done < <(matrix_for_rep "$rep")
done
echo "CROSSOVER_${STUDY_PHASE}_DONE"
