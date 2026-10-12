#!/usr/bin/env bash
set -euo pipefail
out=${1:?usage: scrape_metrics.sh OUT_DIR}
mkdir -p "$out"
while true; do
  ts=$(date -u +%Y%m%dT%H%M%S.%NZ)
  curl --max-time 10 -fsS http://127.0.0.1:8000/metrics > "$out/$ts.prom" || true
  sleep 1
done
