set -euo pipefail
source /root/env.sh
cd /workspace/agentic-inference-lab
echo "MAIN_START $(date -u +%FT%TZ)"
RESULTS_DIR=/root/results/main ARMS=ABCD REPS=5 PREFETCH_MODEL=0 \
  KV_BUDGETS="4831838208 10737418240" MODEL_REVISION=$MODEL_REVISION bash scripts/run_baseline.sh
python scripts/analyze.py --results /root/results/main --slo-ms 3000
echo "MAIN_DONE $(date -u +%FT%TZ)"
