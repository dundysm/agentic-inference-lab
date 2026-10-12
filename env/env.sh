# Source this before running scripts/run_baseline.sh on the pod.
export PATH=/root/.local/bin:$PATH
source /root/venv/bin/activate
export HF_HOME=/root/hf-cache
export HF_HUB_OFFLINE=1
export MODEL=Qwen/Qwen3-4B-Instruct-2507
export MODEL_REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
