#!/usr/bin/env bash
# Day 2 environment on a RunPod 2x RTX PRO 4500 pod (driver 580.x, CUDA 13.0 host), runpod/pytorch image.
# These are the steps used on 2026-10-12, consolidated. Run from the repo root inside the pod.
#
# Workarounds this captures (also documented in scripts/day2_cluster.sh):
#   - Dynamo 1.5.1 pins vllm 0.28.0 and nixl[cu13], so the host driver must support CUDA 13.
#   - No etcd, NATS or docker is needed: workers and frontend use --discovery-backend file,
#     --request-plane tcp and --event-plane zmq.
#   - Keep the venv, HF cache, logs and results on local disk, not on /workspace (network FS).
set -euxo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
git clone -q --depth 1 --branch v1.5.1 https://github.com/ai-dynamo/dynamo.git /root/dynamo   # source for reference
uv venv /root/venv --python 3.12
source /root/venv/bin/activate
uv pip install "ai-dynamo[vllm]==1.5.1"
uv pip install -r requirements.txt pip
uv cache clean

export HF_HOME=/root/hf-cache
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554

python -c "import torch, vllm; print('torch', torch.__version__, torch.version.cuda, torch.cuda.device_count()); print('vllm', vllm.__version__)"
