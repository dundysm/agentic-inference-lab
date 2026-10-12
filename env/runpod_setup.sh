#!/usr/bin/env bash
# Environment used for the 2026-10-12 RunPod A40 runs (driver 570.211.01, CUDA 12.8 max).
#
# The run applied these steps interactively. This file consolidates them, but it has not
# been executed end-to-end as a single script.
#
# Workarounds this captures:
#   1. The default `vllm==0.31.0` wheel needs CUDA 13 (torch 2.13.0 cu13), which a 12.8 driver
#      cannot run. Use the official cu129 build from wheels.vllm.ai.
#   2. `uv --index-strategy unsafe-best-match` pulls a cu13 torchcodec from PyPI. vLLM imports
#      it at startup and fails with libnvrtc.so.13. The workload is text-only, so use the CPU build.
#   3. The runner calls `python -m pip freeze`, and uv venvs have no pip, so install it.
#   4. RunPod /workspace is a network filesystem (MFS). Keep the venv, HF cache, logs and
#      results on local disk. Lost log writes on /workspace silently aborted one cohort.
#   5. The 30 GB container disk fills up unless the uv cache is cleaned.
set -euxo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH

uv venv /root/venv --python 3.12
source /root/venv/bin/activate
uv pip install "vllm==0.31.0+cu129" \
  --extra-index-url https://wheels.vllm.ai/0.31.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
uv pip install --reinstall --no-deps "torchcodec==0.17.0+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install -r "$(dirname "$0")/../requirements.txt" pip
uv cache clean

export HF_HOME=/root/hf-cache
hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554

python -c "import torch, vllm; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('vllm', vllm.__version__)"
