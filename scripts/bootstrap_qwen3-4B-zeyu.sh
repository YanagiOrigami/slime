#!/bin/bash

set -ex

cd /root/slime

# Ensure deps
pip install -e . || true
pip install -U huggingface_hub || true

# Download HF model if not already present
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ ! -d "/root/Qwen3-4B" ]; then
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B", local_dir="/root/Qwen3-4B", local_dir_use_symlinks=False)
PY
fi

# Convert HF checkpoint to Megatron torch_dist if not present
if [ ! -d "/root/Qwen3-4B_torch_dist" ] || [ ! -f "/root/Qwen3-4B_torch_dist/latest_checkpointed_iteration.txt" ]; then
  source scripts/models/qwen3-4B.sh
  CUDA_VISIBLE_DEVICES=0 python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist \
    ${MODEL_ARGS[@]}
fi

# Launch training (detached)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export BASE_DIR=/root
export NUM_GPUS=${NUM_GPUS:-8}
export ACTOR_GPUS=${ACTOR_GPUS:-8}
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
export TP_SIZE=${TP_SIZE:-2}
export PROMPT_DATA=${PROMPT_DATA:-/data/lcbp/train.jsonl}

bash scripts/run-qwen3-4B-coding.sh > /root/train_4b.log 2>&1 &

echo "Qwen3-4B training submitted. Logs: /root/train_4b.log"