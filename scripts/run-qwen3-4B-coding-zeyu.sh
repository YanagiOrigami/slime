#!/bin/bash

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

# Base dir for models/data; override with BASE_DIR=/home/ubuntu when not using root
BASE_DIR=${BASE_DIR:-"/root"}
# GPU layout (override via env): NUM_GPUS, ACTOR_GPUS, ROLLOUT_GPUS, TP_SIZE
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-8}
TP_SIZE=${TP_SIZE:-4}

# Judge service
export JUDGE_HOST=${JUDGE_HOST:-"38.80.122.117"}
export JUDGE_PORT=${JUDGE_PORT:-"8081"}

# Optional dataset override: PROMPT_DATA=/path/to/your.jsonl
PROMPT_DATA_PATH=${PROMPT_DATA:-"/data/lcbp/train_large_ac.jsonl"}

CKPT_ARGS=(
  --hf-checkpoint ${BASE_DIR}/Qwen3-4B
  --ref-load ${BASE_DIR}/Qwen3-4B_torch_dist
  --load ${BASE_DIR}/Qwen3-4B_slime/
  --save ${BASE_DIR}/Qwen3-4B_slime/
  --save-interval 10
)

ROLLOUT_ARGS=(
  --prompt-data ${PROMPT_DATA_PATH}
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type remote_code_judge

  --num-rollout 200
  --rollout-batch-size 2
  --n-samples-per-prompt 4
  --rollout-max-response-len 8000
  --rollout-max-prompt-len 8000
  --rollout-temperature 0.5

  --global-batch-size 4
)

PERF_ARGS=(
  --tensor-model-parallel-size ${TP_SIZE}
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 3072
  --log-probs-max-tokens-per-gpu 3072
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${TP_SIZE}
  --sglang-max-prefill-tokens 12000
  --sglang-max-total-tokens 12000
  --sglang-router-request-timeout-secs 7200
  --sglang-mem-fraction-static 0.6
  --sglang-num-reserved-decode-tokens 256
  --sglang-disable-custom-all-reduce
  --sglang-cuda-graph-max-bs 16
  --sglang-chunked-prefill-size 4096
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
# Clean any stale Ray sessions and ensure fresh start
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
rm -rf /tmp/ray || true
rm -rf /root/ray || true
unset RAY_ADDRESS || true
sleep 2
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "'"${BASE_DIR}"'/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "JUDGE_HOST": "'"${JUDGE_HOST}"'",
    "JUDGE_PORT": "'"${JUDGE_PORT}"'"
  }
}'

WANDB_ARGS=(
   --use-wandb
   --wandb-project CPRL
   --wandb-group qwen3-4B-test
   --wandb-key ${WANDB_KEY}
)

ray job submit --address="http://127.0.0.1:8265" \
  --working-dir "${BASE_DIR}/slime" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 ${BASE_DIR}/slime/train.py \
  --rerun-mode disabled \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${ACTOR_GPUS} \
  --colocate \
  --rollout-num-gpus ${ROLLOUT_GPUS} \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${PERF_ARGS[@]} \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  ${SGLANG_ARGS[@]} \
  ${WANDB_ARGS[@]}


