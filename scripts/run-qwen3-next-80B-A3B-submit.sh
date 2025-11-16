#!/bin/bash

# lightweight submit-only launcher (no pkill, no ray start)
set -x

# Judge service (override if needed)
export JUDGE_HOST=${JUDGE_HOST:-38.80.122.117}
export JUDGE_PORT=${JUDGE_PORT:-8081}

# Interface and cluster addr (must be set in the environment before running)
export MLP_SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-ens6}
export MLP_WORKER_0_HOST=${MLP_WORKER_0_HOST:?Set head IP, e.g. 38.80.122.133}

# prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-next-80B-A3B.sh"

# Example checkpoint and save locations; adjust to your paths
CKPT_ARGS=(
   --hf-checkpoint /workspace/Qwen3-Next-80B-A3B-Thinking
   --ref-load /workspace/Qwen3-Next-80B-A3B-Thinking_torch_dist
   --load /workspace/Qwen3-Next-80B-A3B-Thinking_slime/test1
   --save /workspace/Qwen3-Next-80B-A3B-Thinking_slime/test1
   --save-interval 5
)

ROLLOUT_ARGS=(
   --prompt-data /workspace/lcbp/train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type remote_code_judge
   --num-rollout 10
   --rollout-batch-size 1
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   #--global-batch-size 64
   --num-steps-per-rollout 1
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data /workspace/lcbp/train_small_unsolvable.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator gspo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project CPRL
   --wandb-group qwen3-next-80B-A3B
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 16
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)

   --sglang-enable-dp-attention
   --sglang-dp-size 4
   --sglang-ep-size 16
   --sglang-enable-dp-lm-head
   --sglang-moe-a2a-backend deepep
   --sglang-deepep-mode auto
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# cluster env for job
export MASTER_ADDR=${MLP_WORKER_0_HOST}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "GLOO_SOCKET_IFNAME": "'"${MLP_SOCKET_IFNAME}"'",
        "TP_SOCKET_IFNAME": "'"${MLP_SOCKET_IFNAME}"'",
        "MASTER_ADDR": "'"${MLP_WORKER_0_HOST}"'",
        "PYTHONPATH": "/root/Megatron-LM/",
        "NCCL_CUMEM_ENABLE": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
        "NCCL_IB_TC": "160",
        "NCCL_PXN_DISABLE": "0",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_NET_GDR_LEVEL": "4",
        "NCCL_IB_RETRY_CNT": "7",
        "NCCL_IB_TIMEOUT": "32",
        "NCCL_IB_QPS_PER_CONNECTION": "8",
        "NCCL_P2P_LEVEL": "NVL",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_MIN_CTAS": "4",
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "^openib",
        "OMPI_MCA_routed": "direct",
        "OMPI_MCA_routed_radix": "1024",
        "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
        "OMPI_MCA_oob_tcp_if_include": "'"${MLP_SOCKET_IFNAME}"'",
        "OMPI_MCA_btl_tcp_if_include": "'"${MLP_SOCKET_IFNAME}"'"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}


