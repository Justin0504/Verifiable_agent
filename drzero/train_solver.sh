#!/bin/bash
# Activate conda env
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate verl
# Fact Verification Solver Training with GRPO
# Adapted from Dr.Zero iter1_solver.sh for single-turn verification task
#
# Usage:
#   bash train_solver.sh                    # default: 2 GPUs
#   bash train_solver.sh --gpus 4           # use 4 GPUs
#   bash train_solver.sh --model /path/to/sft_model

set -x

# ============================================================
# GPU Configuration
# RTX 6000 Ada: 48GB each, 8 available
# 3B model: ~6GB weights, single-turn → light memory usage
# ============================================================
tp=1                          # tensor parallel (1 is fine for 3B)
dp=2                          # data parallel
gpus=2                        # total GPUs (tp * dp)
batch_per_gpu=2               # micro batch per GPU (reduced for OOM)
rollout_memory_utilization=0.4

# Use GPU 1-2 (GPU 0 is occupied)
export CUDA_VISIBLE_DEVICES=0,1

# Ray needs short socket path (AF_UNIX 107 byte limit)
export RAY_TMPDIR=/tmp/ray

# ============================================================
# GRPO Configuration
# ============================================================
algorithm=grpo
grpo_group_size=5             # 5 rollouts per prompt for advantage estimation

# ============================================================
# Model - use SFT checkpoint as starting point
# ============================================================
model=${MODEL_PATH:-/home/yinian/verifiable_agent/checkpoints/sft_qwen3b/final}

# ============================================================
# Data
# ============================================================
DATA_DIR=${DATA_DIR:-./data/fact_verification}
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/val.parquet"
TEST_DIR="${DATA_DIR}/test"

# ============================================================
# Output
# ============================================================
EXPERIMENT_NAME="fact_verify_grpo_$(date +%Y%m%d_%H%M)"
CONFIG_PATH="./config"

echo "============================================"
echo "Fact Verification GRPO Training"
echo "============================================"
echo "Model:     ${model}"
echo "Train:     ${TRAIN_DATA}"
echo "Val:       ${VAL_DATA}"
echo "GPUs:      ${gpus} (tp=${tp}, dp=${dp})"
echo "Group:     ${grpo_group_size}"
echo "Batch:     ${batch_per_gpu}/gpu × ${gpus} gpus × grad_accum"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "============================================"

# Kill any existing sglang server
kill -9 $(lsof -t -i :8000) 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name='fact_verification_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=128 \
    data.max_prompt_length=768 \
    data.max_response_length=256 \
    data.truncation=left \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${grpo_group_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    trainer.logger='["console"]' \
    trainer.project_name="fact-verification" \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.total_epochs=3 \
    "$@"
