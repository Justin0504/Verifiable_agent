#!/bin/bash
# SEVA v2 Process Reward GRPO Training
#
# Key differences from train_attribution.sh:
#   - Uses seva_reward.py (process reward with component scoring)
#   - Longer response length (512) for structured output
#   - Higher temperature (1.2) for exploration diversity
#   - Larger group size (8) for better advantage estimation
#   - Entropy coefficient to prevent collapse
#
# Usage:
#   bash train_seva_grpo.sh
#   bash train_seva_grpo.sh --model /path/to/seva_sft_checkpoint

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate verl

set -x

# ============================================================
# GPU Configuration
# ============================================================
tp=1
dp=2
gpus=2
batch_per_gpu=1  # smaller due to longer response_length
rollout_memory_utilization=0.35

export CUDA_VISIBLE_DEVICES=0,1
export RAY_TMPDIR=/tmp/ray

# ============================================================
# GRPO Configuration
# ============================================================
algorithm=grpo
grpo_group_size=8

# ============================================================
# Model - SEVA v2 SFT checkpoint
# ============================================================
model=${MODEL_PATH:-/home/yinian/verifiable_agent/checkpoints/seva_sft/final}

# ============================================================
# Data - SEVA v2 format (structured output prompts)
# ============================================================
DATA_DIR=${DATA_DIR:-/home/yinian/verifiable_agent/data/attribution}
TRAIN_DATA="${DATA_DIR}/seva_grpo_train.parquet"
VAL_DATA="${DATA_DIR}/seva_grpo_val.parquet"

# ============================================================
# Output
# ============================================================
EXPERIMENT_NAME="seva_v2_grpo_$(date +%Y%m%d_%H%M)"

echo "============================================"
echo "SEVA v2 Process Reward GRPO Training"
echo "============================================"
echo "Model:     ${model}"
echo "Train:     ${TRAIN_DATA}"
echo "Val:       ${VAL_DATA}"
echo "GPUs:      ${gpus} (tp=${tp}, dp=${dp})"
echo "Group:     ${grpo_group_size}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "============================================"

kill -9 $(lsof -t -i :8000) 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name='seva_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=64 \
    data.max_prompt_length=768 \
    data.max_response_length=512 \
    data.truncation=left \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
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
    trainer.project_name="seva-v2" \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.total_epochs=5 \
    "$@"
