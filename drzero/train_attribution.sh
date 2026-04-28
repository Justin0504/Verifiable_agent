#!/bin/bash
# SEVA Attribution Verification GRPO Training
# Adapted from train_solver.sh for binary attribution task
#
# Usage:
#   bash train_attribution.sh
#   bash train_attribution.sh --model /path/to/sft_checkpoint

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate verl

set -x

# ============================================================
# GPU Configuration
# ============================================================
tp=1
dp=2
gpus=2
batch_per_gpu=2
rollout_memory_utilization=0.4

export CUDA_VISIBLE_DEVICES=0,1
export RAY_TMPDIR=/tmp/ray

# ============================================================
# GRPO Configuration
# ============================================================
algorithm=grpo
grpo_group_size=5

# ============================================================
# Model - SFT checkpoint (trained on ANLI attribution data)
# ============================================================
model=${MODEL_PATH:-/home/yinian/verifiable_agent/checkpoints/sft_attribution/final}

# ============================================================
# Data - Attribution format
# ============================================================
DATA_DIR=${DATA_DIR:-/home/yinian/verifiable_agent/data/attribution}
TRAIN_DATA="${DATA_DIR}/grpo_train.parquet"
VAL_DATA="${DATA_DIR}/grpo_val.parquet"

# ============================================================
# Output
# ============================================================
EXPERIMENT_NAME="seva_attribution_grpo_$(date +%Y%m%d_%H%M)"

echo "============================================"
echo "SEVA Attribution GRPO Training"
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
    --config-name='attribution_grpo' \
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
    trainer.project_name="seva-attribution" \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.total_epochs=3 \
    "$@"
