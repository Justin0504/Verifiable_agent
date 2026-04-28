#!/bin/bash
#SBATCH --job-name=seva-3stage-7b
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=24:00:00
#SBATCH --account=bfsl-delta-gpu
#SBATCH --output=/scratch/bfsl/ayuan/logs/seva_3stage_%j.log

# SEVA v3: 3-Stage Training Pipeline
# Stage 1: Binary NLI pretraining (59K, 2 epochs)
# Stage 2: Structured SFT (5K+, 3 epochs)
# Stage 3: GRPO with v3 process reward (curriculum)

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/scratch/bfsl/ayuan/models/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-/scratch/bfsl/ayuan/Verifiable_agent/data/attribution}"
CKPT_DIR="${CKPT_DIR:-/scratch/bfsl/ayuan/checkpoints/seva_v3_7b}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"

mkdir -p "$CKPT_DIR" /scratch/bfsl/ayuan/logs

echo "============================================="
echo "SEVA v3 Three-Stage Training"
echo "Base: $BASE_MODEL"
echo "Data: $DATA_DIR"
echo "Output: $CKPT_DIR"
echo "============================================="

# ─── Stage 1: Binary NLI Pretraining ─────────────────────────
echo ""
echo "=== STAGE 1: Binary NLI (59K, 2 epochs) ==="
echo "=== Started at $(date) ==="

STAGE1_OUT="$CKPT_DIR/stage1_nli"

if [ -d "$STAGE1_OUT/final" ]; then
    echo "Stage 1 already complete, skipping..."
else
    python3 -u scripts/train_seva_sft.py \
        --base-model "$BASE_MODEL" \
        --train-file "$DATA_DIR/sft_train_full.jsonl" \
        --output-dir "$STAGE1_OUT" \
        --epochs 2 \
        --batch-size 1 \
        --grad-accum 16 \
        --lr 2e-5 \
        --max-length 1024 \
        --lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA

    echo "=== Stage 1 finished at $(date) ==="
fi

if [ ! -d "$STAGE1_OUT/final" ]; then
    echo "ERROR: Stage 1 failed"
    exit 1
fi

# ─── Stage 2: Structured SFT ─────────────────────────────────
echo ""
echo "=== STAGE 2: Structured SFT (5K, 3 epochs) ==="
echo "=== Started at $(date) ==="

STAGE2_OUT="$CKPT_DIR/stage2_structured"

if [ -d "$STAGE2_OUT/final" ]; then
    echo "Stage 2 already complete, skipping..."
else
    python3 -u scripts/train_seva_sft.py \
        --base-model "$STAGE1_OUT/final" \
        --train-file "$DATA_DIR/seva_sft_train.jsonl" \
        --output-dir "$STAGE2_OUT" \
        --epochs 3 \
        --batch-size 1 \
        --grad-accum 16 \
        --lr 5e-6 \
        --max-length 1024 \
        --lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA

    echo "=== Stage 2 finished at $(date) ==="
fi

if [ ! -d "$STAGE2_OUT/final" ]; then
    echo "ERROR: Stage 2 failed"
    exit 1
fi

# ─── Stage 3: GRPO with v3 Process Reward ────────────────────
echo ""
echo "=== STAGE 3: GRPO with v3 Reward (curriculum) ==="
echo "=== Started at $(date) ==="

STAGE3_OUT="$CKPT_DIR/stage3_grpo"
GRPO_DATA="$DATA_DIR/seva_grpo_train.parquet"

if [ ! -f "$GRPO_DATA" ]; then
    echo "ERROR: GRPO training data not found at $GRPO_DATA"
    echo "Generate with: python scripts/generate_seva_sft_data.py --format grpo"
    exit 1
fi

# Ensure v3 reward is used
export SEVA_REWARD_MODULE="seva_reward_v3"

source /usr/local/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate verl 2>/dev/null || true

kill -9 $(lsof -t -i :8000) 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name='seva_grpo' \
    data.train_files="$GRPO_DATA" \
    data.train_batch_size=64 \
    data.max_prompt_length=768 \
    data.max_response_length=512 \
    data.truncation=left \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$STAGE2_OUT/final" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    trainer.logger='["console"]' \
    trainer.project_name="seva-v3" \
    trainer.experiment_name="seva_v3_7b_grpo" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.total_epochs=5

echo "=== Stage 3 finished at $(date) ==="

# ─── Final Evaluation ────────────────────────────────────────
echo ""
echo "=== FINAL EVALUATION ==="
echo "=== Started at $(date) ==="

# Find latest GRPO checkpoint
GRPO_CKPT=$(ls -d "$STAGE3_OUT"/global_step_* 2>/dev/null | sort -V | tail -1)
if [ -z "$GRPO_CKPT" ]; then
    echo "No GRPO checkpoint found, evaluating Stage 2 model instead"
    GRPO_CKPT="$STAGE2_OUT/final"
fi

export DATA_DIR="$DATA_DIR"
export RESULTS_DIR="$CKPT_DIR/results"
python3 -u scripts/eval_seva.py \
    --model "$GRPO_CKPT" \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir "$CKPT_DIR/results/final"

echo ""
echo "============================================="
echo "SEVA v3 Training Complete!"
echo "Stage 1 (NLI):        $STAGE1_OUT/final"
echo "Stage 2 (Structured): $STAGE2_OUT/final"
echo "Stage 3 (GRPO):       $GRPO_CKPT"
echo "Results:              $CKPT_DIR/results/final"
echo "============================================="
