#!/bin/bash
#SBATCH --job-name=seva-v3-full-7b
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240g
#SBATCH --time=48:00:00
#SBATCH --account=bfsl-delta-gpu
#SBATCH --output=logs/seva_v3_full_%j.log

# SEVA v3: 3-Stage FULL FINE-TUNING Pipeline (no LoRA)
# Stage 1: Binary NLI pretraining (59K, 1 epoch, full params)
# Stage 2: Structured SFT (5K, 3 epochs, full params)
# Stage 3: GRPO with v3 process reward (4 GPU)
#
# Requires: 4x A100 80GB (or 4x A40 48GB with DeepSpeed ZeRO-3)
#
# Usage:
#   export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash scripts/train_seva_3stage_full.sh

set -euo pipefail

# Auto-detect repo root (works from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/data/attribution}"
CKPT_DIR="${CKPT_DIR:-$REPO_DIR/checkpoints/seva_v3_full_7b}"

mkdir -p "$CKPT_DIR" logs

echo "============================================="
echo "SEVA v3 FULL Fine-Tuning (7B, 4x A40)"
echo "Base: $BASE_MODEL"
echo "Data: $DATA_DIR"
echo "Output: $CKPT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================="

# ─── Stage 1: Binary NLI Pretraining (FULL) ──────────────────
echo ""
echo "=== STAGE 1: Binary NLI — FULL FT (59K, 1 epoch) ==="
echo "=== Started at $(date) ==="

STAGE1_OUT="$CKPT_DIR/stage1_nli"

if [ -d "$STAGE1_OUT/final" ]; then
    echo "Stage 1 already complete, skipping..."
else
    # Full fine-tuning: no --lora flag
    # batch-size 1 x grad-accum 16 x 4 GPU = effective batch 64
    python3 -u scripts/train_seva_sft.py \
        --base-model "$BASE_MODEL" \
        --train-file "$DATA_DIR/sft_train_full.jsonl" \
        --output-dir "$STAGE1_OUT" \
        --epochs 1 \
        --batch-size 1 \
        --grad-accum 16 \
        --lr 1e-5 \
        --max-length 1024

    echo "=== Stage 1 finished at $(date) ==="
fi

if [ ! -d "$STAGE1_OUT/final" ]; then
    echo "ERROR: Stage 1 failed"
    exit 1
fi

# ─── Stage 2: Structured SFT (FULL) ─────────────────────────
echo ""
echo "=== STAGE 2: Structured SFT — FULL FT (5K, 3 epochs) ==="
echo "=== Started at $(date) ==="

STAGE2_OUT="$CKPT_DIR/stage2_structured"

if [ -d "$STAGE2_OUT/final" ]; then
    echo "Stage 2 already complete, skipping..."
else
    # Lower LR for structured stage to avoid catastrophic forgetting
    python3 -u scripts/train_seva_sft.py \
        --base-model "$STAGE1_OUT/final" \
        --train-file "$DATA_DIR/seva_sft_train.jsonl" \
        --output-dir "$STAGE2_OUT" \
        --epochs 3 \
        --batch-size 1 \
        --grad-accum 16 \
        --lr 3e-6 \
        --max-length 1024

    echo "=== Stage 2 finished at $(date) ==="
fi

if [ ! -d "$STAGE2_OUT/final" ]; then
    echo "ERROR: Stage 2 failed"
    exit 1
fi

# ─── Stage 3: GRPO with v3 Process Reward ────────────────────
echo ""
echo "=== STAGE 3: GRPO with v3 Reward ==="
echo "=== Started at $(date) ==="

STAGE3_OUT="$CKPT_DIR/stage3_grpo"
GRPO_DATA="$DATA_DIR/seva_grpo_train.parquet"

if [ ! -f "$GRPO_DATA" ]; then
    echo "WARNING: GRPO data not found at $GRPO_DATA, skipping Stage 3"
    echo "Evaluate Stage 2 instead"
    FINAL_MODEL="$STAGE2_OUT/final"
else
    # Copy v3 reward into verl's custom_reward directory
    cp "$REPO_DIR/seva_reward_v3.py" "$REPO_DIR/drzero/verl/custom_reward/seva_reward_v3.py"

    # Patch config to use v3 reward directly
    sed -i.bak 's|verl/custom_reward/seva_reward.py|verl/custom_reward/seva_reward_v3.py|' \
        "$REPO_DIR/drzero/config/seva_grpo.yaml" 2>/dev/null || \
    sed -i '' 's|verl/custom_reward/seva_reward.py|verl/custom_reward/seva_reward_v3.py|' \
        "$REPO_DIR/drzero/config/seva_grpo.yaml"

    source /usr/local/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate verl 2>/dev/null || true

    kill -9 $(lsof -t -i :8000) 2>/dev/null || true

    python -m verl.trainer.main_ppo \
        --config-name='seva_grpo' \
        data.train_files="$GRPO_DATA" \
        data.val_files="$DATA_DIR/seva_grpo_val.parquet" \
        data.train_batch_size=64 \
        data.max_prompt_length=768 \
        data.max_response_length=512 \
        data.truncation=left \
        algorithm.kl_ctrl.kl_coef=0.02 \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path="$STAGE2_OUT/final" \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.grad_clip=0.1 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=0.8 \
        actor_rollout_ref.rollout.top_p=0.9 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        trainer.logger='["console"]' \
        trainer.project_name="seva-v3-full" \
        trainer.experiment_name="seva_v3_full_7b_grpo" \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=50 \
        trainer.test_freq=20 \
        trainer.val_before_train=True \
        trainer.total_epochs=5

    echo "=== Stage 3 finished at $(date) ==="

    FINAL_MODEL=$(ls -d "$STAGE3_OUT"/global_step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "$FINAL_MODEL" ]; then
        FINAL_MODEL="$STAGE2_OUT/final"
    fi
fi

# ─── Final Evaluation ────────────────────────────────────────
echo ""
echo "=== FINAL EVALUATION ==="
echo "=== Started at $(date) ==="

export DATA_DIR="${DATA_DIR}"
export RESULTS_DIR="${CKPT_DIR}/results"
python3 -u scripts/eval_seva.py \
    --model "${FINAL_MODEL:-$STAGE2_OUT/final}" \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir "$CKPT_DIR/results/final"

echo ""
echo "============================================="
echo "SEVA v3 FULL FT Complete!"
echo "Stage 1 (NLI):        $STAGE1_OUT/final"
echo "Stage 2 (Structured): $STAGE2_OUT/final"
echo "Stage 3 (GRPO):       ${FINAL_MODEL:-N/A}"
echo "Results:              $CKPT_DIR/results/final"
echo "============================================="
