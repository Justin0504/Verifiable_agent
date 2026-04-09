#!/bin/bash
# ============================================================
# SEVA Full Pipeline: SFT → GRPO → Self-Evolution
# Run on server: bash scripts/run_seva_pipeline.sh
# ============================================================

set -e

# ============================================================
# Paths (edit as needed)
# ============================================================
BASE_MODEL=${BASE_MODEL:-/home/yinian/models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}
DATA_DIR=${DATA_DIR:-/home/yinian/verifiable_agent/data/attribution}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/yinian/verifiable_agent/checkpoints}

SFT_OUTPUT="${CHECKPOINT_DIR}/sft_attribution"
SEVA_OUTPUT="${CHECKPOINT_DIR}/seva_attribution"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

echo "============================================"
echo "SEVA Attribution Verification Pipeline"
echo "============================================"
echo "Base model:  ${BASE_MODEL}"
echo "Data dir:    ${DATA_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "============================================"

# ============================================================
# Step 1: Prepare data (if not already done)
# ============================================================
if [ ! -f "${DATA_DIR}/sft_train.jsonl" ]; then
    echo ""
    echo "[Step 1] Preparing attribution data..."
    python scripts/prepare_attribution_data.py
else
    echo ""
    echo "[Step 1] Data already prepared, skipping"
    echo "  SFT train: $(wc -l < ${DATA_DIR}/sft_train.jsonl) samples"
    echo "  GRPO train: ${DATA_DIR}/grpo_train.parquet"
    echo "  ClearFacts: $(wc -l < ${DATA_DIR}/clearfacts.jsonl) samples"
fi

# ============================================================
# Step 2: SFT warm-up
# ============================================================
if [ ! -d "${SFT_OUTPUT}/final" ]; then
    echo ""
    echo "[Step 2] Running SFT warm-up training..."
    python scripts/run_sft_attribution.py \
        --model "${BASE_MODEL}" \
        --data "${DATA_DIR}/sft_train.jsonl" \
        --val-data "${DATA_DIR}/sft_val.jsonl" \
        --output "${SFT_OUTPUT}" \
        --epochs 2 \
        --lr 2e-5 \
        --batch-size 4 \
        --grad-accum 8
else
    echo ""
    echo "[Step 2] SFT checkpoint found, skipping"
fi

# ============================================================
# Step 3: Evaluate SFT baseline
# ============================================================
echo ""
echo "[Step 3] Evaluating SFT baseline on ClearFacts..."
python scripts/eval_attribution.py \
    --model "${SFT_OUTPUT}/final" \
    --benchmarks clearfacts \
    --max-samples 200

# ============================================================
# Step 4: GRPO training (single round, baseline)
# ============================================================
if [ ! -d "${CHECKPOINT_DIR}/grpo_attribution" ]; then
    echo ""
    echo "[Step 4] Running GRPO training..."
    MODEL_PATH="${SFT_OUTPUT}/final" \
    DATA_DIR="${DATA_DIR}" \
    bash drzero/train_attribution.sh
else
    echo ""
    echo "[Step 4] GRPO checkpoint found, skipping"
fi

# ============================================================
# Step 5: SEVA self-evolution loop
# ============================================================
echo ""
echo "[Step 5] Running SEVA self-evolution loop..."
echo "  (Requires OPENAI_API_KEY for proposer LLM)"

if [ -z "${OPENAI_API_KEY}" ]; then
    echo "  WARNING: OPENAI_API_KEY not set"
    echo "  Running without proposer (reflect + verify only)"
    python scripts/train_seva.py \
        --model "${SFT_OUTPUT}/final" \
        --epochs 3 \
        --eval-data "${DATA_DIR}/clearfacts.jsonl" \
        --eval-samples 200 \
        --output "${SEVA_OUTPUT}" \
        --skip-probe \
        --skip-refine
else
    python scripts/train_seva.py \
        --model "${SFT_OUTPUT}/final" \
        --epochs 3 \
        --n-probes 40 \
        --eval-data "${DATA_DIR}/clearfacts.jsonl" \
        --eval-samples 200 \
        --output "${SEVA_OUTPUT}" \
        --grpo-script "drzero/train_attribution.sh" \
        --proposer-model gpt-4o-mini \
        --proposer-provider openai
fi

# ============================================================
# Step 6: Final evaluation on all benchmarks
# ============================================================
echo ""
echo "[Step 6] Final evaluation..."
BEST_MODEL=$(ls -td ${SEVA_OUTPUT}/epoch_*/checkpoint 2>/dev/null | head -1)
if [ -z "${BEST_MODEL}" ]; then
    BEST_MODEL="${SFT_OUTPUT}/final"
fi

python scripts/eval_attribution.py \
    --model "${BEST_MODEL}" \
    --all \
    --output-dir "${SEVA_OUTPUT}/final_eval"

echo ""
echo "============================================"
echo "SEVA Pipeline Complete!"
echo "Best model: ${BEST_MODEL}"
echo "Results:    ${SEVA_OUTPUT}/final_eval/"
echo "============================================"
