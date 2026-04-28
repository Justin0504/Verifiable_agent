# SEVA v3: Experiment Reproduction Guide

**Goal**: Train SEVA 7B (full-parameter) to reach 81+ F1 on ClearFacts (matching MiniCheck-7B from the VtV paper).

## Overview

SEVA (Self-Evolving Verification Agent) is a structured evidence verification model. We use a 3-stage training pipeline:

1. **Stage 1 — Binary NLI Pretraining** (59K samples, 1 epoch): Teaches the model basic attribution classification (Attributable / Not Attributable)
2. **Stage 2 — Structured SFT** (5K samples, 3 epochs): Teaches structured JSON output with reasoning chains, evidence spans, confidence scores, and error diagnosis
3. **Stage 3 — GRPO** (64K prompts, 5 epochs): Reinforcement learning with a custom 8-component process reward function

Base model: **Qwen2.5-7B-Instruct**

## Existing Results (LoRA baseline — what we need to beat with full FT + GRPO)

### ClearFacts (N=1,590) — Primary benchmark

| Model | Acc | F1 | Notes |
|-------|-----|-----|-------|
| MiniCheck-7B (target) | ~81% | **0.810** | VtV paper, our target |
| GPT-4o-mini (structured) | 69.9% | 0.698 | Commercial API baseline |
| **SEVA-GRPO 3B** | 69.6% | **0.690** | 3B with GRPO — shows GRPO works |
| MiniCheck-Flan-T5-Large (binary) | 68.3% | 0.683 | Binary-only, no structured output |
| SEVA-SFT 7B LoRA-128 (single stage) | 68.6% | 0.686 | Single-stage LoRA r=128 |
| SEVA-SFT 7B LoRA S2 (two-stage) | 67.5% | 0.674 | Two-stage LoRA r=64 |
| SEVA-SFT 7B LoRA S1 (NLI only) | 66.9% | 0.669 | Stage 1 only |
| SEVA-SFT 3B | 65.2% | 0.649 | 3B full FT |
| Qwen2.5-7B CoT (zero-shot) | 54.5% | 0.508 | No training |

### FEVER (N=200)

| Model | Acc | F1 |
|-------|-----|-----|
| SEVA-SFT 7B LoRA S1 | 95.0% | **0.943** |
| SEVA-SFT 7B LoRA S2 | 93.0% | 0.921 |
| GPT-4o-mini | 92.5% | 0.910 |
| GRPO 3B | 85.0% | 0.849 |
| SFT 3B | 76.5% | 0.763 |

### TruthfulQA (N=400)

| Model | Acc | F1 |
|-------|-----|-----|
| GRPO 3B | 82.8% | **0.827** |
| SFT 3B | 74.7% | 0.721 |
| SEVA-SFT 7B LoRA S2 | 71.0% | 0.688 |

### Key Observations

- **GRPO dramatically improves 3B**: ClearFacts 0.649→0.690, TruthfulQA 0.721→0.827
- **LoRA 7B underperforms expectations**: Two-stage LoRA (0.674) < single-stage LoRA (0.686), suggesting LoRA capacity limits structured output learning
- **Full-parameter 7B + GRPO is the path to 81**: Full FT removes LoRA bottleneck, GRPO adds ~4-5 F1 points (proven on 3B)

### Sample Model Output (SEVA-SFT 7B LoRA S2)

**Correct — Attributable:**
```json
// Claim: "James Davis, the candidate of the Green Party, received the lowest number of votes..."
// Gold: Attributable | Pred: Attributable | Conf: 1.0
{
  "evidence_alignment": [
    {"claim_span": "the lowest number of votes", "source_span": "Jim Davies | 4,106 | 16.9 | +2.6", "status": "match"},
    {"claim_span": "James Davis, the candidate of the Green Party", "source_span": "Jim Davies | 4,106 | 16.9 | +2.6", "status": "match"}
  ],
  "reasoning_chain": ["The table shows Jim Davies received 4,106 votes...", "Comparing with other candidates..."],
  "label": "Attributable",
  "confidence": 1.0
}
```

**Correct — Not Attributable:**
```json
// Claim: "The Dewildemania belongs to the Plantae Kingdom and Asteraceae Tribe..."
// Gold: Not Attributable | Pred: Not Attributable | Conf: 0.8
{
  "evidence_alignment": [
    {"claim_span": "Plantae Kingdom", "source_span": "Kingdom: | [[Plantae|Plantae]]", "status": "match"},
    {"claim_span": "Asteraceae Tribe", "source_span": "...", "status": "contradiction"}
  ],
  "reasoning_chain": ["The source confirms Plantae Kingdom...", "But Asteraceae is a family, not a tribe..."],
  "label": "Not Attributable",
  "confidence": 0.8,
  "error_type": "entity_error"
}
```

---

## Hardware Requirements

- **Stage 1 & 2 (SFT)**: 4x A100 80GB (or 4x A40 48GB with DeepSpeed ZeRO-3)
  - Single GPU: possible with Adafactor optimizer + gradient checkpointing + max_length=512 on 48GB GPU
  - 4x A100: ~4 hours for Stage 1, ~1 hour for Stage 2
- **Stage 3 (GRPO)**: 4x A100 80GB recommended
  - Uses vLLM for rollout generation + FSDP for training
  - ~8-12 hours for 5 epochs

## Environment Setup

```bash
# Python 3.10+
pip install torch>=2.1 transformers>=4.40 accelerate datasets peft
pip install deepspeed  # for multi-GPU SFT
pip install vllm==0.6.3  # for GRPO rollouts (must match verl version)
pip install verl==0.3.0.post1  # GRPO trainer
pip install flash-attn  # optional but recommended

# Clone repo
git clone <repo_url>
cd Verifiable_agent
```

## Data

All training and evaluation data is in `data/attribution/`:

### Training Data

| File | Samples | Stage | Description |
|------|---------|-------|-------------|
| `sft_train_full.jsonl` | 59,500 | 1 | Binary NLI training data. Format: `{"messages": [system, user, assistant]}` |
| `seva_sft_train.jsonl` | 4,992 | 2 | Structured SFT data. Same format but assistant outputs structured JSON |
| `seva_grpo_train.parquet` | 63,992 | 3 | GRPO prompts with ground truth for reward computation |
| `seva_grpo_val.parquet` | 500 | 3 | GRPO validation set |

Note: `sft_train.jsonl` is identical to `sft_train_full.jsonl` (kept for backward compatibility).

### Evaluation Data

| File | Samples | Description |
|------|---------|-------------|
| `clearfacts.jsonl` | 1,590 | Primary benchmark — claim-source attribution |
| `fever.jsonl` | 200 | Fact verification (3-class: Supported/Contradicted/NotEnoughInfo) |
| `truthfulqa.jsonl` | 400 | Truthfulness detection |

### Data Format

**SFT data** (`messages` format):
```json
{
  "messages": [
    {"role": "system", "content": "You are a fact verification expert..."},
    {"role": "user", "content": "Claim: ... Source: ..."},
    {"role": "assistant", "content": "{\"label\": \"Attributable\", \"confidence\": 0.92, ...}"}
  ]
}
```

**GRPO data** (parquet columns):
- `prompt`: JSON-encoded list of messages (system + user only)
- `reward_model`: JSON with `ground_truth` containing `target` label, `claim`, `source`, `error_type`
- `extra_info`: metadata for curriculum learning (difficulty scores)

---

## Stage 1: Binary NLI Pretraining

```bash
# Single GPU (48GB, Adafactor) — ~31GB peak VRAM
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_seva_sft.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --train-file data/attribution/sft_train_full.jsonl \
    --output-dir checkpoints/seva_v3_7b/stage1_nli \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 32 \
    --lr 1e-5 \
    --max-length 512 \
    --optim-8bit

# Multi-GPU (4x A100, DeepSpeed ZeRO-3, recommended)
torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/train_seva_sft.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --train-file data/attribution/sft_train_full.jsonl \
    --output-dir checkpoints/seva_v3_7b/stage1_nli \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 1e-5 \
    --max-length 1024
```

**Key notes:**
- The script auto-detects multi-GPU and configures DeepSpeed ZeRO-3
- `--optim-8bit` uses Adafactor optimizer (lower memory, works on single GPU)
- Without `--optim-8bit`, uses AdamW (requires multi-GPU or 80GB GPU)
- Saves checkpoints every 200 steps, keeps last 2
- Output: `checkpoints/seva_v3_7b/stage1_nli/final/`

**Expected results after Stage 1 (LoRA reference):**
- ClearFacts: 66.9% acc, 0.669 F1
- FEVER: 95.0% acc, 0.943 F1

## Stage 2: Structured SFT

```bash
# Uses Stage 1 output as base model
torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/train_seva_sft.py \
    --base-model checkpoints/seva_v3_7b/stage1_nli/final \
    --train-file data/attribution/seva_sft_train.jsonl \
    --output-dir checkpoints/seva_v3_7b/stage2_structured \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 3e-6 \
    --max-length 1024
```

**Key notes:**
- Lower LR (3e-6 vs 1e-5) to avoid catastrophic forgetting of Stage 1 knowledge
- 3 epochs on the smaller structured dataset
- Output: `checkpoints/seva_v3_7b/stage2_structured/final/`

**Expected results after Stage 2 (LoRA reference):**
- ClearFacts: 67.5% acc, 0.674 F1
- FEVER: 93.0% acc, 0.921 F1
- TruthfulQA: 71.0% acc, 0.688 F1

**Run evaluation after each stage:**
```bash
python3 -u scripts/eval_seva.py \
    --model checkpoints/seva_v3_7b/stage2_structured/final \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/seva_v3_7b_s2
```

## Stage 3: GRPO with Process Reward

This stage uses [veRL](https://github.com/volcengine/verl) for Group Relative Policy Optimization.

### Reward Function

The reward is computed by `seva_reward_v3.py` with 8 grounded components:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| R_format | 0.10 | Valid JSON with normalized label |
| R_accuracy | 0.50-0.80 | Correct label match (adaptive weight — higher when model is already good) |
| R_calibration | 0.10-0.25 | Confidence aligns with correctness |
| R_alignment | 0.15-0.30 | Evidence spans grounded in claim/source (substring overlap) |
| R_chain | 0.10-0.25 | Reasoning steps grounded + internally consistent |
| R_coherence | 0.10-0.20 | Cross-component consistency penalties |
| R_diagnosis | 0.05-0.15 | Error type correctness for "Not Attributable" |
| R_specificity | 0.05-0.10 | Penalizes generic/templated outputs |

### Running GRPO

```bash
export SEVA_REWARD_MODULE="seva_reward_v3"

python -m verl.trainer.main_ppo \
    --config-name='seva_grpo' \
    data.train_files="data/attribution/seva_grpo_train.parquet" \
    data.val_files="data/attribution/seva_grpo_val.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=768 \
    data.max_response_length=512 \
    data.truncation=left \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="checkpoints/seva_v3_7b/stage2_structured/final" \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    trainer.logger='["console"]' \
    trainer.project_name="seva-v3" \
    trainer.experiment_name="seva_v3_7b_grpo" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.val_before_train=True \
    trainer.total_epochs=5
```

**Key notes:**
- veRL 0.3 API is used; veRL 0.7+ has breaking changes (DTensorSpec requires torch 2.6+)
- vLLM 0.6.3 is required for veRL 0.3 compatibility
- `tensor_model_parallel_size=2` for 7B model across GPUs
- `gpu_memory_utilization=0.25` leaves room for the actor model
- Curriculum learning is available via `src/training/curriculum.py` (optional)

**Expected improvement from GRPO (based on 3B results):**
- ClearFacts: +4-5 F1 points (3B went 0.649→0.690)
- TruthfulQA: +10 F1 points (3B went 0.721→0.827)

### Curriculum Learning (Optional)

```python
from src.training.curriculum import CurriculumManager

cm = CurriculumManager("data/attribution/seva_grpo_train.parquet")
for epoch in range(5):
    epoch_path = cm.get_epoch_data(epoch, total_epochs=5)
    # Use epoch_path as data.train_files for this epoch
```

Schedule: easy-only → add medium → all data → oversample hard examples.

---

## Evaluation

```bash
python3 -u scripts/eval_seva.py \
    --model checkpoints/seva_v3_7b/stage3_grpo/latest \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/seva_v3_7b_final
```

### Benchmarks

| Benchmark | N | Description | Target |
|-----------|---|-------------|--------|
| ClearFacts | 1,590 | Primary benchmark. Claim-source attribution | 81+ F1 |
| FEVER | 200 | Fact verification (3-class) | 94+ F1 |
| TruthfulQA | 400 | Truthfulness detection | 83+ F1 |

### Quality Metrics (in structured output)

Beyond accuracy/F1, `eval_seva.py` also measures output quality:
- **alignment_quality**: How well evidence spans map to claim/source text
- **chain_quality**: Reasoning step quality and internal consistency
- **groundedness**: Whether spans are actually from the source
- **ECE**: Expected calibration error (confidence vs correctness)

---

## Key Files

```
scripts/
  train_seva_sft.py              # SFT training (Stages 1 & 2)
  eval_seva.py                   # Evaluation on benchmarks
  generate_adversarial_probes.py # Generate training data
  run_self_evolution.py          # Self-evolution loop (data generation)
  train_seva_3stage_full.sh      # All-in-one 3-stage pipeline

src/
  training/
    curriculum.py                # Curriculum learning for GRPO

seva_reward_v3.py                # 8-component GRPO reward function

data/attribution/
  sft_train_full.jsonl           # Stage 1 data (59K)
  seva_sft_train.jsonl           # Stage 2 data (5K)
  seva_grpo_train.parquet        # Stage 3 data (64K)
  seva_grpo_val.parquet          # Stage 3 validation (500)
  clearfacts.jsonl               # Eval: ClearFacts (1,590)
  fever.jsonl                    # Eval: FEVER (200)
  truthfulqa.jsonl               # Eval: TruthfulQA (400)

results/                         # Existing experiment results (JSON)
  baselines/                     # GPT-4o-mini, MiniCheck, Qwen zero-shot
  seva_lora2s_7b_s1/             # 7B LoRA Stage 1 results
  seva_lora2s_7b_s2/             # 7B LoRA Stage 2 results
  sft_6bench_eval/               # 3B SFT results (6 benchmarks)
  grpo_6bench_eval/              # 3B GRPO results (6 benchmarks)
  zeroshot_6bench_eval/          # 3B zero-shot results
```

## Quick Start (All 3 Stages)

```bash
# Full pipeline script (adjust GPU count and paths as needed)
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # or local path
export DATA_DIR="data/attribution"
export CKPT_DIR="checkpoints/seva_v3_full_7b"
bash scripts/train_seva_3stage_full.sh
```

## Troubleshooting

- **Qwen2.5 rope_scaling "default" type**: vLLM 0.6.3 may not recognize this. Patch `vllm/model_executor/layers/rotary_embedding.py` to treat "default" as standard RoPE.
- **AutoModelForVision2Seq import error**: transformers 5.x removed this. Patch `verl/workers/fsdp_workers.py` with try/except.
- **OOM on single GPU**: Use `--optim-8bit` flag and reduce `--max-length` to 512 or 256. Adafactor uses ~31GB peak vs ~47GB for AdamW on 7B.
- **DeepSpeed CPU Adam compile failure**: If `python3-dev` headers are missing, set `offload_optimizer.device` to `"none"` in the DeepSpeed config (the script does this automatically).
- **veRL import errors with torch < 2.6**: Stick with veRL 0.3.0.post1. veRL 0.7+ requires torch 2.6+ for DTensorSpec.

## Contact

Justin Yuan (USC) — ayuan@usc.edu
