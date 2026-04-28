# SEVA v3: 7B Full Fine-Tuning Experiment Guide

> **For collaborators**: This is a self-contained experiment reproduction guide. Clone this repo onto a machine with 4xA100 80GB GPUs and follow the steps below. All training data, eval data, and code are included in the repo — no extra downloads needed.

---

## 1. Experiments to Run

Our best 7B result so far used LoRA and reached ClearFacts F1=0.674. The target is **81+ F1** (matching MiniCheck-7B from the VtV paper).

**Core hypothesis**: LoRA lacks the capacity to learn structured output well. Switching to full-parameter fine-tuning removes this bottleneck. GRPO is proven effective on 3B (+4 F1 points), and should yield even larger gains on 7B.

### Three stages (run in order):

| Stage | Description | Est. Time | Output |
|-------|-------------|-----------|--------|
| **Stage 1: Binary NLI** | 59K samples, 1 epoch, full fine-tuning | ~4h on 4xA100 | `checkpoints/seva_v3_7b/stage1_nli/final/` |
| **Stage 2: Structured SFT** | 5K samples, 3 epochs, full fine-tuning | ~1h on 4xA100 | `checkpoints/seva_v3_7b/stage2_structured/final/` |
| **Stage 3: GRPO** | 64K prompts, 5 epochs, 8-component process reward | ~8-12h on 4xA100 | `checkpoints/seva_v3_7b/stage3_grpo/` |

Run evaluation after each stage (see Section 7).

---

## 2. Existing Results (baselines to beat)

### ClearFacts (N=1,590) — Primary metric

| Model | Acc | F1 | Training |
|-------|-----|-----|----------|
| MiniCheck-7B (target) | ~81% | **0.810** | Full FT on NLI data |
| GPT-4o-mini (structured) | 69.9% | 0.698 | Commercial API |
| **SEVA-GRPO 3B** | 69.6% | **0.690** | Full FT + GRPO (proves GRPO works) |
| SEVA-SFT 7B LoRA-128 (single-stage) | 68.6% | 0.686 | LoRA r=128 |
| SEVA-SFT 7B LoRA S2 (two-stage) | 67.5% | 0.674 | LoRA r=64, two-stage |
| SEVA-SFT 3B | 65.2% | 0.649 | Full FT |

### FEVER (N=200)

| Model | Acc | F1 |
|-------|-----|-----|
| SEVA-SFT 7B LoRA S1 | 95.0% | 0.943 |
| SEVA-SFT 7B LoRA S2 | 93.0% | 0.921 |
| GRPO 3B | 85.0% | 0.849 |

### TruthfulQA (N=400)

| Model | Acc | F1 |
|-------|-----|-----|
| GRPO 3B | 82.8% | 0.827 |
| SEVA-SFT 7B LoRA S2 | 71.0% | 0.688 |

> All existing result JSONs are in `results/`.

---

## 3. Environment Setup

### Hardware

- **Stage 1 & 2 (SFT)**: 4xA100 80GB + DeepSpeed ZeRO-3
- **Stage 3 (GRPO)**: 4xA100 80GB, veRL framework (Ray + FSDP + vLLM)
- Single-GPU is possible for Stage 1/2 with `--optim-8bit` (~31GB peak), but slow

### Install dependencies

```bash
# Core
pip install torch>=2.1 transformers>=4.40 accelerate datasets peft
pip install deepspeed  # multi-GPU SFT (Stage 1 & 2)
pip install scikit-learn pandas numpy  # evaluation

# Stage 3 GRPO (must use these exact versions)
pip install vllm==0.6.3
pip install verl==0.3.0.post1
pip install flash-attn  # optional but recommended

# Clone repo
git clone https://github.com/Justin0504/Verifiable_agent.git
cd Verifiable_agent
```

### Version compatibility (important)

| Component | Requirement | Reason |
|-----------|-------------|--------|
| veRL | **0.3.0.post1** | 0.7+ has breaking changes (DTensorSpec requires torch 2.6+) |
| vLLM | **0.6.3** | veRL 0.3 depends on this version's API |
| torch | >=2.1, <2.6 | veRL 0.3 is incompatible with torch 2.6 |
| transformers | >=4.40 | Qwen2.5 support |

---

## 4. Data

All data is in `data/attribution/`, included in the repo.

### Training data

| File | Samples | Stage | Description |
|------|---------|-------|-------------|
| `sft_train_full.jsonl` | 59,500 | 1 | Binary NLI data. Each line: `{"messages": [system, user, assistant]}`, assistant outputs only the label |
| `seva_sft_train.jsonl` | 4,992 | 2 | Structured SFT data. Same format but assistant outputs full JSON (evidence_alignment, reasoning_chain, etc.) |
| `seva_grpo_train.parquet` | 63,992 | 3 | GRPO training prompts + ground truth for reward computation |
| `seva_grpo_val.parquet` | 500 | 3 | GRPO validation set |

### Evaluation data

| File | Samples | Description |
|------|---------|-------------|
| `clearfacts.jsonl` | 1,590 | Primary benchmark: claim-source attribution |
| `fever.jsonl` | 200 | Fact verification (3-class remapped to binary) |
| `truthfulqa.jsonl` | 400 | Truthfulness detection |

### Data format

**SFT data** (`messages` format, shared by Stage 1 & 2):
```json
{
  "messages": [
    {"role": "system", "content": "You are a fact verification expert..."},
    {"role": "user", "content": "Claim: James Davis received... Source: The 2016 Brisbane..."},
    {"role": "assistant", "content": "{\"label\": \"Attributable\", \"confidence\": 0.92, ...}"}
  ]
}
```
- Stage 1 assistant: binary label only
- Stage 2 assistant: full structured JSON with evidence_alignment + reasoning_chain + error_type

**GRPO data** (parquet columns):
- `prompt`: JSON-encoded messages (system + user, no assistant)
- `reward_model`: JSON with `ground_truth` containing `target` label, `claim`, `source`, `error_type`
- `extra_info`: metadata (difficulty scores for curriculum learning)

---

## 5. Code Structure

```
Verifiable_agent/
|-- scripts/
|   |-- train_seva_sft.py              # [CORE] SFT training script (Stage 1 & 2)
|   |-- train_seva_3stage_full.sh      # [CORE] One-click script for all 3 stages
|   |-- eval_seva.py                   # [CORE] Evaluation script
|   |-- generate_seva_sft_data.py      # Data generation (GPT-4o teacher)
|   |-- generate_adversarial_probes.py # Adversarial sample generation
|   +-- run_self_evolution.py          # Self-evolution loop
|
|-- src/
|   |-- verifier/
|   |   +-- seva_format.py             # [CORE] Structured output schema + system prompt
|   |-- training/
|   |   +-- curriculum.py              # Curriculum learning (optional)
|   |-- benchmarks/                    # Benchmark loaders
|   +-- llm/                           # LLM wrappers (OpenAI, vLLM)
|
|-- seva_reward_v3.py                  # [CORE] 8-component GRPO reward function
|
|-- drzero/
|   |-- config/
|   |   +-- seva_grpo.yaml             # [CORE] GRPO training config (Hydra)
|   +-- verl/custom_reward/
|       |-- seva_reward.py             # v2 reward (used by eval script)
|       +-- seva_reward_v3.py          # v3 reward (used by GRPO training)
|
|-- data/attribution/                  # All training + eval data
+-- results/                           # Existing experiment results (JSON)
```

### Key file descriptions

#### `scripts/train_seva_sft.py` — SFT training (Stage 1 & 2 share this script)
- Auto-detects multi-GPU (`WORLD_SIZE > 1`) and configures DeepSpeed ZeRO-3
- Key args: `--base-model`, `--train-file`, `--epochs`, `--lr`, `--max-length`
- `--optim-8bit`: switches to Adafactor optimizer (single-GPU, peak ~31GB vs AdamW ~47GB)
- `--resume-from`: resume from checkpoint
- Loss masking: only computes loss on assistant tokens (system + user masked to -100)
- Saves model + tokenizer to `{output-dir}/final/` on completion

#### `seva_reward_v3.py` — 8-component process reward
- Entry point: `compute_score(data_source, solution_str, ground_truth, extra_info)` -> float
- Batch entry: `compute_score_batch(...)` -> list[float] (with boundary bonus)
- Returns reward in range [0.0, ~1.65]
- 8 components with dynamic weights that shift across epochs (see Section 6)
- Called by veRL's `NaiveRewardManager` during GRPO training

#### `scripts/eval_seva.py` — Evaluation
- Loads model, generates structured JSON responses on benchmarks, scores them
- Outputs: accuracy, macro_F1, ECE, alignment_quality, chain_quality, groundedness
- Generates `{benchmark}_results.json` (per-sample details) + `summary.json`
- **Important**: default DATA_DIR is hardcoded to a dev server path. You **must** set `export DATA_DIR="$(pwd)/data/attribution"` before running

#### `src/verifier/seva_format.py` — Structured output schema
- Defines `SEVA_SYSTEM_PROMPT`: instructs model to output 6-field JSON
- Defines `SEVA_USER_TEMPLATE`: `"Claim: {claim}\nSource: {source}"`
- Defines `ERROR_TYPES`: 6-category error taxonomy
- `make_system_prompt_with_rules()`: optional ReasoningBank rule injection

#### `drzero/config/seva_grpo.yaml` — GRPO config
- Hydra config for veRL, extends `ppo_trainer` defaults
- The `train_seva_3stage_full.sh` script auto-patches this to use v3 reward and copies `seva_reward_v3.py` into the correct directory
- Rollout settings: temperature=0.8, top_p=0.9, n=16 (16 rollouts per prompt)

---

## 6. Training Logic

### Stage 1: Binary NLI Pretraining

**Goal**: Teach the model claim-source binary classification (Attributable / Not Attributable).

**How it works**:
1. Load Qwen2.5-7B-Instruct base model
2. Standard causal LM SFT on 59K binary NLI samples
3. Only compute loss on assistant tokens (system + user masked to -100)
4. DeepSpeed ZeRO-3 shards model parameters across 4 GPUs

**Hyperparameter rationale**:
- LR=1e-5: standard SFT learning rate for 7B models
- 1 epoch: large dataset (59K), 1 epoch is sufficient for classification; more risks overfitting
- max_length=1024: most claim-source pairs fit within 512 tokens; 1024 provides margin
- effective_batch=64: per_gpu=1 x grad_accum=16 x 4GPU

### Stage 2: Structured SFT

**Goal**: On top of Stage 1's classification ability, teach structured JSON output.

**How it works**:
1. Load Stage 1 checkpoint as base model
2. SFT on 5K structured samples
3. Structured output has 6 fields: evidence_alignment, reasoning_chain, label, confidence, error_type, fix_suggestion

**Hyperparameter rationale**:
- LR=3e-6 (3x lower than Stage 1): prevents catastrophic forgetting of Stage 1 classification
- 3 epochs: small dataset (5K), needs multiple passes to learn the structured format
- Other params same as Stage 1

### Stage 3: GRPO with Process Reward

**Goal**: Use reinforcement learning to further improve structured output quality and classification accuracy.

**How it works**:
1. Load Stage 2 checkpoint as both actor and reference model
2. For each prompt, generate 16 rollouts via vLLM (temperature=0.8, moderate diversity)
3. Score each rollout with the 8-component reward function (normalized weights, sum=1.0)
4. GRPO advantage estimation: compare within each group of 16 rollouts
5. PPO loss with light KL penalty (kl_coef=0.02) updates the actor model

**8-component reward function** (weights normalized to sum=1.0):

| Component | Weight (epoch 1 -> 5) | What it does |
|-----------|----------------------|--------------|
| R_format | ~0.10 | **Strict**: requires all 4 fields (label, confidence, evidence_alignment, reasoning_chain). Partial JSON gets 80% penalty |
| R_accuracy | ~0.55 -> 0.35 | Correct label match. High weight early, decreases to let grounding improve |
| R_calibration | ~0.05 -> 0.15 | **Asymmetric**: optimal confidence ~0.85 for correct; 2x penalty for wrong+confident |
| R_alignment | ~0.10 -> 0.20 | Evidence spans grounded in claim/source. **Strict threshold=0.75** with sliding penalty below |
| R_chain | ~0.08 -> 0.15 | Reasoning step quality: substantive content (>10 chars), consistent with label, 2-4 steps optimal |
| R_coherence | ~0.05 -> 0.08 | Cross-component consistency: label vs error_type, alignment status, chain judgments |
| R_diagnosis | ~0.04 -> 0.07 | Error type correctness + fix_suggestion quality (only for "Not Attributable") |
| R_specificity | ~0.03 -> 0.05 | Penalizes generic/templated outputs, rewards referencing specific claim content |

**Key design decisions**:
- **Strict format reward**: partial JSON (label-only without structured fields) gets near-zero reward. Forces model to produce full structured output
- **Normalized weights** (sum=1.0): stable reward signal across epochs. No more total-reward drift
- **No boundary bonus**: removed because it penalized high-accuracy groups (perverse incentive). GRPO's group-relative advantage estimation already normalizes
- **Stricter span matching** (threshold=0.75): forces near-exact span extraction, not fuzzy paraphrasing. Sliding penalty between 0.5-0.75
- **KL penalty** (kl_coef=0.02): light regularization prevents reward hacking while allowing improvement
- **16 rollouts** (up from 8): better advantage estimation for 7B model on sparse reward
- **Temperature 0.8** (down from 1.2): structured JSON needs more coherent outputs

**Framework**: veRL 0.3 (Ray + FSDP + vLLM)
- vLLM handles rollout generation (fast sampling)
- FSDP handles actor model training (parameter sharding)
- optimizer_offload=True: optimizer states on CPU, saving GPU memory for vLLM

---

## 7. Commands

### Option A: One-click script (recommended)

```bash
# Setup: copy reward function into verl's custom_reward directory (needed for Stage 3)
cp seva_reward_v3.py drzero/verl/custom_reward/seva_reward_v3.py

# Set environment variables
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # or local path
export DATA_DIR="$(pwd)/data/attribution"
export CKPT_DIR="$(pwd)/checkpoints/seva_v3_full_7b"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run all 3 stages + final evaluation
bash scripts/train_seva_3stage_full.sh
```

The script automatically:
- Detects its own repo directory (no hardcoded paths to edit)
- Detects completed stages (checks for `final/` directory), skips if already done
- Configures DeepSpeed ZeRO-3 for Stage 1/2
- Copies v3 reward function + patches GRPO config for Stage 3
- Runs final evaluation and saves results
- The `#SBATCH` headers at the top are for SLURM clusters; ignored when running directly with `bash`

### Option B: Run stages manually

#### Stage 1: Binary NLI

```bash
# 4xA100, DeepSpeed ZeRO-3 (auto-configured)
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

```bash
# Evaluate Stage 1
export DATA_DIR="$(pwd)/data/attribution"
python3 -u scripts/eval_seva.py \
    --model checkpoints/seva_v3_7b/stage1_nli/final \
    --benchmarks clearfacts fever \
    --output-dir results/seva_v3_full_7b_s1
```

**Expected results** (LoRA baseline reference; full FT should be higher):
- ClearFacts F1 >= 0.67 (LoRA=0.669)
- FEVER F1 >= 0.94 (LoRA=0.943)

#### Stage 2: Structured SFT

```bash
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

```bash
# Evaluate Stage 2
python3 -u scripts/eval_seva.py \
    --model checkpoints/seva_v3_7b/stage2_structured/final \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/seva_v3_full_7b_s2
```

**Expected results** (full FT should beat LoRA):
- ClearFacts F1 >= 0.69 (LoRA=0.674)
- FEVER F1 >= 0.92
- TruthfulQA F1 >= 0.70

#### Stage 3: GRPO

```bash
# Setup: copy v3 reward into verl directory + patch config
cp seva_reward_v3.py drzero/verl/custom_reward/seva_reward_v3.py
sed -i 's|verl/custom_reward/seva_reward.py|verl/custom_reward/seva_reward_v3.py|' drzero/config/seva_grpo.yaml
# (on macOS, use: sed -i '' 's|...|...|' ...)

export DATA_DIR="$(pwd)/data/attribution"

python -m verl.trainer.main_ppo \
    --config-name='seva_grpo' \
    data.train_files="data/attribution/seva_grpo_train.parquet" \
    data.val_files="data/attribution/seva_grpo_val.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=768 \
    data.max_response_length=512 \
    data.truncation=left \
    algorithm.kl_ctrl.kl_coef=0.02 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="checkpoints/seva_v3_7b/stage2_structured/final" \
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
```

```bash
# After GRPO finishes, find the latest checkpoint and evaluate
FINAL_MODEL=$(ls -d checkpoints/seva_v3_7b/stage3_grpo/global_step_* 2>/dev/null | sort -V | tail -1)

python3 -u scripts/eval_seva.py \
    --model "$FINAL_MODEL" \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/seva_v3_full_7b_final
```

**Expected results** (based on 3B GRPO gains):
- ClearFacts F1 >= 0.75 (3B: 0.649 -> 0.690, +4pts)
- TruthfulQA F1 >= 0.80 (3B: 0.721 -> 0.827, +10pts)
- Target: ClearFacts F1 >= **0.81**

---

## 8. Evaluation Details

### Usage

```bash
# MUST set data directory (otherwise uses hardcoded dev server path)
export DATA_DIR="$(pwd)/data/attribution"

python3 -u scripts/eval_seva.py \
    --model /path/to/checkpoint \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/my_eval
```

### Output files

```
results/my_eval/
|-- clearfacts_results.json    # Per-sample details + aggregate metrics
|-- fever_results.json
|-- truthfulqa_results.json
+-- summary.json               # Summary across all benchmarks
```

### Metrics

| Metric | Meaning | Priority |
|--------|---------|----------|
| `macro_f1` | Macro-averaged F1 (**primary metric for the paper**) | Highest |
| `accuracy` | Classification accuracy | High |
| `ece` | Expected Calibration Error (lower is better) | Medium |
| `avg_alignment_quality` | Evidence span localization quality | High |
| `avg_chain_quality` | Reasoning chain quality | High |
| `avg_groundedness` | Whether spans come from source text | High |
| `format_error_rate` | JSON format error rate (lower is better) | Medium |

---

## 9. Troubleshooting

### Common issues

| Problem | Solution |
|---------|----------|
| **Qwen2.5 rope_scaling "default" type** | vLLM 0.6.3 doesn't recognize it. Patch `vllm/model_executor/layers/rotary_embedding.py` to treat `"default"` as standard RoPE |
| **AutoModelForVision2Seq import error** | transformers 5.x removed this class. Add try/except in `verl/workers/fsdp_workers.py` |
| **OOM on single GPU** | Use `--optim-8bit` (Adafactor, ~31GB peak) + `--max-length 512` |
| **DeepSpeed CPU Adam compile failure** | Missing `python3-dev` headers. The script already sets `offload_optimizer.device="none"` automatically |
| **veRL import errors (torch 2.6+)** | Do not use veRL 0.7+. Pin `verl==0.3.0.post1` + `torch<2.6` |
| **vLLM port 8000 already in use** | `kill -9 $(lsof -t -i :8000)` |
| **eval_seva.py "file not found"** | Set `export DATA_DIR="$(pwd)/data/attribution"` |
| **GRPO NaN rewards** | Model is outputting non-JSON. Stage 2 may not have trained properly |

### Hardcoded paths to fix

`scripts/eval_seva.py` lines 44-45 have hardcoded dev server paths for DATA_DIR and RESULTS_DIR. **Always** override via environment variables:

```bash
export DATA_DIR="$(pwd)/data/attribution"
export RESULTS_DIR="$(pwd)/results"
```

### GRPO config reward path

The `train_seva_3stage_full.sh` script auto-patches `drzero/config/seva_grpo.yaml` to use v3 reward. If running Stage 3 manually, make sure to:
1. Copy the reward file: `cp seva_reward_v3.py drzero/verl/custom_reward/seva_reward_v3.py`
2. Patch the config: `sed -i 's|seva_reward.py|seva_reward_v3.py|' drzero/config/seva_grpo.yaml`

---

## 10. Results to Send Back

After completing all stages, please send the following:

1. **Stage 1 eval**: `results/seva_v3_full_7b_s1/summary.json`
2. **Stage 2 eval**: `results/seva_v3_full_7b_s2/summary.json`
3. **Stage 3 eval**: `results/seva_v3_full_7b_final/summary.json`
4. **Training logs**: stdout/stderr from each stage (loss curves)
5. **If ClearFacts F1 > 0.75**: save Stage 3 final checkpoint (needed for the paper)

### Success criteria

| Metric | Baseline (LoRA) | Expected (Full FT) | Target |
|--------|----------------|---------------------|--------|
| ClearFacts F1 | 0.674 | 0.75+ | **0.81+** |
| FEVER F1 | 0.921 | 0.94+ | 0.95+ |
| TruthfulQA F1 | 0.688 | 0.80+ | 0.83+ |

---

## Contact

Justin Yuan (USC) — ayuan@usc.edu
