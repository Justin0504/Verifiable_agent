# SEVA v3: 7B Full Fine-Tuning Experiment Guide

> **给协作者的说明**：这个文档是自包含的实验复现指南。你可以把整个 repo 克隆到有 4xA100 80GB 的机器上，按下面的步骤跑完全部实验。所有训练数据、评估数据、代码都在 repo 里，不需要额外下载。
>
> **For collaborators**: This is a self-contained experiment reproduction guide. Clone this repo to a machine with 4xA100 80GB GPUs and follow the steps below. All training data, eval data, and code are included.

---

## 1. What We Need to Run (待跑实验)

我们之前用 LoRA 跑的 7B 最好成绩是 ClearFacts F1=0.674。目标是 **81+ F1**（匹配 MiniCheck-7B）。

**核心假设**：LoRA 的参数容量不足以学好结构化输出 -> 改用全量微调。GRPO 在 3B 上证明有效（+4 F1）-> 在 7B 上应该更大提升。

### 需要跑的 3 个实验（按顺序）：

| 实验 | 说明 | 预计时间 | 产出 |
|------|------|----------|------|
| **Stage 1: Binary NLI** | 59K 样本，1 epoch，全量微调 | ~4h on 4xA100 | `checkpoints/seva_v3_7b/stage1_nli/final/` |
| **Stage 2: Structured SFT** | 5K 样本，3 epochs，全量微调 | ~1h on 4xA100 | `checkpoints/seva_v3_7b/stage2_structured/final/` |
| **Stage 3: GRPO** | 64K prompts，5 epochs，8-component reward | ~8-12h on 4xA100 | `checkpoints/seva_v3_7b/stage3_grpo/` |

每个 Stage 结束后需要跑一次 eval（见第 7 节）。

---

## 2. 已有结果（Baseline — 新实验需要超过这些）

### ClearFacts (N=1,590) — 主要指标

| Model | Acc | F1 | 训练方式 |
|-------|-----|-----|---------|
| MiniCheck-7B (目标) | ~81% | **0.810** | Full FT on NLI data |
| GPT-4o-mini (structured) | 69.9% | 0.698 | Commercial API |
| **SEVA-GRPO 3B** | 69.6% | **0.690** | Full FT + GRPO (证明 GRPO 有效) |
| SEVA-SFT 7B LoRA-128 (单阶段) | 68.6% | 0.686 | LoRA r=128 |
| SEVA-SFT 7B LoRA S2 (两阶段) | 67.5% | 0.674 | LoRA r=64, 两阶段 |
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

> 所有已有结果的 JSON 文件在 `results/` 目录下。

---

## 3. 环境配置

### 硬件要求

- **Stage 1 & 2 (SFT)**: 4xA100 80GB + DeepSpeed ZeRO-3
- **Stage 3 (GRPO)**: 4xA100 80GB, veRL 框架 (Ray + FSDP + vLLM)
- 单卡也能跑 Stage 1/2（用 `--optim-8bit`，峰值 ~31GB），但很慢

### 安装依赖

```bash
# 基础环境
pip install torch>=2.1 transformers>=4.40 accelerate datasets peft
pip install deepspeed  # Stage 1/2 多卡训练
pip install scikit-learn pandas numpy  # 评估用

# Stage 3 GRPO（必须用这个版本组合）
pip install vllm==0.6.3
pip install verl==0.3.0.post1
pip install flash-attn  # 可选但推荐

# 克隆 repo
git clone https://github.com/Justin0504/Verifiable_agent.git
cd Verifiable_agent
```

### 版本兼容性（重要）

| 组件 | 要求 | 原因 |
|------|------|------|
| veRL | **0.3.0.post1** | 0.7+ 有 breaking changes（DTensorSpec 需要 torch 2.6+） |
| vLLM | **0.6.3** | veRL 0.3 依赖此版本的 API |
| torch | >=2.1, <2.6 | veRL 0.3 不兼容 torch 2.6 |
| transformers | >=4.40 | Qwen2.5 支持 |

---

## 4. 数据说明

所有数据在 `data/attribution/`，已包含在 repo 中：

### 训练数据

| 文件 | 样本数 | Stage | 说明 |
|------|--------|-------|------|
| `sft_train_full.jsonl` | 59,500 | 1 | Binary NLI 数据。每行 `{"messages": [system, user, assistant]}`，assistant 只输出 label |
| `seva_sft_train.jsonl` | 4,992 | 2 | 结构化 SFT 数据。assistant 输出完整 JSON（含 evidence_alignment, reasoning_chain 等） |
| `seva_grpo_train.parquet` | 63,992 | 3 | GRPO 训练 prompts + ground truth |
| `seva_grpo_val.parquet` | 500 | 3 | GRPO 验证集 |

### 评估数据

| 文件 | 样本数 | 说明 |
|------|--------|------|
| `clearfacts.jsonl` | 1,590 | 主要 benchmark: claim-source attribution |
| `fever.jsonl` | 200 | 事实验证（3-class remapped to binary） |
| `truthfulqa.jsonl` | 400 | 真实性检测 |

### 数据格式

**SFT 数据**（`messages` 格式，Stage 1 和 2 通用）：
```json
{
  "messages": [
    {"role": "system", "content": "You are a fact verification expert..."},
    {"role": "user", "content": "Claim: James Davis received... Source: The 2016 Brisbane..."},
    {"role": "assistant", "content": "{\"label\": \"Attributable\", \"confidence\": 0.92, ...}"}
  ]
}
```
- Stage 1 的 assistant 只有 binary label
- Stage 2 的 assistant 有完整的 evidence_alignment + reasoning_chain + error_type

**GRPO 数据**（parquet 列）：
- `prompt`: JSON-encoded messages（system + user，不含 assistant）
- `reward_model`: JSON，含 `ground_truth`（`target` label, `claim`, `source`, `error_type`）
- `extra_info`: metadata（difficulty 分数，用于 curriculum learning）

---

## 5. 代码结构

```
Verifiable_agent/
|-- scripts/
|   |-- train_seva_sft.py              # [核心] SFT 训练脚本 (Stage 1 & 2)
|   |-- train_seva_3stage_full.sh      # [核心] 一键跑完 3 个 Stage
|   |-- eval_seva.py                   # [核心] 评估脚本
|   |-- generate_seva_sft_data.py      # 数据生成（GPT-4o teacher）
|   |-- generate_adversarial_probes.py # 对抗样本生成
|   +-- run_self_evolution.py          # 自进化循环
|
|-- src/
|   |-- verifier/
|   |   +-- seva_format.py             # [核心] 结构化输出 schema + system prompt
|   |-- training/
|   |   +-- curriculum.py              # Curriculum learning（可选）
|   |-- benchmarks/                    # Benchmark loaders
|   +-- llm/                           # LLM wrappers (OpenAI, vLLM)
|
|-- seva_reward_v3.py                  # [核心] 8-component GRPO reward function
|
|-- drzero/
|   |-- config/
|   |   +-- seva_grpo.yaml             # [核心] GRPO 训练 config (Hydra)
|   +-- verl/custom_reward/
|       |-- seva_reward.py             # v2 reward（eval 脚本依赖）
|       +-- seva_reward_v3.py          # v3 reward（GRPO 训练用，同根目录的副本）
|
|-- data/attribution/                  # 所有训练 + 评估数据
+-- results/                           # 已有实验结果 (JSON)
```

### 关键文件说明

#### `scripts/train_seva_sft.py` — SFT 训练（Stage 1 & 2 共用）
- 自动检测多卡（`WORLD_SIZE > 1`）并配置 DeepSpeed ZeRO-3
- 关键参数：`--base-model`, `--train-file`, `--epochs`, `--lr`, `--max-length`
- `--optim-8bit`: 切换 Adafactor optimizer（单卡时用，峰值 ~31GB vs AdamW ~47GB）
- `--resume-from`: 从 checkpoint 恢复训练
- Loss masking: 只在 assistant 部分计算 loss（system + user tokens 被 mask 为 -100）
- 训练完成后自动保存 model + tokenizer 到 `{output-dir}/final/`

#### `seva_reward_v3.py` — 8-component 过程奖励
- 入口函数：`compute_score(data_source, solution_str, ground_truth, extra_info)` -> float
- 批量入口：`compute_score_batch(...)` -> list[float]（含 boundary bonus）
- 返回奖励值范围 [0.0, ~1.65]
- 8 个组件权重随 epoch 动态调整（详见第 6 节）
- veRL 通过 `NaiveRewardManager` 调用此函数

#### `scripts/eval_seva.py` — 评估
- 加载模型，在 benchmarks 上逐样本生成 structured JSON + 打分
- 输出：accuracy, macro_F1, ECE, alignment_quality, chain_quality, groundedness
- 每个 benchmark 生成 `{name}_results.json`（含逐样本详情）+ `summary.json`
- **重要**: 默认 DATA_DIR 是 Yi Nian 路径，运行前必须设置 `export DATA_DIR="$(pwd)/data/attribution"`

#### `src/verifier/seva_format.py` — 结构化输出 schema
- 定义 `SEVA_SYSTEM_PROMPT`: 指导模型输出 6 个字段的 JSON
- 定义 `SEVA_USER_TEMPLATE`: `"Claim: {claim}\nSource: {source}"`
- 定义 `ERROR_TYPES`: 6 种错误类型 taxonomy
- `make_system_prompt_with_rules()`: 可注入 ReasoningBank rules

#### `drzero/config/seva_grpo.yaml` — GRPO config
- veRL 的 Hydra config，继承 `ppo_trainer` 默认配置
- 默认 reward path 指向 v2（运行时通过环境变量覆盖为 v3）
- rollout: temperature=1.2, top_p=0.95, n=8（每个 prompt 生成 8 个候选）

---

## 6. 训练逻辑详解

### Stage 1: Binary NLI Pretraining

**目标**：让模型学会 claim-source 二分类（Attributable / Not Attributable）

**训练逻辑**：
1. 加载 Qwen2.5-7B-Instruct 基座
2. 在 59K binary NLI 样本上做标准 causal LM SFT
3. 只训练 assistant 部分的 loss（system + user tokens 被 mask 为 -100）
4. DeepSpeed ZeRO-3 把模型参数分片到 4 张卡

**超参选择理由**：
- LR=1e-5: 标准 SFT learning rate，7B 模型适中
- 1 epoch: NLI 数据量大（59K），1 epoch 足够学会分类能力，多了容易过拟合
- max_length=1024: 大部分 claim-source pair 在 512 token 内，1024 留余量
- effective_batch=64: per_gpu=1 x grad_accum=16 x 4GPU

### Stage 2: Structured SFT

**目标**：在 Stage 1 学会分类的基础上，教模型输出完整的结构化 JSON

**训练逻辑**：
1. 加载 Stage 1 的 checkpoint 作为 base model
2. 在 5K 结构化样本上做 SFT
3. 结构化输出包含 6 个字段：evidence_alignment, reasoning_chain, label, confidence, error_type, fix_suggestion

**超参选择理由**：
- LR=3e-6（比 Stage 1 低 3x）: 避免 catastrophic forgetting Stage 1 的分类能力
- 3 epochs: 数据集小（5K），需要多轮学习结构化格式
- 其他参数与 Stage 1 相同

### Stage 3: GRPO with Process Reward

**目标**：用强化学习进一步提升结构化输出质量和分类准确率

**训练逻辑**：
1. 加载 Stage 2 checkpoint 作为 actor 和 reference model
2. 每个 prompt 用 vLLM 生成 8 个 rollout（temperature=1.2，鼓励多样性）
3. 用 8-component reward function 对每个 rollout 打分
4. GRPO advantage estimation: 每组 8 个样本内相互比较，计算相对优势
5. PPO loss 更新 actor 模型（reference model 固定不动）

**8-component reward function 详解**：

| 组件 | 权重 (epoch 1 -> 5) | 做什么 | 代码位置 |
|------|---------------------|--------|---------|
| R_format | 0.10 (固定) | 检查 JSON 合法性，必须含 label/confidence/evidence_alignment/reasoning_chain | `extract_json_from_response()` |
| R_accuracy | 0.80 -> 0.50 | label 是否正确。前期重准确率，后期降低让 grounding 发力 | `normalize_label()` |
| R_calibration | 0.10 -> 0.25 | confidence 与正确性对齐: 答对+高置信=正奖励，答错+高置信=负奖励 | 直接计算 |
| R_alignment | 0.15 -> 0.30 | evidence spans 是否真的出现在 claim/source 中。用 SequenceMatcher fuzzy match (threshold=0.6) | `_r_alignment()` |
| R_chain | 0.10 -> 0.25 | reasoning steps 质量: 有实质内容(>10 chars)、与 label 一致、2-4 步最优 | `_r_chain()` |
| R_coherence | 0.10 -> 0.20 | 跨组件一致性: label<->error_type, label<->alignment status, label<->chain judgment | `_r_coherence()` |
| R_diagnosis | 0.05 -> 0.15 | error_type 是否正确、fix_suggestion 是否具体（仅 Not Attributable 时生效） | `_r_diagnosis()` |
| R_specificity | 0.05 -> 0.10 | 惩罚模板化/通用输出，鼓励引用具体 claim 内容 | `_r_specificity()` |

**关键设计决策**：
- **动态权重**: 前期 accuracy 占 80% -> 后期 50%，让 grounding 组件逐步发力。原因是模型需要先学对分类，再提升推理质量
- **Boundary bonus**: 对 group 内正确率接近 50% 的 batch 给更高权重（信息量最大的样本）
- **所有 grounding 检查都用 fuzzy matching** (threshold=0.6): 容忍模型输出的 span 与原文有小差异

**框架**: veRL 0.3（基于 Ray + FSDP + vLLM）
- vLLM 做 rollout generation（快速推理采样）
- FSDP 做 actor 模型训练（参数分片到多卡）
- optimizer_offload=True: optimizer state 放 CPU，省 GPU 显存给 vLLM

---

## 7. 运行命令

### 方法一：一键脚本（推荐）

```bash
# 设置环境变量
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # 或本地已下载路径
export DATA_DIR="$(pwd)/data/attribution"
export CKPT_DIR="$(pwd)/checkpoints/seva_v3_full_7b"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 跑完 Stage 1 -> 2 -> 3 -> 评估
bash scripts/train_seva_3stage_full.sh
```

**使用前需修改 `train_seva_3stage_full.sh`**:
1. 第 22 行 `cd /scratch/bfsl/ayuan/Verifiable_agent` 改为你的 repo 路径
2. 第 23-25 行的默认路径改为你的环境（或用上面的环境变量覆盖）
3. 如果不是 SLURM 环境，忽略开头的 `#SBATCH` header

脚本会自动：
- 检测已完成的 Stage（检查 `final/` 目录是否存在），跳过不重跑
- Stage 1/2 用 `torchrun` + DeepSpeed ZeRO-3（自动配置）
- Stage 3 用 veRL GRPO
- 最后跑 eval 并保存结果

### 方法二：分步手动运行

#### Stage 1: Binary NLI

```bash
# 4xA100, DeepSpeed ZeRO-3 (自动配置)
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
# 评估 Stage 1
export DATA_DIR="$(pwd)/data/attribution"
python3 -u scripts/eval_seva.py \
    --model checkpoints/seva_v3_7b/stage1_nli/final \
    --benchmarks clearfacts fever \
    --output-dir results/seva_v3_full_7b_s1
```

**期望结果**（LoRA baseline 参考，全量微调应更高）：
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
# 评估 Stage 2
python3 -u scripts/eval_seva.py \
    --model checkpoints/seva_v3_7b/stage2_structured/final \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/seva_v3_full_7b_s2
```

**期望结果**（全量微调应比 LoRA 好）：
- ClearFacts F1 >= 0.69 (LoRA=0.674)
- FEVER F1 >= 0.92
- TruthfulQA F1 >= 0.70

#### Stage 3: GRPO

```bash
# 设置环境
export SEVA_REWARD_MODULE="seva_reward_v3"
export DATA_DIR="$(pwd)/data/attribution"

python -m verl.trainer.main_ppo \
    --config-name='seva_grpo' \
    data.train_files="data/attribution/seva_grpo_train.parquet" \
    data.val_files="data/attribution/seva_grpo_val.parquet" \
    data.train_batch_size=64 \
    data.max_prompt_length=768 \
    data.max_response_length=512 \
    data.truncation=left \
    algorithm.use_kl_in_reward=False \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
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
# GRPO 完成后，找到最新 checkpoint 并评估
FINAL_MODEL=$(ls -d checkpoints/seva_v3_7b/stage3_grpo/global_step_* 2>/dev/null | sort -V | tail -1)

python3 -u scripts/eval_seva.py \
    --model "$FINAL_MODEL" \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/seva_v3_full_7b_final
```

**期望结果**（基于 3B 的 GRPO 提升幅度）：
- ClearFacts F1 >= 0.75 (3B: 0.649 -> 0.690, +4pts)
- TruthfulQA F1 >= 0.80 (3B: 0.721 -> 0.827, +10pts)
- 目标 ClearFacts F1 >= **0.81**

---

## 8. eval_seva.py 使用说明

### 基本用法

```bash
# 必须设置数据目录（否则会用 Yi Nian 服务器的硬编码路径）
export DATA_DIR="$(pwd)/data/attribution"

python3 -u scripts/eval_seva.py \
    --model /path/to/checkpoint \
    --benchmarks clearfacts fever truthfulqa \
    --output-dir results/my_eval
```

### 输出文件

```
results/my_eval/
|-- clearfacts_results.json    # 逐样本详情 + 汇总指标
|-- fever_results.json
|-- truthfulqa_results.json
+-- summary.json               # 所有 benchmark 的汇总
```

### 指标说明

| 指标 | 含义 | 重要性 |
|------|------|--------|
| `macro_f1` | 宏平均 F1（**主要报告指标**） | 最重要 |
| `accuracy` | 分类准确率 | 重要 |
| `ece` | Expected Calibration Error（越低越好） | 次要 |
| `avg_alignment_quality` | evidence span 定位质量 | 重要 |
| `avg_chain_quality` | reasoning chain 质量 | 重要 |
| `avg_groundedness` | span 是否来自原文 | 重要 |
| `format_error_rate` | JSON 格式错误率（越低越好） | 次要 |

---

## 9. Troubleshooting

### 常见问题

| 问题 | 解决方案 |
|------|---------|
| **Qwen2.5 rope_scaling "default" type** | vLLM 0.6.3 不认识。找到 `vllm/model_executor/layers/rotary_embedding.py`，把 `"default"` 当作标准 RoPE 处理 |
| **AutoModelForVision2Seq import error** | transformers 5.x 删了这个 class。在 `verl/workers/fsdp_workers.py` 加 try/except |
| **OOM on single GPU** | 用 `--optim-8bit`（Adafactor，峰值 ~31GB）+ `--max-length 512` |
| **DeepSpeed CPU Adam 编译失败** | 缺 `python3-dev` 头文件。`train_seva_sft.py` 已自动把 `offload_optimizer.device` 设为 `"none"` |
| **veRL import errors (torch 2.6+)** | 不要用 veRL 0.7+，固定 `verl==0.3.0.post1` + `torch<2.6` |
| **vLLM port 8000 already in use** | `kill -9 $(lsof -t -i :8000)` |
| **eval_seva.py "file not found"** | 设置 `export DATA_DIR="$(pwd)/data/attribution"` |
| **GRPO NaN rewards** | 检查模型是否在输出非 JSON，可能是 Stage 2 没训好 |

### eval_seva.py 路径问题

脚本第 44-45 行的 DATA_DIR/RESULTS_DIR 默认值是 Yi Nian 服务器路径。在其他机器上**必须**通过环境变量覆盖：

```bash
export DATA_DIR="$(pwd)/data/attribution"
export RESULTS_DIR="$(pwd)/results"
```

### GRPO config 路径

`drzero/config/seva_grpo.yaml` 第 16 行的 `custom_reward_function.path` 指向 `verl/custom_reward/seva_reward.py`（v2）。Stage 3 命令通过 `SEVA_REWARD_MODULE=seva_reward_v3` 环境变量覆盖为 v3。如果环境变量不生效，手动修改 yaml:

```yaml
custom_reward_function:
  path: verl/custom_reward/seva_reward_v3.py  # 改这行
  name: compute_score
```

---

## 10. 结果提交

跑完后请把以下文件发给我（ayuan@usc.edu）：

1. **Stage 1 eval**: `results/seva_v3_full_7b_s1/summary.json`
2. **Stage 2 eval**: `results/seva_v3_full_7b_s2/summary.json`
3. **Stage 3 eval**: `results/seva_v3_full_7b_final/summary.json`
4. **训练 log**: 每个 Stage 的 stdout/stderr（训练 loss 曲线）
5. **如果 ClearFacts F1 > 0.75**: 保存 Stage 3 最终 checkpoint（用于论文）

### 关键判断标准

| 指标 | 基线 (LoRA) | 预期 (Full FT) | 目标 |
|------|------------|----------------|------|
| ClearFacts F1 | 0.674 | 0.75+ | **0.81+** |
| FEVER F1 | 0.921 | 0.94+ | 0.95+ |
| TruthfulQA F1 | 0.688 | 0.80+ | 0.83+ |

---

## Contact

Justin Yuan (USC) -- ayuan@usc.edu
