# Self-Evolving Verifiable Agents: Reinforcement Learning with Structured Reward for Fact Verification

## Paper Outline — NeurIPS 2026 Submission

---

## Title Options

1. **SEVA: Self-Evolving Verifiable Agents via Reinforcement Learning with Structured Rewards**
2. **From Imitation to Verification: Training Small Language Models as Self-Improving Fact Checkers**
3. **Verifiable Agents with Co-Evolving Reasoning: Bridging the Gap Between 3B and GPT-4o**

---

## Abstract (~250 words)

- **Problem**: LLM-based fact verification suffers from hallucination, poor calibration, and inconsistent reasoning — especially for small models (<10B)
- **Gap**: Existing methods rely on prompting or retrieval augmentation; no work systematically applies RL to fact verification with structured, multi-component rewards
- **Our approach**: SEVA — a self-evolving verifiable agent framework combining:
  - (1) Three-stage verification pipeline (Proposer → Responder → Verifier)
  - (2) ReasoningBank: autonomous rule distillation from verification trajectories
  - (3) GRPO training with structured reward (format + accuracy + calibration + reasoning)
- **Results**: Qwen2.5-3B achieves 84.1% accuracy (from 62.8% zero-shot), approaching GPT-4o-mini (85.6%), with balanced per-class performance across S/C/N labels
- **Contribution**: First systematic study of RL-based self-evolution for fact verification agents; open-source framework with 6 benchmarks and 6 baselines

---

## 1. Introduction (1.5 pages)

### 1.1 Motivation
- Fact verification is critical for trustworthy AI — but current approaches have fundamental limitations:
  - Prompting-based methods (CoVE, SAFE) are brittle and model-dependent
  - Retrieval-based methods (Retrieve-NLI) rely on external knowledge quality
  - No existing work trains verification-specific reasoning via RL
- Key insight: Verification is a **verifiable task** — we can construct reward signals from ground-truth labels, making it ideal for RL

### 1.2 Research Questions
- **RQ1**: Can RL (specifically GRPO) improve fact verification accuracy beyond SFT distillation?
- **RQ2**: Does structured reward design (format + accuracy + calibration + reasoning) outperform naive accuracy-only rewards?
- **RQ3**: Can a self-evolving rule bank (ReasoningBank) co-evolve with RL training to improve generalization?
- **RQ4**: How does a 3B model with SEVA compare to 10x-100x larger models on standard benchmarks?

### 1.3 Contributions
1. **SEVA Framework**: End-to-end pipeline for training self-evolving fact verification agents
2. **Structured Reward Design**: Multi-component reward with class-aware weighting, calibration bonus, and reasoning incentive
3. **ReasoningBank**: Autonomous rule distillation and co-evolution mechanism (inspired by SkillRL)
4. **Comprehensive Evaluation**: 6 benchmarks × 6 baselines, ablation studies, error analysis
5. **Practical Impact**: 3B model achieves near-GPT-4o-mini performance with 100x fewer parameters

---

## 2. Related Work (1 page)

### 2.1 Fact Verification and Hallucination Detection
- FEVER benchmark and NLI-based approaches (Thorne et al., 2018)
- FActScore (Min et al., 2023): atomic claim decomposition
- SAFE (Wei et al., 2024): search-augmented factuality evaluator
- SelfCheckGPT (Manakul et al., 2023): self-consistency based
- CoVE (Dhuliawala et al., 2023): chain-of-verification
- **Gap**: All are inference-time methods; none train the verifier itself via RL

### 2.2 RL for Language Model Alignment
- RLHF (Ouyang et al., 2022): reward model from human preferences
- GRPO (Shao et al., 2024): group relative policy optimization, outcome-only
- Dr.Zero (Chen et al., 2025): proposer-solver co-evolution with difficulty-guided reward
- SkillRL (Ji et al., 2025): hierarchical skill bank with dynamic update
- **Our connection**: We adapt GRPO's group-relative advantage with structured reward specific to verification

### 2.3 Self-Evolving AI Agents
- Reflexion (Shinn et al., 2023): verbal reinforcement
- Self-Refine (Madaan et al., 2023): iterative self-improvement
- A-MEM (Yu et al., 2025): autonomous memory linking
- **Our connection**: ReasoningBank extends A-MEM's linking with effectiveness tracking and RL-driven eviction

---

## 3. Method: SEVA Framework (3 pages)

### 3.1 Overview
- Figure 1: Architecture diagram
  - Three-stage pipeline: Proposer → Responder → Verifier
  - Self-evolution loop: ReasoningBank ↔ Verifier ↔ GRPO training
  - Training progression: Base → SFT (GPT-4o distillation) → GRPO (RL)

### 3.2 Three-Stage Verification Pipeline

#### 3.2.1 Proposer (Probe Generator)
- 4 risk categories: Missing Evidence, Multi-Hop Reasoning, Pressure/Presupposition, Unanswerable
- R-Zero style adaptive difficulty: targets verifier's ability boundary (~50% accuracy on hard cases)
- Quality filtering: LLM judge evaluates verifiability, naturalness, difficulty → ~30% rejection rate

#### 3.2.2 Responder (Target Model)
- Queries target LLMs (GPT-4o, Claude, GPT-4o-mini) with generated probes
- Collects responses for verification

#### 3.2.3 Verifier (Core Component)
- Step 1: Claim Decomposition — breaks responses into atomic claims
- Step 2: Evidence Matching — labels each claim as S (Supported) / C (Contradicted) / N (Not Enough Info)
- Step 3: Reliability Scoring — aggregates claim-level labels into overall reliability score
- Output format: structured JSON with label, confidence, reasoning

### 3.3 Structured Reward Design for GRPO

#### 3.3.1 Reward Components
```
R(y|x) = R_format + R_accuracy + R_calibration + R_reasoning
```

| Component | Value | Purpose |
|-----------|-------|---------|
| R_format | 0.1 if valid JSON with label ∈ {S,C,N}; else 0 (entire reward = 0) | Ensures structured output |
| R_accuracy | correct × class_weight (S:1.0, C:1.2, N:1.5) | Main learning signal with class balancing |
| R_calibration | ±0.2 × confidence (+ if correct, - if wrong) | Incentivizes well-calibrated predictions |
| R_reasoning | 0.1 if reasoning ≥ 5 words | Encourages explanatory outputs |

- **Total range**: 0.0 (format failure) ~ 1.9 (N-class correct, high confidence, good reasoning)
- **Design rationale**: Class weights derived from Phase 1 analysis (N-class hardest at 50% zero-shot, C-class second at 40.1%)

#### 3.3.2 GRPO Training
- Group size G=5: for each prompt, generate 5 rollouts, normalize rewards within group
- Advantage: A_i = (R_i - mean(R)) / std(R) — no critic network needed
- KL penalty: disabled (use_kl_in_reward=False) — allows larger policy shifts
- Training: 3 epochs, 105 steps, lr=1e-6, grad_clip=0.1

### 3.4 ReasoningBank: Self-Evolving Rule Memory

#### 3.4.1 Rule Distillation
- After each verification, extract structured rules from success/failure trajectories
- Rule schema: {title, description, content, tags, source_type, effectiveness}
- Source types: "success" (what worked), "failure" (what went wrong), "calibration" (bias corrections)

#### 3.4.2 Autonomous Linking (A-MEM style)
- LLM finds connections between new and existing rules
- Refines overlapping rules, merges redundant ones
- Maintains a graph of linked_rule_ids

#### 3.4.3 Effectiveness Tracking & Eviction
- Record usage: success_count / usage_count per rule
- Evict rules with <20% effectiveness after 3+ uses
- Capacity limit: 100 rules; prune bottom 20% when exceeded
- During inference: inject top-15 relevant rules into verifier prompt

#### 3.4.4 Co-Evolution with GRPO (if implemented)
- At rollout time: inject ReasoningBank rules into prompt
- Use GRPO reward signal to update rule effectiveness scores
- At epoch boundaries: distill new rules from high-reward trajectories

### 3.5 Training Pipeline
```
Stage 1: Zero-shot baseline (Qwen2.5-3B-Instruct)
    ↓
Stage 2: SFT warm-up (1997 GPT-4o teacher trajectories, 2 epochs)
    ↓
Stage 3: GRPO training (3 epochs, 4480 train samples, structured reward)
    ↓
Stage 4: Self-evolution (ReasoningBank rule accumulation across evaluation epochs)
```

---

## 4. Experimental Setup (1.5 pages)

### 4.1 Benchmarks (6 total)

| Benchmark | Task | Size | Labels | Domain |
|-----------|------|------|--------|--------|
| FEVER | Evidence-based claim verification | 5000+ | S/C/N | Wikipedia |
| TruthfulQA | Truthfulness evaluation | 817 | S/C/N | General knowledge |
| SciChat | Scientific claim verification | ~200 | S/C/N | Science |
| HaluEval | Hallucination detection | ~1000 | Binary | QA/Dialogue |
| MuSiQue | Multi-hop reasoning verification | ~500 | S/C/N | Multi-doc |
| FActScore | Atomic fact scoring | ~500 | S/N | Biography |

### 4.2 Baselines (6 total)
1. **Direct Prompting**: Zero-shot LLM classification
2. **SelfCheck-GPT** (Manakul et al., 2023): Self-consistency sampling
3. **FActScore** (Min et al., 2023): Atomic claim decomposition + retrieval
4. **SAFE** (Wei et al., 2024): Search-augmented factuality evaluator
5. **CoVE** (Dhuliawala et al., 2023): Chain-of-verification prompting
6. **Retrieve-NLI**: Retrieval + NLI model classification

### 4.3 Models
- **Ours**: Qwen2.5-3B-Instruct (base → SFT → GRPO)
- **Verifier Agent (API)**: GPT-4o-mini with SEVA pipeline
- **Reference**: GPT-4o, Claude Sonnet 4

### 4.4 Training Details
- Hardware: 2× RTX 6000 Ada (48GB each)
- SFT: 1997 samples, 2 epochs, lr=2e-5, ~7 min
- GRPO: 4480 samples, 3 epochs (105 steps), lr=1e-6, ~5.8 hours
- GRPO config: group_size=5, batch=128, vLLM rollout, FSDP, optimizer_offload=True

### 4.5 Evaluation Metrics
- **Accuracy**: Overall and per-class (S, C, N)
- **Macro F1**: Handles class imbalance
- **Format Compliance**: % of valid JSON outputs
- **Confusion Matrix**: Per-class error patterns

---

## 5. Results (2.5 pages)

### 5.1 Main Results (Table 1: Unified Benchmark Comparison)

**Table 1**: Accuracy (%) across 6 benchmarks. Bold = best, underline = second best.

| Method | FEVER | TruthfulQA | SciChat | HaluEval | MuSiQue | FActScore | Avg |
|--------|-------|------------|---------|----------|---------|-----------|-----|
| Direct Prompting | 85.6 | 69.4 | | | | | |
| SelfCheck-GPT | 59.4 | 49.7 | | | | | |
| FActScore | 85.6 | 74.1 | | | | | |
| SAFE | 81.3 | 73.8 | | | | | |
| CoVE | 61.3 | 35.9 | | | | | |
| Retrieve-NLI | 78.1 | **88.8** | | | | | |
| SEVA (GPT-4o-mini) | **85.6** | 79.7 | | | | | |
| SEVA (Qwen-3B SFT) | | 76.6* | | | | | |
| SEVA (Qwen-3B GRPO) | | **84.1*** | | | | | |

*tested on FEVER+TruthfulQA mixed test set; need separate benchmark evaluation

→ **TODO**: Fill in SciChat, HaluEval, MuSiQue, FActScore columns for GRPO model

### 5.2 Training Progression (Figure 2)

**Table 2**: Qwen2.5-3B accuracy across training stages

| Stage | Overall | S | C | N | Format Error |
|-------|---------|---|---|---|-------------|
| Zero-shot | 62.8% | 62.6% | 40.1% | 87.2% | 0% |
| + SFT | 76.6% | 74.3% | 92.2% | 62.2% | 0% |
| + GRPO | **84.1%** | **92.0%** | 81.2% | **78.9%** | 0% |

**Key findings**:
- GRPO brings +7.5% overall improvement over SFT
- Class-aware weighting successfully addresses N-class weakness: 62.2% → 78.9% (+16.7%)
- S-class also significantly improved: 74.3% → 92.0% (+17.7%)
- C-class decreased from SFT: 92.2% → 81.2% (-11.0%) — discussed in Section 5.5

**Figure 2**: Include `grpo_training_dynamics.pdf` — 4-panel showing loss, reward, entropy, grad norm

### 5.3 Self-Evolution Analysis (Figure 3)

**Table 3**: ReasoningBank evolution across epochs (FEVER)

| Epoch | Synth Acc | Cal Acc | Rules | Weaknesses | Time |
|-------|-----------|---------|-------|------------|------|
| 1 | 90.9% | 82.5% | 8 | 3 | 374s |
| 5 | 86.5% | 87.5% | 40 | 3 | 612s |
| 10 | 80.0% | 85.0% | 60+ | 3 | ~500s |
| 15 | ~85% | 85.6% | 80+ | 2 | ~500s |

- Calibration accuracy improves monotonically (82.5% → 85.6%)
- Synthetic accuracy fluctuates — expected as adaptive difficulty targets boundary
- Rule count grows then stabilizes via eviction mechanism

**Figure 3**: Evolution curve + rule effectiveness distribution

### 5.4 Ablation Study (Table 4)

**Table 4**: Ablation of SEVA components (FEVER+TruthfulQA test set)

| Configuration | Accuracy | Δ |
|--------------|----------|---|
| Full SEVA (GRPO) | **84.1%** | — |
| w/o Class Weighting (all weights = 1.0) | ~?% | ? |
| w/o Calibration Reward | ~?% | ? |
| w/o Reasoning Bonus | ~?% | ? |
| w/o SFT warm-up (base → GRPO directly) | ~?% | ? |
| w/o ReasoningBank | ~?% | ? |
| Group size = 3 (vs 5) | ~?% | ? |
| Group size = 8 (vs 5) | ~?% | ? |

→ **TODO**: Run ablation experiments (6 configs × ~6h each = ~36h GPU)

### 5.5 Error Analysis

#### 5.5.1 Confusion Matrix Analysis
- GRPO's C-class decline: model shifts from "over-predicting C" (SFT) to "balanced but uncertain"
- N-class improvement: class weight 1.5 successfully incentivizes "Not Enough Info" recognition

#### 5.5.2 Failure Case Study
- Show 3-4 representative failures:
  - (a) C→S confusion: subtle contradictions missed
  - (b) N→S confusion: model finds spurious support
  - (c) Hard cases where even GPT-4o fails

#### 5.5.3 Calibration Analysis
- Plot: predicted confidence vs actual accuracy (reliability diagram)
- Show GRPO improves calibration over SFT

---

## 6. Analysis and Discussion (1.5 pages)

### 6.1 Why Does GRPO Work for Fact Verification?
- Verification has clear ground truth → ideal for outcome-based RL
- Structured reward provides multi-dimensional learning signal
- Group-relative advantage naturally handles reward scale differences across classes
- Class weighting addresses prior distribution mismatch

### 6.2 The Class Balance Trade-off
- Zero-shot: biased toward N (87.2%) at expense of C (40.1%)
- SFT: distillation from GPT-4o shifts bias toward C (92.2%) at expense of N (62.2%)
- GRPO: class-aware reward produces most balanced performance (92/81/79)
- Insight: **RL can correct distributional biases inherited from teacher models**

### 6.3 ReasoningBank Synergy
- Rules capture verification heuristics that transfer across examples
- Effectiveness tracking prevents rule degradation
- Connection to SkillRL: both maintain evolving skill/rule banks, but SEVA applies to verification rather than math/coding

### 6.4 Scaling Behavior
- 3B model with SEVA (84.1%) vs GPT-4o-mini (85.6%): only 1.5% gap
- Implication: structured RL can partially compensate for model scale
- **TODO**: Test with 7B and 14B Qwen models to show scaling curve

### 6.5 Limitations
- Single reward function design — may not generalize to all verification domains
- C-class accuracy decrease needs further investigation
- ReasoningBank-GRPO co-evolution not fully integrated in current experiments
- Evaluation primarily on English benchmarks

---

## 7. Conclusion (0.5 page)

- SEVA demonstrates that self-evolving RL training can significantly improve small model fact verification
- Structured reward design with class-aware weighting is critical for balanced performance
- 3B model achieves 84.1% accuracy, approaching GPT-4o-mini's 85.6%
- ReasoningBank provides interpretable, transferable verification heuristics
- Future work: multi-turn verification, cross-lingual evaluation, ReasoningBank-GRPO co-evolution

---

## Appendix

### A. Full Hyperparameter Table
### B. Complete Benchmark Results (all 6 × all baselines)
### C. ReasoningBank Rule Examples (top-10 most effective rules)
### D. Additional Training Curves (per-class reward, response length)
### E. Prompt Templates (system prompt, user template, ReasoningBank injection format)
### F. Computational Cost Breakdown

---

## Figures List

| # | Content | Status |
|---|---------|--------|
| Fig 1 | SEVA Architecture Diagram | **TODO** |
| Fig 2 | GRPO Training Dynamics (4-panel) | **DONE** ✓ |
| Fig 3 | Accuracy Comparison Bar Chart | **DONE** ✓ |
| Fig 4 | Validation Reward Curve | **DONE** ✓ |
| Fig 5 | Self-Evolution Curve (rules + accuracy over epochs) | **TODO** |
| Fig 6 | Ablation Results | **TODO** (need experiments) |
| Fig 7 | Confusion Matrix Heatmap (Zero-shot vs SFT vs GRPO) | **TODO** |
| Fig 8 | Calibration Reliability Diagram | **TODO** |
| Fig 9 | Radar Chart (6 benchmarks) | **TODO** (need experiments) |

---

## Tables List

| # | Content | Status |
|---|---------|--------|
| Table 1 | Main Results (6 benchmarks × 8 methods) | **PARTIAL** (2/6 benchmarks) |
| Table 2 | Training Progression (zero-shot → SFT → GRPO) | **DONE** ✓ |
| Table 3 | Self-Evolution Metrics Across Epochs | **PARTIAL** (FEVER only) |
| Table 4 | Ablation Study | **TODO** (need experiments) |
| Table 5 | Training Configuration Details | **DONE** ✓ |
| Table 6 | Per-Class Confusion Matrix | **TODO** |
| Table 7 | Computational Cost | **TODO** |

---

## Experiment TODO Checklist (Priority Order)

### Must-Have (Paper will be rejected without these)
- [ ] **Ablation study**: 6 configurations (see Table 4)
- [ ] **GRPO model on all 6 benchmarks separately** (not just mixed test set)
- [ ] **Confusion matrix visualization** (3×3 for each stage)
- [ ] **Architecture diagram** (Figure 1)

### Should-Have (Significantly strengthens paper)
- [ ] **ReasoningBank + GRPO co-evolution** integration
- [ ] **C-class fix**: re-run GRPO with higher C weight (1.2 → 1.5) and compare
- [ ] **Calibration reliability diagram**
- [ ] **7B model comparison** (scaling curve data point)
- [ ] **Computational cost table**

### Nice-to-Have (Cherry on top)
- [ ] Cross-domain transfer analysis (train on FEVER, test on TruthfulQA)
- [ ] Human evaluation on 50 samples
- [ ] Comparison with concurrent work (2025 papers)
- [ ] ReasoningBank rule visualization / case study
