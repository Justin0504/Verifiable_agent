# SEVA: Self-Evolving Verification Agents for Reliable Fact Attribution
## NeurIPS 2026 — Paper Outline v2

---

## Novelty vs Distinction: 明确分开

### Novelty (我们独有的，没人做过)

**N1. Self-Evolving Training Paradigm for Fact Verifiers**
现有fact verifier训练都是one-shot：收集数据→训练→部署。
SEVA是第一个跨epoch自进化的训练范式：每个epoch从失败中学习新规则、
生成针对性数据、在decision boundary上精确强化。

**N2. Adversarial Probe Generation for Verification**
没有任何prior work用adversarial probing来训练fact verifier。
SEVA的Proposer根据Verifier的weakness profile主动生成
最难判断的(claim, document)对——4类歧义(contextual/linguistic/
knowledge-level/numerical)的对抗性样本。

**N3. Cumulative Verification Knowledge (ReasoningBank)**
没有任何fact verification工作有动态的知识积累机制。
ReasoningBank从验证轨迹中蒸馏规则、追踪效果、自动淘汰，
形成可解释的验证知识库。每条规则的贡献可量化追溯。

**N4. Structured Verification Reward with Boundary Optimization**
将数学推理领域的boundary-optimal reward (Dr.Zero) 和
process reward (Tango) 的思想首次引入fact attribution。
多组件reward分解验证质量的每个维度。

### Distinction (我们和别人的区别，不是novelty但必须讲清)

**D1. vs ClearCheck (VtV, COLM 2025)**
- 他们: 清洗数据 + synthetic data + one-shot SFT (8B)
- 我们: 自进化RL + adversarial probing + knowledge accumulation (3B)
- 关系: 他们发现问题(verifier不可靠)，我们提供更好的解法
- 公平对比: 我们用完全相同的SFT起点(ANLI + synthetic multi-hop)

**D2. vs MARCH (2026.03)**
- 他们: 训Generator让它不幻觉 (Checker是工具)
- 我们: 训Verifier让它判断更准 (Verifier是产品)
- 他们: single-run co-evolution, shared policy, PPO
- 我们: cross-epoch self-evolution, independent models, GRPO
- 本质区别: 不同的产品目标 (Generator vs Verifier)

**D3. vs RL Tango (NeurIPS 2025)**
- 他们: 数学推理验证, Generator和Verifier联合训练
- 我们: 事实归因验证, 独立训练standalone Verifier
- 他们: 隐式co-evolution (训练动态驱动)
- 我们: 显式self-evolution (Proposer + ReasoningBank + validation gate)
- 借鉴: 引用Tango证明RL训verifier > SFT (但我们在不同domain)

**D4. vs KnowRL (2025.06)**
- 他们: 验证QA推理链中的atomic facts (against Wikipedia)
- 我们: 验证text是否可归因于source document
- 他们: 单次GRPO, 静态knowledge base
- 我们: 跨epoch self-evolution, 动态ReasoningBank

**D5. vs TruthRL (2025.09)**
- 他们: 训练模型说"I don't know" (abstention)
- 我们: 训练模型准确判断attribution
- 任务完全不同

**D6. vs MiniCheck, AlignScore, etc. (existing fact verifiers)**
- 他们: one-shot SFT or NLI-based
- 我们: self-evolving RL
- 他们: 固定训练数据
- 我们: 动态生成的adversarial data

---

## Title

**SEVA: Self-Evolving Verification Agents for Reliable Fact Attribution**

Subtitle (workshop version): *Training Small Verifiers to Discover and Fix
Their Own Blind Spots through Adversarial Reinforcement Learning*

---

## Abstract (~250 words)

Fact attribution verifiers — models that judge whether a claim is supported
by a source document — are critical infrastructure for hallucination
detection, RAG quality control, and factuality-based alignment. Yet recent
work reveals that existing verifiers suffer from systematic errors, with
~16% of benchmark annotations being ambiguous or incorrect (Seo et al.,
2025). Current approaches address this through data cleaning and one-shot
supervised fine-tuning, but cannot adapt to novel error patterns or
accumulate verification expertise over time.

We introduce SEVA, a self-evolving framework that trains fact attribution
verifiers through iterative adversarial reinforcement learning. SEVA
operates via a four-phase loop: (1) PROBE — an adversarial Proposer
generates challenging (claim, document) pairs targeting the verifier's
current weaknesses; (2) REFLECT — failures are analyzed and distilled
into reusable verification rules in a ReasoningBank; (3) REFINE — GRPO
training with structured reward reinforces the verifier at its decision
boundary; (4) VERIFY — a multi-benchmark validation gate prevents
catastrophic forgetting.

Crucially, SEVA's improvement is itself verifiable: each learned rule,
each weakness addressed, and each capability gain is traceable through
the ReasoningBank's effectiveness scores and the structured reward
decomposition.

On ClearFacts and 14 fact attribution benchmarks, a 3B-parameter SEVA
verifier outperforms ClearCheck (8B, SFT) and approaches few-shot GPT-4o
performance, while providing an interpretable audit trail of its
verification capabilities. We release our code, trained models, and
the complete evolution logs.

---

## §1 Introduction (1.5 pages)

### Opening (3 sentences)
Fact attribution verification — determining whether a statement is
supported by a given source — underpins the reliability of modern NLP
systems. It serves as the backbone of RAG quality control, hallucination
detection, and factuality-based model alignment. Yet a systematic audit
of 14 verification benchmarks reveals that existing verifiers and their
evaluation data are unreliable: ~16% of benchmark labels are ambiguous
or incorrect, and state-of-the-art small verifiers (7-8B) still lag
significantly behind frontier LLMs.

### The Gap (1 paragraph)
Recent work addresses this through two complementary strategies. On the
data side, Seo et al. (2025) construct cleaner benchmarks (ClearFacts)
and augment training with synthetic multi-hop examples. On the model
side, Zha et al. (2025) demonstrate that RL-trained verifiers outperform
SFT verifiers in mathematical reasoning. However, both approaches share
a fundamental limitation: they treat verifier training as a **one-shot
process**. The verifier cannot discover its own blind spots, generate
targeted training data, or accumulate verification expertise over time.

### Our Approach (1 paragraph)
We introduce SEVA (Self-Evolving Verification Agents), a framework that
trains fact attribution verifiers through iterative adversarial RL. SEVA
implements a four-phase self-evolution loop:
- **PROBE**: An adversarial Proposer generates (claim, document) pairs
  targeting the verifier's current weaknesses
- **REFLECT**: Failures are analyzed and verification rules are distilled
  into a ReasoningBank
- **REFINE**: GRPO training with structured reward reinforces the verifier
  at its decision boundary
- **VERIFY**: A multi-benchmark validation gate prevents catastrophic
  forgetting

### Key Insight (1 paragraph)
SEVA embodies a dual notion of "verifiable": the model verifies facts
(its task), and its improvement process is itself verifiable (its property).
Through the ReasoningBank's rule tracking, structured reward decomposition,
and per-benchmark performance profiling, we can trace exactly what the
model learned, why it improved, and where it still fails — a transparency
that one-shot training methods fundamentally cannot provide.

### Results Preview (2-3 sentences)
A 3B-parameter SEVA verifier, starting from the same SFT checkpoint as
ClearCheck, outperforms the 8B ClearCheck model on ClearFacts (+X.X F1)
and achieves competitive performance with few-shot GPT-4o across 14
diverse benchmarks. Ablations confirm that each component of the
self-evolution loop contributes measurably, with adversarial probing
and ReasoningBank providing the largest individual gains.

### Contributions (numbered list)
1. **SEVA Framework**: The first self-evolving training paradigm for
   fact attribution verifiers, implementing a PROBE-REFLECT-REFINE-VERIFY
   loop with adversarial data generation and cumulative knowledge
   accumulation. (N1)
2. **ReasoningBank**: A dynamic, effectiveness-tracked verification
   knowledge base that distills reusable rules from verification
   trajectories, providing interpretable audit trails. (N3)
3. **Structured Verification Reward**: A multi-component reward design
   with boundary-optimal weighting, adapted from mathematical reasoning
   to fact attribution for the first time. (N4)
4. **Comprehensive Evaluation**: On ClearFacts and 14 benchmarks, SEVA-3B
   outperforms ClearCheck-8B, demonstrating that self-evolving RL
   training can compensate for 2.5x fewer parameters.

---

## §2 Related Work (1 page)

### 2.1 Fact Attribution Verification
- Task definition: given (claim, source) → attributable or not
- MiniCheck (Tang et al., 2024): Bespoke-7B, grounding-focused
- AlignScore (Zha et al., 2023): unified alignment scoring
- ClearCheck (Seo et al., 2025): SFT on ANLI + synthetic multi-hop
- LLM-AggreFact (Tang et al., 2024): leaderboard aggregating 11 benchmarks
- **Gap**: All use one-shot SFT or NLI transfer; none use RL or self-evolution

### 2.2 RL for Verification and Reasoning
- GRPO (Shao et al., 2024): group-relative policy optimization
- RL Tango (Zha et al., 2025): joint generator-verifier RL for math → 
  proves RL > SFT for verifier training, but limited to math
- MARCH (Li et al., 2026): multi-agent RL for RAG generation quality →
  trains Generator (not standalone Verifier)
- KnowRL (Ren et al., 2025): factuality reward for QA reasoning chains
- TruthRL (Wei et al., 2025): ternary reward for truthful QA
- **Gap**: No RL-based training for standalone fact attribution verifiers

### 2.3 Self-Evolving AI Systems
- Dr.Zero (Chen et al., 2025): proposer-solver co-evolution with
  boundary-optimal reward → math domain
- SkillRL (Ji et al., 2025): hierarchical skill bank → coding domain
- Self-Taught Evaluators (Wang et al., 2024): iterative SFT for judges
  → no RL, no adversarial probing, no knowledge accumulation
- Meta-Rewarding (Yuan et al., 2024): LLM meta-judges own judgments
  → SFT-based, no systematic evolution
- **SEVA combines**: self-evolution (Dr.Zero) + knowledge bank (SkillRL) +
  RL training (Tango) + fact attribution (ClearCheck) — a novel synthesis

### Positioning Table

| | ClearCheck | Tango | MARCH | Dr.Zero | **SEVA** |
|---|---|---|---|---|---|
| Task | Fact attrib. | Math | RAG gen. | Math | **Fact attrib.** |
| Product | Verifier | Gen.+Ver. | Generator | Solver | **Verifier** |
| Training | SFT | RL (GRPO) | RL (PPO) | RL (GRPO) | **RL (GRPO)** |
| Self-evolution | ✗ | ✗ | ✗ (single-run) | ✓ (implicit) | **✓ (explicit)** |
| Knowledge bank | ✗ | ✗ | ✗ | ✗ | **✓ ReasoningBank** |
| Adversarial data | ✗ | ✗ | ��� | ✓ (boundary) | **✓ (weakness-targeted)** |
| Verifiable improve. | ��� | ✗ | ✗ | ✗ | **✓ (rule tracking)** |

---

## §3 Method: SEVA (3 pages)

### 3.1 Problem Formulation

**Task**: Given a claim c and source document d, predict whether c is
attributable to d:
  f(c, d) → {Attributable, Not Attributable}

**Training objective**: Learn a verifier policy π_θ that maximizes
structured verification reward through iterative self-evolution.

**Self-evolution objective**: Over E epochs, simultaneously:
  (a) Improve verifier accuracy on a fixed multi-benchmark validation set
  (b) Accumulate reusable verification knowledge in ReasoningBank
  (c) Adapt training data distribution to the verifier's evolving boundary

### 3.2 SEVA Overview: The Four-Phase Loop

**Figure 1** (full-page architecture diagram):

```
Epoch e:
  ┌─────────┐                ┌─────────┐
  │  PROBE  │───────────────▶│ REFLECT │
  │         │  adversarial   │         │
  │ Generate│  (claim, doc)  │ Rollouts│
  │ probes  │  pairs         │ + reward│
  │ at      │                │ + fail  │
  │ boundary│                │ analysis│
  └────▲────┘                └────┬────┘
       │                          │
       │    ┌─────────┐           │
       │    │ VERIFY  │           │
       └────│         │◀──────────┘
            │ Validate│     ┌─────────┐
            │ + gate  │◀────│ REFINE  │
            │ + evolve│     │         │
            └─────────┘     │ GRPO    │
                            │ training│
                            └─────────┘
```

### 3.3 Phase 1: PROBE — Adversarial Probe Generation

**Proposer** (GPT-4o) generates (claim, document) pairs designed to
challenge the verifier's current weaknesses.

**Inputs to Proposer**:
1. Weakness profile W_e: per-benchmark accuracy, per-ambiguity-type error
   rates, confusion patterns from epoch e-1
2. Ambiguity taxonomy (from VtV): contextual, linguistic, knowledge-level,
   numerical
3. Difficulty curriculum: epoch 1 has 70% clear-cut samples; epoch 5 has
   90% boundary samples
4. Domain diversity: sample across benchmark domains (summarization, QA,
   scientific, Wikipedia, dialogue, financial, multi-hop)

**Probe types** (6 categories):
- **Subtle non-attribution**: claim almost matches source but differs in
  one critical detail (number, date, entity, negation)
- **Misleading evidence**: source contains related but non-supporting info
- **Multi-hop required**: attribution requires chaining 2-3 facts from source
- **Ambiguity-targeted**: exploit the 4 ambiguity types from VtV taxonomy
- **Safe attribution**: clearly supported claims (prevent over-rejection)
- **Weakness-targeted**: specifically target verifier's worst benchmark domain

**Quality filter**: LLM judge rejects probes without clear ground truth
(~30% rejection rate)

**Batch composition**: 80% new probes + 20% replay buffer (hard examples
from previous epochs)

### 3.4 Phase 2: REFLECT — Analysis and Rule Distillation

**Step 1: Rollouts and Reward**
For each probe, generate G=5 rollouts from the current verifier.
Each rollout produces:
```json
{
  "label": "Attributable",
  "confidence": 0.83,
  "reasoning": "The source explicitly states...",
  "rules_cited": ["R3", "R7"]
}
```

**Step 2: Structured Verification Reward**
```
R_total = R_structured × (α + β × R_boundary)

R_structured = R_format       (0.1: valid JSON)
             + R_attribution  (0-1.0: correct label)
             + R_calibration  (±0.2: confidence alignment)
             + R_reasoning    (0-0.1: explanation quality)
             + R_rule_cite    (0.05: cited relevant rules)

R_boundary = 1 - |mean(correct_in_group) - 0.5| × 2
```

R_boundary ensures maximum learning signal at the verifier's decision
boundary (2-3 of 5 rollouts correct), automatically downweighting
samples that are too easy or too hard.

**Step 3: Failure Analysis**
Extract informative failures:
- False positives: claimed attributable but actually not
- False negatives: claimed not attributable but actually is
- Ambiguity confusion: errors on contextual/linguistic/knowledge/numerical cases
- Domain-specific failures: errors concentrated in specific benchmark domains

**Step 4: Rule Distillation → ReasoningBank**
From failure patterns, distill new verification rules:
```
Rule R12: "Numerical Precision Mismatch"
Content: "When a claim contains a specific number (percentage, count, date),
         verify exact match with source. Approximate expressions ('about 30%')
         are attributable if source says '29-31%', but '40%' when source
         says '30%' is NOT attributable."
Source: failure_distillation (epoch 2)
Effectiveness: 0.76 (updated each epoch)
```

Update existing rules' effectiveness scores based on citation tracking.
Auto-link related rules. Evict rules with <20% effectiveness after 3+ uses.

### 3.5 Phase 3: REFINE — GRPO Training

Standard GRPO (Shao et al., 2024) with SEVA-specific adaptations:

- **Rule-conditioned generation**: Top-K ReasoningBank rules injected
  into verifier prompt. Model instructed to cite applicable rules.
- **Group size**: G=5 (balance variance reduction vs compute)
- **KL penalty**: λ=0.01 against SFT reference (prevent catastrophic drift)
- **Anti-forgetting**: 20% replay buffer + 10% easy anchors in each batch
- **Steps per epoch**: ~35 (single pass over data, prevent overfitting)

### 3.6 Phase 4: VERIFY — Validation Gate

After GRPO training, evaluate on fixed multi-benchmark validation set:
- ClearFacts_val (200 samples)
- LLM-AggreFact_val (subset: 100 per sub-benchmark)
- CoverBench_val (100 samples)
- HoVer_val (100 samples)

**Gate conditions**:
- No individual benchmark regresses > 5% from best checkpoint
- Overall macro F1 improves or is stable
- If gate fails → revert to previous checkpoint, adjust Proposer strategy

**Evolution outputs** (feed back to Phase 1):
1. Updated weakness profile W_{e+1}
2. Updated ReasoningBank (new rules, evicted rules, updated effectiveness)
3. Hard examples → replay buffer
4. Proposer strategy adjustment (from Evolver)

### 3.7 Training Pipeline Summary

```
Phase 0: SFT Foundation
  Data: ANLI (57K) + synthetic multi-hop (25K)  [same as ClearCheck]
  Model: Qwen2.5-3B-Instruct
  Config: 2 epochs, lr=2e-5
  → SFT checkpoint (= fair comparison baseline against ClearCheck)

Phase 1-5: Self-Evolution
  Per epoch: PROBE → REFLECT → REFINE → VERIFY
  Total: 5 outer epochs × ~35 inner GRPO steps = ~175 gradient updates
  Compute: ~30h on 2× RTX 6000 Ada (48GB)
  → SEVA checkpoint + ReasoningBank + evolution logs
```

---

## §4 Experiments (3 pages)

### 4.1 Setup

**Benchmarks** (3 groups):

| Group | Benchmarks | Purpose |
|-------|-----------|---------|
| Primary | ClearFacts (1,590) | Clean, corrected evaluation (VtV's contribution) |
| Breadth | LLM-AggreFact 11 subsets (~30K) | Cross-domain generalization |
| Depth | SciFact, HoVer, CoverBench | Scientific, multi-hop, long-context |

**Baselines**:

| Baseline | Type | Size | Training |
|----------|------|------|----------|
| MiniCheck-Bespoke | SFT verifier | 7B | NLI transfer |
| ClearCheck | SFT verifier | 8B | ANLI + synthetic |
| AlignScore | Unified scorer | 355M | Multi-task SFT |
| Few-shot GPT-4o | Frontier LLM | - | In-context |
| Few-shot o1 | Frontier LLM | - | In-context |
| SEVA-SFT (ours, no RL) | SFT verifier | 3B | ANLI + synthetic |
| **SEVA (ours, full)** | **Self-evolved** | **3B** | **SFT + self-evolving RL** |

**Key comparison**: SEVA-3B vs ClearCheck-8B with identical SFT data
(ANLI + synthetic multi-hop). The ONLY difference is training paradigm:
one-shot SFT vs self-evolving RL.

### 4.2 Main Results

**Table 1: ClearFacts Results**

| Model | Size | Macro F1 | Acc | Bal. Acc |
|-------|------|----------|-----|----------|
| AlignScore | 355M | - | - | - |
| MiniCheck | 7B | ~82 | - | - |
| ClearCheck | 8B | ~84 | - | - |
| SEVA-SFT | 3B | ~? | - | - |
| **SEVA** | **3B** | **>85?** | - | - |
| Few-shot GPT-4o | - | ~86 | - | - |
| Few-shot o1 | - | ~88.7 | - | - |

**Headline**: SEVA-3B > ClearCheck-8B (2.5x fewer params, better paradigm)

**Table 2: LLM-AggreFact + Extended Benchmarks (14+3 = 17 benchmarks)**

Per-benchmark breakdown showing SEVA's cross-domain generalization.
Key columns: AggreFact-CNN, AggreFact-XSum, RAGTruth, SciFact, HoVer,
CoverBench, ...

### 4.3 Ablation Study

**Table 3: Component Ablation (on ClearFacts)**

| Config | Macro F1 | Δ | Tests |
|--------|----------|---|-------|
| Full SEVA | **X.X** | — | — |
| − ReasoningBank | ? | ? | N3: knowledge accumulation |
| − Adversarial Proposer (random probes) | ? | ? | N2: targeted probing |
| − Boundary Reward (uniform weight) | ? | ? | N4: difficulty-aware |
| − Replay Buffer | ? | ? | Anti-forgetting |
| − Self-Evolution (one-shot GRPO) | ? | ? | N1: iterative evolution |
| − GRPO (SFT only = ClearCheck equivalent) | ? | ? | RL vs SFT |

**Key comparisons**:
- Full SEVA vs "SFT only": proves RL paradigm matters
- Full SEVA vs "one-shot GRPO": proves self-evolution matters (not just RL)
- Full SEVA vs "− ReasoningBank": proves knowledge accumulation matters

### 4.4 Self-Evolution Analysis

**Figure 2: Evolution Trajectory (4-panel)**
- (a) Per-benchmark accuracy across 5 epochs
- (b) ReasoningBank size and avg effectiveness across epochs
- (c) Weakness profile evolution (which ambiguity types improve when)
- (d) Proposer adaptation (probe difficulty distribution per epoch)

**Table 4: ReasoningBank Case Study (top-5 most effective rules)**

| Rule | Content (abbreviated) | Effectiveness | Epoch | Citations |
|------|----------------------|---------------|-------|-----------|
| R3 | Numerical precision mismatch... | 0.82 | 1 | 127 |
| R7 | Negation flip detection... | 0.78 | 1 | 98 |
| R12 | Temporal qualifier ("around", "about")... | 0.71 | 2 | 64 |
| R19 | Multi-hop chain verification... | 0.68 | 3 | 43 |
| R25 | Source scope limitation... | 0.65 | 4 | 31 |

This table is a KEY differentiator — no other method can produce this.

### 4.5 GrayFacts: Handling Ambiguity

**Table 5: Performance on ambiguous samples (GrayFacts, 159 samples)**

| Model | Avg Confidence (correct) | Avg Confidence (ambiguous) | ECE |
|-------|-------------------------|---------------------------|-----|
| MiniCheck | high | also high (bad) | high |
| ClearCheck | high | medium | medium |
| **SEVA** | **high** | **low (good calibration)** | **low** |

SEVA learns to express uncertainty on genuinely ambiguous samples
through the calibration reward component. ClearCheck can't — it was
trained on data that excluded ambiguous samples.

### 4.6 Efficiency Analysis

**Table 6: Cost and Speed**

| Model | Params | Inference/sample | $/1K samples |
|-------|--------|-----------------|-------------|
| Few-shot GPT-4o | - | ~2s | ~$50 |
| MiniCheck | 7B | ~0.3s | ~$0.50 |
| ClearCheck | 8B | ~0.4s | ~$0.60 |
| **SEVA** | **3B** | **~0.15s** | **~$0.20** |

SEVA is the most efficient AND the most accurate small verifier.

---

## §5 Analysis & Discussion (0.5 pages)

### 5.1 Why Does Self-Evolution Work?
- **Adaptive data**: Proposer targets the boundary, not the easy cases
- **Cumulative knowledge**: ReasoningBank transfers insights across epochs
- **Focused training**: Boundary-optimal reward allocates gradient to
  maximally informative samples
- Analogy: self-evolution is to one-shot training as spaced repetition
  is to cramming

### 5.2 When Does SEVA Fail?
- Long documents (>4K tokens): 3B model's context window limits
- Highly domain-specific claims requiring expert knowledge
- Adversarial attacks specifically designed to fool the ReasoningBank

### 5.3 Broader Impact
- SEVA-trained verifiers can serve as factuality reward models for RLHF
- ReasoningBank rules are human-readable → auditable verification
- Framework is task-agnostic: can be applied to other verification
  domains (safety, code, instruction following)

---

## §6 Conclusion (0.5 pages)

SEVA demonstrates that fact attribution verifiers benefit significantly
from self-evolving RL training over conventional one-shot SFT. Through
adversarial probing, cumulative knowledge accumulation, and structured
reward at the decision boundary, a 3B-parameter model outperforms
8B SFT baselines and approaches frontier LLM performance. The
ReasoningBank provides an interpretable audit trail — making SEVA's
improvement not just measurable but verifiable.

---

## Appendix Plan

| Section | Content |
|---------|---------|
| A | Full hyperparameter table (SFT + GRPO + self-evolution) |
| B | All 17 benchmark results (full table) |
| C | ReasoningBank: complete rule list (top-20) |
| D | Proposer prompt templates |
| E | Reward function implementation details |
| F | Training curves (loss, reward, grad norm, response length) |
| G | Additional ablations (group size, KL penalty, replay ratio) |
| H | Failure case studies (5 representative errors) |
| I | Computational cost breakdown |

---

## Figures Plan

| # | Content | Type | Priority |
|---|---------|------|----------|
| 1 | SEVA architecture (4-phase loop) | Diagram | P0 |
| 2 | Self-evolution trajectory (4-panel) | Line charts | P0 |
| 3 | Main results comparison | Bar chart | P0 |
| 4 | Ablation results | Bar chart | P0 |
| 5 | ReasoningBank growth + effectiveness | Line chart | P1 |
| 6 | Proposer adaptation across epochs | Stacked bar | P1 |
| 7 | Calibration reliability diagram | Scatter | P1 |
| 8 | Radar chart (17 benchmarks) | Radar | P2 |

---

## Experiment Execution Plan

### Phase 1: Data Preparation (Week 1)
- [ ] Download ANLI, LLM-AggreFact, ClearFacts, GrayFacts, SciFact, HoVer, CoverBench
- [ ] Unify format: {claim, source, label} for all datasets
- [ ] Split: train/val/test (ensure no leakage)
- [ ] Prepare SFT dataset (ANLI + synthetic multi-hop, same as ClearCheck)

### Phase 2: SFT Baseline (Week 1)
- [ ] SFT Qwen2.5-3B on ANLI + synthetic (ClearCheck-equivalent)
- [ ] Evaluate on ClearFacts + all benchmarks → SFT baseline numbers
- [ ] This is our "SEVA-SFT" row AND the starting point for self-evolution

### Phase 3: Self-Evolution Training (Week 2-3)
- [ ] Implement train_seva.py (orchestrates 4-phase loop)
- [ ] Run 5 outer epochs of self-evolution
- [ ] Log: ReasoningBank rules, weakness profiles, per-epoch metrics
- [ ] Evaluate final checkpoint on all benchmarks

### Phase 4: Ablations (Week 3)
- [ ] 6 ablation configs (see Table 3)
- [ ] Each ~6h GPU → total ~36h

### Phase 5: Analysis & Writing (Week 4)
- [ ] Generate all figures and tables
- [ ] Write paper
- [ ] Internal review and revision

### Total GPU Budget
- SFT: ~1h
- Self-evolution (5 epochs × ~6h): ~30h
- Ablations (6 configs × ~6h): ~36h
- Evaluation: ~5h
- **Total: ~72h on 2× RTX 6000 Ada**
