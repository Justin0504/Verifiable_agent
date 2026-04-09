# SEVA: Self-Evolving Verifiable Agent
## Paper Storytelling & Architecture Design

---

## I. Core Thesis (一句话)

> **A verification agent that discovers its own blind spots, distills reusable
> verification strategies from failures, and reinforces itself at its ability
> boundary — producing improvement that is itself verifiable and traceable.**

---

## II. Storytelling Arc

### Opening Hook

"How do expert fact-checkers get better?"

Not by memorizing more facts — but through a cycle:
1. They encounter claims that **challenge** their current knowledge
2. They **fail** on some, analyze why, and extract **reusable heuristics**
3. They **practice** on similar cases until the pattern is internalized
4. They **expand** to new domains, carrying accumulated expertise

No existing RL method for factuality models this cycle. KnowRL, TruthRL,
and MARCH all treat training as a one-shot process: fixed data → fixed
reward → train → deploy. The model improves, but it doesn't *learn how
to learn*.

### The Two Pillars

**SEVA is built on two complementary principles:**

#### Pillar 1: Self-Evolving (自进化)

The agent **actively discovers** what it doesn't know and **generates its
own curriculum** to fix it. This is not just "train on more data" — it's
a closed loop where:

- The Proposer **adapts** to the Verifier's current weaknesses
- Failures are **analyzed** and converted to actionable strategies
- Training focuses on the **decision boundary** where learning happens
- Knowledge **accumulates** across epochs rather than being discarded

**Distinction from related work:**
- KnowRL/TruthRL: One-shot RL, no evolution, no adversarial probing
- MARCH: Co-evolution within a single training run, no cross-epoch adaptation
- Dr.Zero: Boundary-optimal but no knowledge accumulation or rule distillation
- SEVA: **Cross-epoch evolution with cumulative knowledge and adversarial targeting**

#### Pillar 2: Verifiable (可验证)

The agent's improvement process is **transparent and auditable**. At any
point we can answer:
- **What** did the model learn? → ReasoningBank rules with effectiveness scores
- **Why** did it improve? → Rule citation tracking in model outputs
- **Where** does it still fail? → Per-domain weakness profiles
- **How much** did each component help? → Decomposed structured reward

**Distinction from related work:**
- KnowRL: Factuality reward is a black-box NLI score
- TruthRL: Ternary reward gives no insight into *why* the model abstains
- MARCH: Zero-tolerance reward is binary, no decomposition
- SEVA: **Multi-component structured reward + traceable rule citations + interpretable evolution trajectory**

### The Narrative Arc (for paper Sections 1-2)

```
§1 Introduction:
   Hook    → "LLMs hallucinate. Can we build agents that verify AND self-improve?"
   Gap     → Existing RL methods are one-shot, single-domain, opaque
   Thesis  → SEVA: self-evolving + verifiable, two properties that reinforce each other
   Preview → 3B model matches GPT-4o-mini on 6 benchmarks, with traceable improvement

§2 Related Work:
   [Map SEVA into the landscape — show what's missing in each line of work]
   
   Line 1: RL for Factuality (KnowRL, TruthRL)
     → One-shot training, no evolution, limited reward structure
   
   Line 2: Multi-Agent Verification (MARCH)
     → Co-evolution but within single run, no knowledge accumulation
   
   Line 3: Self-Play / Co-Evolution (Dr.Zero, SkillRL, SPIN)
     → Boundary-optimal training but no fact-verification-specific design
   
   Line 4: Reasoning Rule Banks (SkillRL, A-MEM)
     → Skill/memory banks but not integrated with RL training loop
   
   ★ SEVA unifies: adversarial co-evolution (Line 3) + structured verification
     reward (Line 1) + cumulative knowledge bank (Line 4) + verifiable
     improvement trajectory (novel)
```

---

## III. Architecture: The SEVA Loop

### Design Philosophy

The loop has **4 phases** per epoch, named to tell the story:

```
    ╭──────────────────────────────────────────────────╮
    │           SEVA Self-Evolution Loop                │
    │                                                  │
    │   ┌─────────┐    ┌─────────┐    ┌─────────┐    │
    │   │ PROBE   │───▶│ REFLECT │───▶│ REFINE  │    │
    │   │         │    │         │    │         │    │
    │   │ Generate│    │ Analyze │    │ RL      │    │
    │   │ boundary│    │ failures│    │ training│    │
    │   │ probes  │    │ distill │    │ at      │    │
    │   │         │    │ rules   │    │ boundary│    │
    │   └────▲────┘    └─────────┘    └────┬────┘    │
    │        │                             │         │
    │        │         ┌─────────┐         │         │
    │        └─────────│ VERIFY  │◀────────┘         │
    │                  │         │                    │
    │                  │ Validate│                    │
    │                  │ + gate  │                    │
    │                  └─────────┘                    │
    │                                                  │
    ╰──────────────────────────────────────────────────╯
```

### Detailed Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        SEVA: Self-Evolving Verifiable Agent                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                             ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    PHASE 1: PROBE (数据生成)                        │    ║
║  │                                                                     │    ║
║  │  Adversarial Proposer (GPT-4o)                                     │    ║
║  │  ┌──────────────────────────────────────────────────────────┐      │    ║
║  │  │  Inputs:                                                  │      │    ║
║  │  │    ① Weakness Profile W_e (from epoch e-1)               │      │    ║
║  │  │       - Per-domain accuracy gaps                          │      │    ║
║  │  │       - Per-class confusion patterns (S↔C, C↔N, etc.)   │      │    ║
║  │  │       - Specific failure patterns from Evolver            │      │    ║
║  │  │    ② Domain Sampling (inverse-accuracy weighting)        │      │    ║
║  │  │       - scientific, biographical, news, common_knowledge │      │    ║
║  │  │       - multi_hop, hallucination, wikipedia              │      │    ║
║  │  │    ③ Difficulty Curriculum                                │      │    ║
║  │  │       - Epoch 1: 70% easy + 30% boundary                │      │    ║
║  │  │       - Epoch N: 10% anchor + 90% boundary              │      │    ║
║  │  │    ④ Risk Type Coverage                                  │      │    ║
║  │  │       - MISSING_EVIDENCE, MULTI_HOP                      │      │    ║
║  │  │       - PRESSURE_PRESUPPOSITION, UNANSWERABLE            │      │    ║
║  │  │                                                           │      │    ║
║  │  │  Output: 400-600 probes per epoch                        │      │    ║
║  │  │    → {claim, evidence, gold_label, domain, risk_type}    │      │    ║
║  │  │    → Quality filter (~30% rejection by LLM judge)        │      │    ║
║  │  └──────────────────────────────────────────────────────────┘      │    ║
║  │                                                                     │    ║
║  │  Replay Buffer (20% of batch):                                     │    ║
║  │    Hard examples from ALL previous epochs, stratified by domain    │    ║
║  │                                                                     │    ║
║  └──────────────────────────┬──────────────────────────────────────────┘    ║
║                             │                                               ║
║                             ▼                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    PHASE 2: REFLECT (失败分析)                      │    ║
║  │                                                                     │    ║
║  │  Step 1: Verifier Rollouts                                         │    ║
║  │    - G=5 rollouts per probe (GRPO group)                           │    ║
║  │    - Each rollout: claim + evidence + rules → {label, conf, why}  │    ║
║  │    - Rules injected from ReasoningBank (top-K by relevance)        │    ║
║  │                                                                     │    ║
║  │  Step 2: Structured Reward Decomposition                           │    ║
║  │    ┌────────────────────────────────────────────────────────┐      │    ║
║  │    │  R_total = R_structured × (α + β × R_boundary)         │      │    ║
║  │    │                                                        │      │    ║
║  │    │  R_structured = R_format    (valid JSON output)        │      │    ║
║  │    │               + R_accuracy  (correct label × weight)   │      │    ║
║  │    │               + R_calibrate (confidence alignment)     │      │    ║
║  │    │               + R_reasoning (explanation quality)      │      │    ║
║  │    │               + R_rule_cite (cited applicable rules)   │      │    ║
║  │    │                                                        │      │    ║
║  │    │  R_boundary = 1 - |mean(correct_in_group) - 0.5| × 2  │      │    ║
║  │    │  (maximum signal when 2-3 of 5 rollouts correct)       │      │    ║
║  │    └────────────────────────────────────────────────────────┘      │    ║
║  │                                                                     │    ║
║  │  Step 3: Failure Analysis                                          │    ║
║  │    - Extract informative failures (false_pos, false_neg, boundary) │    ║
║  │    - Identify confusion patterns: which labels confuse which?      │    ║
║  │    - Cluster failures by domain and risk type                      │    ║
║  │                                                                     │    ║
║  │  Step 4: Rule Distillation → ReasoningBank                        │    ║
║  │    - Success patterns → "when X, check for Y" rules               │    ║
║  │    - Failure patterns → "avoid Z" defensive rules                  │    ║
║  │    - Auto-link: LLM finds connections between related rules        │    ║
║  │    - Track: which rules were cited, which led to correct answers   │    ║
║  │    - Evict: rules with <20% effectiveness after 3+ uses           │    ║
║  │                                                                     │    ║
║  └──────────────────────────┬──────────────────────────────────────────┘    ║
║                             │                                               ║
║                             ▼                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    PHASE 3: REFINE (RL训练)                        │    ║
║  │                                                                     │    ║
║  │  GRPO Training (veRL framework)                                    │    ║
║  │  ┌──────────────────────────────────────────────────────────┐      │    ║
║  │  │  For each batch:                                          │      │    ║
║  │  │    1. Retrieve top-K rules from ReasoningBank             │      │    ║
║  │  │    2. Inject into system prompt:                          │      │    ║
║  │  │       "You MUST cite which rule(s) you applied."         │      │    ║
║  │  │       [R1] Negation flip: ...                            │      │    ║
║  │  │       [R2] Temporal mismatch: ...                        │      │    ║
║  │  │    3. Generate G=5 rollouts                               │      │    ║
║  │  │    4. Compute R_total (structured × boundary)            │      │    ║
║  │  │    5. GRPO advantage estimation + gradient update         │      │    ║
║  │  │    6. KL penalty (λ=0.01) against SFT reference          │      │    ║
║  │  │    7. Record rule citations from responses                │      │    ║
║  │  └──────────────────────────────────────────────────────────┘      │    ║
║  │                                                                     │    ║
║  │  Anti-Forgetting Mechanisms:                                       │    ║
║  │    - 20% replay buffer in each batch                               │    ║
║  │    - 10% easy anchor samples (curriculum)                          │    ║
║  │    - KL divergence from SFT reference model                        │    ║
║  │                                                                     │    ║
║  └──────────────────────────┬──────────────────────────────────────────┘    ║
║                             │                                               ║
║                             ▼                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    PHASE 4: VERIFY (验证门控)                       │    ║
║  │                                                                     │    ║
║  │  Multi-Domain Validation Gate                                      │    ║
║  │  ┌──────────────────────────────────────────────────────────┐      │    ║
║  │  │  Fixed validation set (never used in training):           │      │    ║
║  │  │    FEVER_val(100) + TruthfulQA_val(100)                  │      │    ║
║  │  │    + SciFact_val(50) + HaluEval_val(50)                  │      │    ║
║  │  │                                                           │      │    ║
║  │  │  Gate conditions:                                         │      │    ║
║  │  │    ✓ No domain regresses > 5% from best checkpoint       │      │    ║
║  │  │    ✓ Overall mean accuracy improves                       │      │    ║
║  │  │    ✗ If gate fails → revert to last good checkpoint      │      │    ║
║  │  └──────────────────────────────────────────────────────────┘      │    ║
║  │                                                                     │    ║
║  │  Evolution Outputs → feed back to Phase 1:                         │    ║
║  │    ① Updated weakness profile W_{e+1}                              │    ║
║  │    ② Updated ReasoningBank (new rules, evicted rules)             │    ║
║  │    ③ Hard examples → replay buffer                                 │    ║
║  │    ④ Proposer strategy update (from Evolver)                       │    ║
║  │                                                                     │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## IV. Novelty Claims — Why SEVA is Different

### Claim 1: Closed-Loop Self-Evolution (vs. Open-Loop RL)

```
                    KnowRL / TruthRL / MARCH:
                    ┌──────┐    ┌───────┐    ┌──────┐
                    │ Data │───▶│  RL   │───▶│Model │  (one-shot, no feedback)
                    └──────┘    └───────┘    └──────┘

                    SEVA:
                    ┌──────────────────────────────────────┐
                    │  ┌───────┐  ┌───────┐  ┌───────┐   │
                    │  │ Probe │─▶│Reflect│─▶│Refine │   │  (closed-loop,
                    │  └───▲───┘  └───────┘  └───┬───┘   │   each epoch
                    │      │      ┌───────┐      │       │   builds on
                    │      └──────│Verify │◀─────┘       │   the last)
                    │             └───────┘               │
                    └──────────────────────────────────────┘
```

**What makes SEVA's loop genuinely self-evolving (not just iterative):**
- The Proposer doesn't generate random data — it **targets the Verifier's
  current weaknesses** using the weakness profile
- The ReasoningBank **accumulates** verification knowledge — epoch 5's
  Verifier has access to rules distilled from epochs 1-4
- The difficulty curriculum **adapts** — early epochs have more easy anchors,
  later epochs focus on the boundary
- The replay buffer **prevents forgetting** — hard examples from all epochs
  are retained

### Claim 2: Verifiable Improvement Trajectory

Unlike black-box RL where we only see a loss curve, SEVA produces a
**traceable evolution log**:

```
Epoch 1:
  Rules distilled: 12 (8 from failures, 4 from successes)
  Top rule: [R3] "Negation flip" (effectiveness: 0.72, cited 45 times)
  Weakness: C-class accuracy 40.1% → confusion with N
  
Epoch 2:
  Rules distilled: 8 new, 3 evicted (low effectiveness)
  Top rule: [R3] still dominant, [R7] "Temporal qualifier" emerged
  C-class improved: 40.1% → 67.3%
  New weakness: scientific domain underperforming
  
Epoch 3:
  Rules distilled: 6 new, 5 evicted
  Scientific domain probes increased (inverse-accuracy weighting)
  SciFact accuracy: 39.5% → 68.2%
  ReasoningBank: 23 active rules, avg effectiveness 0.61
```

This is **not possible** with KnowRL (single NLI score), TruthRL (ternary
reward), or MARCH (binary zero-tolerance). Only SEVA's combination of
structured reward + rule tracking + per-domain profiling produces this
level of interpretability.

### Claim 3: Adversarial Boundary Probing (vs. Static/Random Data)

```
          KnowRL: NqOpen + WebQ + ComplexQ (fixed datasets)
          TruthRL: CRAG benchmark (fixed dataset)
          MARCH:  BioASQ + 2WikiMHQA + MuSiQue (fixed datasets)
          
          SEVA:   Proposer generates NEW data each epoch
                  → Targets current weakness profile
                  → Samples from underperforming domains
                  → Difficulty-aware (boundary optimal)
```

### Claim 4: Cumulative Knowledge via ReasoningBank

```
          KnowRL: Static Wikipedia (never updated)
          TruthRL: No knowledge store
          MARCH:  No knowledge store
          
          SEVA:   ReasoningBank grows and refines across epochs
                  → Rules distilled from success AND failure
                  → Autonomous linking (A-MEM style)
                  → Effectiveness tracking per rule
                  → Auto-eviction of low-quality rules
                  → Injected into prompt → model learns to USE rules
                  → Rule citations tracked → verifiable influence
```

---

## V. The "Verifiable" Story — Dual Meaning

The word "Verifiable" in SEVA carries **two meanings**, and this duality
is the conceptual hook of the paper:

### Meaning 1: The Agent Verifies Facts (任务层)

The agent's task is fact verification: given a claim and evidence,
determine if the claim is Supported, Contradicted, or has Not enough
information. This is the surface-level meaning.

### Meaning 2: The Agent's Evolution is Verifiable (元层)

The agent's **improvement process itself is transparent and auditable**:

| What is verifiable? | How? |
|---------------------|------|
| What the model learned | ReasoningBank rules with descriptions |
| Why it improved | Rule citation tracking in outputs |
| Where it still fails | Per-domain weakness profiles |
| How much each component helped | Decomposed reward (format/accuracy/calibration/reasoning) |
| Whether improvement is real | Multi-domain validation gate |
| What knowledge accumulated | ReasoningBank size, effectiveness distribution |

**This is the paper's conceptual contribution.** We argue that for
safety-critical applications like fact verification, it's not enough for
the model to be accurate — we need to **verify that the improvement process
itself is sound**. SEVA provides this through its combination of structured
rewards, rule tracking, and evolution logging.

### Paper Hook (§1, paragraph 1)

> "Can we build verification agents whose improvement is itself verifiable?
> Current RL approaches for factuality produce better models, but the path
> from 'before training' to 'after training' is a black box. We cannot
> answer basic questions: What did the model learn? Why does it now succeed
> where it previously failed? Will it maintain this improvement on unseen
> domains? SEVA addresses these questions through a self-evolving architecture
> where every component of improvement — from adversarial data generation
> to rule distillation to reward decomposition — is transparent, traceable,
> and auditable."

---

## VI. Comparison Table (for paper §2 or §5)

| Property | KnowRL | TruthRL | MARCH | **SEVA** |
|----------|--------|---------|-------|----------|
| RL Algorithm | GRPO | GRPO | PPO | GRPO |
| Task | QA reasoning | QA + abstention | RAG hallucination | **Fact verification (S/C/N)** |
| Evolution | None (one-shot) | None (one-shot) | Single-run co-evolve | **Cross-epoch self-evolution** |
| Data source | Fixed benchmarks | Fixed CRAG | Fixed BioASQ/MuSiQue | **Adversarial Proposer (adaptive)** |
| Knowledge store | Static Wikipedia | None | None | **ReasoningBank (dynamic)** |
| Reward structure | 3-component (format+correct+fact) | Ternary (+1/0/-1) | Binary (0/-1) | **5-component structured × boundary** |
| Improvement traceable? | No (NLI score) | No (ternary signal) | No (binary signal) | **Yes (rules + citations + profiles)** |
| Multi-domain | Single | 4 QA benchmarks | RAG only | **6 diverse benchmarks** |
| Anti-forgetting | None | None | None | **Replay buffer + validation gate + KL** |
| Proposer role | N/A | N/A | Decompose response (passive) | **Generate adversarial probes (active)** |
| Model size | 7B | 3B-32B | 8B | **3B (efficiency focus)** |

---

## VII. Ablation Design (proves each component matters)

To validate the architecture, we need ablations that isolate each novel
component. Each ablation removes ONE component:

| ID | Configuration | Tests contribution of |
|----|---------------|-----------------------|
| A0 | **Full SEVA** | (reference) |
| A1 | SEVA - ReasoningBank | Knowledge accumulation |
| A2 | SEVA - Adversarial Proposer (random probes) | Weakness-targeted probing |
| A3 | SEVA - Boundary Reward (uniform weight) | Difficulty-aware learning |
| A4 | SEVA - Replay Buffer | Anti-forgetting |
| A5 | SEVA - Validation Gate | Catastrophic specialization prevention |
| A6 | SEVA - Rule Citations | Verifiable reasoning |
| A7 | SEVA - Multi-Domain (single domain only) | Cross-domain generalization |
| A8 | SFT only (no self-evolution) | The entire self-evolution loop |
| A9 | One-shot GRPO (no evolution, current v1) | Cross-epoch evolution |

**Key comparisons:**
- A0 vs A9: "Does self-evolution help beyond one-shot GRPO?" (THE core claim)
- A0 vs A1: "Does ReasoningBank contribute beyond prompt engineering?"
- A0 vs A2: "Does adversarial targeting matter vs random data?"
- A0 vs A3: "Does boundary-optimal reward improve over uniform reward?"
- A9 vs A8: "Does GRPO help beyond SFT?" (already shown: 84.1 vs 76.6)

---

## VIII. Expected Results Narrative

### Quantitative Story

```
Zero-shot (3B)         → 62.8%  (baseline: LLM has some ability)
+ SFT (teacher dist.)  → 76.6%  (knowledge transfer: +13.8pp)
+ GRPO v1 (one-shot)   → 84.1%  (RL sharpening: +7.5pp, but catastrophic OOD)
+ SEVA (self-evolving)  → ???%   (self-evolution: +Xpp, WITH domain robustness)

Key narrative:
  GRPO v1 achieves 84.1% on in-distribution but COLLAPSES out-of-domain
  (SciFact 80→39.5%, FActScore 100→60.5%, avg 74.0→62.8%)
  
  SEVA maintains in-distribution performance AND recovers OOD:
  - SciFact: 39.5% → 70%+ (recovery through domain-diverse probing)
  - FActScore: 60.5% → 85%+ (recovery through replay + validation gate)
  - 6-bench average: 62.8% → 78%+ (robust generalization)
  
Bonus: 3B model competitive with GPT-4o-mini (85.6%) on verification,
       at 100x fewer parameters
```

### Qualitative Story (verifiable evolution)

```
Show: ReasoningBank growth over 5 epochs
  - Epoch 1: 12 rules, avg effectiveness 0.45
  - Epoch 5: 35 rules, avg effectiveness 0.68
  - Show: specific rules that emerged, e.g.:
    [R3] "Negation flip" — contributed to C-class improvement
    [R7] "Temporal qualifier" — caught date mismatches
    [R12] "Source authority" — scientific claim handling

Show: Weakness profile evolution
  - Epoch 1: C-class confusion (40.1%)
  - Epoch 3: C-class recovered (75%), scientific domain weak
  - Epoch 5: Balanced across all classes and domains

Show: Proposer adaptation
  - Epoch 1: uniform probing
  - Epoch 3: 40% scientific domain (targeting weakness)
  - Epoch 5: rebalanced as weakness resolved
```

---

## IX. Paper Structure (refined)

```
§1 Introduction (1.5 pages)
   - Hook: "verifiable improvement" dual meaning
   - Gap: one-shot RL, no evolution, opaque improvement
   - SEVA thesis: self-evolving + verifiable
   - Key results: 3B matches GPT-4o-mini, robust across 6 benchmarks
   - Contributions: (1) self-evolution loop, (2) ReasoningBank,
     (3) structured verifiable reward, (4) 6-benchmark evaluation

§2 Related Work (1 page)
   - RL for Factuality (KnowRL, TruthRL)
   - Multi-Agent Verification (MARCH)
   - Self-Play & Co-Evolution (Dr.Zero, SkillRL, SPIN)
   - Table: SEVA vs all (the comparison table above)

§3 Method: SEVA (3.5 pages)
   §3.1 Overview: The PROBE-REFLECT-REFINE-VERIFY Loop
   §3.2 Adversarial Proposer with Weakness-Targeted Probing
   §3.3 Structured Verifiable Reward Design
     - 5-component decomposition
     - Boundary-optimal weighting
     - Rule citation bonus
   §3.4 ReasoningBank: Cumulative Verification Knowledge
     - Rule distillation from trajectories
     - Effectiveness tracking and auto-eviction
     - A-MEM style autonomous linking
   §3.5 Anti-Forgetting: Replay Buffer + Validation Gate

§4 Experiments (2.5 pages)
   §4.1 Setup: model, benchmarks, baselines
   §4.2 Main Results: 6-benchmark comparison
   §4.3 Ablation Study: 10 configurations
   §4.4 Evolution Analysis: ReasoningBank growth, weakness profiles
   §4.5 Efficiency: 3B vs larger models

§5 Analysis (1 page)
   §5.1 What makes self-evolution work? (ablation insights)
   §5.2 Verifiable improvement: rule trajectory case study
   §5.3 When does SEVA fail? (limitations)

§6 Conclusion (0.5 page)
```

---

## X. Implementation Priority

### Must-have for NeurIPS (P0):

1. **`train_seva.py`**: Full self-evolution training loop
   - Integrates Proposer → GRPO → ReasoningBank → Validation Gate
   - 5 outer epochs, each with GRPO inner training
   - Logs evolution metrics (rules, weaknesses, domain accuracy)

2. **Multi-domain Proposer**: Extend current proposer.py
   - Add domain dimension (7 domains)
   - Inverse-accuracy weighting
   - Boundary-targeting from weakness profile

3. **ReasoningBank ↔ GRPO integration**: 
   - Rules injected into GRPO prompt template
   - Rule citations parsed from model outputs
   - Effectiveness updated after each epoch

4. **Validation gate**: 
   - Fixed multi-domain validation set
   - Checkpoint gating logic

5. **Replay buffer**:
   - Stratified hard example retention
   - Mixed into GRPO training batches

6. **Ablation sweep**: At minimum A0, A1, A2, A3, A9

### Nice-to-have (P1):

7. Boundary-optimal reward (R_boundary)
8. Rule citation reward bonus
9. Proposer curriculum (progressive difficulty)
10. Full 10-configuration ablation

---

## XI. Key Selling Points for Reviewers

1. **Practical**: 3B model achieves GPT-4o-mini level verification
   → deployable on single GPU, not just a research artifact

2. **Interpretable**: ReasoningBank provides auditable improvement path
   → safety-critical applications need this transparency

3. **Robust**: Multi-domain validation prevents catastrophic specialization
   → addresses a real failure mode we empirically observed

4. **Novel combination**: No prior work combines adversarial self-evolution
   + cumulative knowledge bank + structured verifiable reward for fact
   verification

5. **Reproducible**: Built on open-source stack (veRL + Qwen2.5 + vLLM)
   → easy for community to build on
