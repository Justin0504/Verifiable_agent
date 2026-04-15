# SEVA v2: Self-Evolving Verifiable Agent — Optimized Architecture

## Overview

```
                        ┌─────────────────────────────┐
                        │      SEVA Training Loop       │
                        └──────────────┬──────────────┘
                                       │
    ┌──────────────────────────────────┐│┌──────────────────────────────────┐
    │         OUTER LOOP               │││          INNER LOOP              │
    │   (Epoch-level, ~5 epochs)       │││   (Step-level, GRPO updates)     │
    │                                  │││                                  │
    │  Proposer → Data → Verifier     │││  Prompt + Rules → 5 rollouts    │
    │  Failures → Rules → Evolve      │││  → Reward → Gradient → Weights  │
    └──────────────────────────────────┘│└──────────────────────────────────┘
                                       │
```

## Detailed Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │                    OUTER LOOP (Epoch e)                          │  ║
║  │                                                                  │  ║
║  │  ┌────────────┐    ┌──────────────┐    ┌─────────────────────┐  │  ║
║  │  │  Proposer   │───▶│  Responder   │───▶│  Data Constructor   │  │  ║
║  │  │  (GPT-4o)   │    │  (Target LLM)│    │  → Parquet          │  │  ║
║  │  │             │    └──────────────┘    └──────────┬──────────┘  │  ║
║  │  │  Inputs:    │                                   │             │  ║
║  │  │  - weakness │                                   ▼             │  ║
║  │  │  - boundary │    ┌──────────────────────────────────────────┐ │  ║
║  │  │  - memory   │    │              INNER LOOP (GRPO)           │ │  ║
║  │  │  - domain   │    │                                          │ │  ║
║  │  │    diversity │    │  For each batch:                        │ │  ║
║  │  └──────▲──────┘    │   1. Retrieve top-K rules from RB       │ │  ║
║  │         │           │   2. Inject rules into system prompt     │ │  ║
║  │         │           │   3. Generate G=5 rollouts (Verifier)    │ │  ║
║  │         │           │   4. Compute structured reward           │ │  ║
║  │         │           │   5. Compute boundary bonus              │ │  ║
║  │         │           │   6. GRPO advantage + gradient update    │ │  ║
║  │         │           │   7. Update rule effectiveness scores    │ │  ║
║  │         │           │                                          │ │  ║
║  │         │           └──────────────────┬───────────────────────┘ │  ║
║  │         │                              │                         │  ║
║  │         │                              ▼                         │  ║
║  │  ┌──────┴───────────────────────────────────────────────────┐   │  ║
║  │  │                  EVOLUTION STEP                           │   │  ║
║  │  │                                                          │   │  ║
║  │  │  1. Evaluate on multi-domain validation set              │   │  ║
║  │  │  2. Extract informative failures                         │   │  ║
║  │  │  3. Distill new rules → ReasoningBank                   │   │  ║
║  │  │  4. Evict low-effectiveness rules                        │   │  ║
║  │  │  5. Update weakness profile for Proposer                 │   │  ║
║  │  │  6. Replay buffer: keep 20% hardest from this epoch      │   │  ║
║  │  │  7. Validation gate: only keep ckpt if val improves      │   │  ║
║  │  │                                                          │   │  ║
║  │  └──────────────────────────────────────────────────────────┘   │  ║
║  │                                                                  │  ║
║  └──────────────────────────────────────────────────────────────────┘  ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## 6 Key Improvements over v1

### 1. Difficulty-Guided Reward (from Dr.Zero)

**Problem**: Static reward treats easy and hard samples equally.

**Solution**: Add a "boundary bonus" based on within-group variance.

```
R_boundary(x) = 1 - |mean(correct_in_group) - 0.5| * 2

If all 5 rollouts correct  → boundary = 0 (too easy, no learning signal)
If all 5 rollouts wrong    → boundary = 0 (too hard, no learning signal)  
If 2-3 out of 5 correct    → boundary = 1.0 (maximum learning signal!)
```

Final reward:
```
R_total = R_structured × (α + β × R_boundary)
        = (format + accuracy + calibration + reasoning) × (0.5 + 0.5 × boundary)
```

This automatically downweights samples that are too easy or too hard, focusing
the gradient on the model's "ability boundary" — exactly where learning happens.

### 2. Domain-Diversified Proposer

**Problem**: Proposer generates probes only in 4 risk types, but all within 
similar knowledge domains → domain overfitting.

**Solution**: Add an explicit domain dimension to probe generation.

```python
DOMAINS = [
    "scientific_claims",      # SciFact-like
    "biographical_facts",     # FActScore-like  
    "current_events",         # News verification
    "common_knowledge",       # TruthfulQA-like
    "multi_hop_reasoning",    # MuSiQue-like
    "hallucination_detection",# HaluEval-like
    "wikipedia_claims",       # FEVER-like
]

# Each epoch: sample probes across ALL domains
# Domain weights adapt based on per-domain accuracy:
#   - Worse domains get higher sampling weight
#   - Prevents overfitting to easy domains
```

This ensures the Verifier sees diverse data every epoch, naturally preventing
the catastrophic forgetting we observed (SciFact 80→39.5%).

### 3. Rule-Conditioned Generation (SkillRL-style)

**Problem**: Rules injected into prompt may be ignored by the model.

**Solution**: Make rule usage explicit and measurable.

```
System prompt includes:
"Below are verification rules. You MUST cite which rule(s) you applied."

Rules:
[R1] "Negation flip": When evidence contains negation, check if claim...
[R2] "Temporal mismatch": Verify that dates in claim match evidence...

Expected output:
{"label": "C", "confidence": 0.9, "reasoning": "Applied [R1]: ...", "rules_used": ["R1"]}
```

Track which rules get cited → update effectiveness based on actual usage,
not just batch-level correlation.

Add a small reward bonus for citing applicable rules:
```
R_rule_citation = 0.05 if model cites relevant rules AND is correct
```

### 4. Experience Replay Buffer (Anti-Forgetting)

**Problem**: Each epoch only trains on new Proposer data → old knowledge forgotten.

**Solution**: Maintain a replay buffer of hard examples from all previous epochs.

```python
class ReplayBuffer:
    def __init__(self, max_size=500):
        self.buffer = []  # (sample, reward, epoch)
    
    def add_hard_examples(self, epoch_results):
        # Keep samples where model was wrong OR barely right (low confidence)
        hard = [r for r in epoch_results if not r.correct or r.confidence < 0.6]
        self.buffer.extend(hard)
        if len(self.buffer) > self.max_size:
            # Keep diverse mix: stratify by domain + risk_type
            self.buffer = stratified_sample(self.buffer, self.max_size)
    
    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))
```

Each GRPO epoch trains on:
- 80% new Proposer-generated data (current boundary)
- 20% replay buffer (hard examples from past epochs)

### 5. Multi-Domain Validation Gate

**Problem**: No mechanism to prevent catastrophic specialization.

**Solution**: Validate on a fixed multi-domain set after each epoch. Only keep
checkpoint if it improves (or doesn't significantly regress) on ALL domains.

```python
VALIDATION_SET = {
    "fever_val": 100 samples,
    "truthfulqa_val": 100 samples,  
    "scifact_val": 50 samples,
    "halueval_val": 50 samples,
}

def should_keep_checkpoint(current_metrics, best_metrics):
    # Must not regress >5% on any domain
    for domain in VALIDATION_SET:
        if current_metrics[domain] < best_metrics[domain] - 0.05:
            return False  # Reject: catastrophic regression
    # Must improve overall
    return mean(current_metrics) >= mean(best_metrics)
```

### 6. Proposer Curriculum with Warm-Start

**Problem**: First epoch probes may be randomly difficult, wasting training signal.

**Solution**: Progressive difficulty schedule.

```
Epoch 1: 70% easy (clear S/C/N) + 30% boundary
Epoch 2: 50% easy + 50% boundary
Epoch 3: 30% easy + 70% boundary  
Epoch 4+: 10% easy + 90% boundary (maintain baseline)
```

The "easy" samples serve as anchors that prevent forgetting basic verification.
The "boundary" samples provide the learning signal.

The 10% easy samples in later epochs act like the replay buffer — they keep
the model grounded on fundamentals.

## Training Pipeline Summary

```
Epoch 0 (Warm-start):
  - SFT checkpoint as starting point
  - Evaluate on multi-domain validation → baseline metrics
  - Initialize empty ReasoningBank + ReplayBuffer

Epoch 1-N (Self-Evolution):
  1. Proposer generates domain-diverse probes
     - Risk types × Domains × Difficulty levels
     - Quality filtering (~30% rejection)
     - Adaptive: targets Verifier's current weaknesses
  
  2. Construct training batch:
     - 80% new probes
     - 20% replay buffer (hard examples)
  
  3. GRPO training (1 inner epoch):
     - Retrieve rules → inject into prompt
     - 5 rollouts per sample
     - Structured reward × boundary bonus
     - KL penalty (λ=0.01) against SFT reference
     - Track rule citations in responses
  
  4. Evolution step:
     - Evaluate on multi-domain validation set
     - Validation gate: keep or revert checkpoint
     - Extract failures → distill rules
     - Update rule effectiveness from citations
     - Evict low-performing rules
     - Update Proposer weakness profile
     - Add hard examples to replay buffer
  
  5. Log: accuracy curve, rule count, domain balance, difficulty distribution
```

## Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Outer epochs | 5-8 | Enough for convergence without overfitting |
| Inner GRPO steps per epoch | ~35 (1 pass over data) | Prevents overfitting to one epoch's data |
| GRPO group size | 5 | Balance: variance reduction vs compute |
| KL penalty λ | 0.01 | Light regularization, not too constraining |
| Boundary bonus weight β | 0.5 | Equal weight to structured + boundary |
| Replay buffer size | 500 | ~10% of total training data |
| Replay ratio | 20% | Enough to prevent forgetting |
| ReasoningBank max rules | 100 | Manageable prompt length |
| Rules injected per sample | 10 | Top-K by relevance |
| Validation gate threshold | -5% | Allow small regression, block catastrophic |
| Domain weights | Inverse accuracy | Auto-balance toward weak domains |
| Proposer probes per epoch | 400-600 | Enough diversity per epoch |

## Comparison with Related Work

| Feature | Dr.Zero | SkillRL | SEVA v1 | **SEVA v2** |
|---------|---------|---------|---------|-------------|
| Co-evolution | Proposer+Solver | Skill+Policy | Proposer+Verifier (separate) | **Proposer+Verifier (joint)** |
| Difficulty control | Boundary-optimal reward | N/A | Adaptive difficulty (prompt) | **Boundary reward + curriculum** |
| Rule/Skill bank | N/A | SkillBank | ReasoningBank (prompt only) | **Rule-conditioned + tracked** |
| Anti-forgetting | N/A | N/A | N/A | **Replay buffer + val gate** |
| Domain diversity | Single domain | Single domain | Single domain | **Multi-domain diversification** |
| Reward design | Binary | Task-specific | 4-component structured | **Structured × boundary bonus** |
