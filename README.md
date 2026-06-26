# SEVA: Structured Evidence Verification Agent with Process Reward Optimization

**Paper**: *SEVA: Structured Evidence Verification Agents with Process Reward Optimization* (ARR 2026)

SEVA is a fact attribution verification framework that produces **structured, interpretable output** --- evidence alignment, step-by-step reasoning chains, calibrated confidence, and fine-grained error diagnosis --- trained with **GRPO and a novel process reward function**.

## Key Contributions

1. **Structured verification output**: Instead of opaque binary labels, SEVA produces evidence alignment spans, reasoning chains, and error diagnosis with a 6-category taxonomy.
2. **Process reward function**: Decomposes verification quality into 5 independently scored components (format, alignment, chain, label, diagnosis), creating a smooth optimization landscape for RL.
3. **GRPO training for fact verification**: First application of GRPO to fact attribution, achieving +4.1 F1 over SFT with near-perfect structural quality.

## Architecture

```
Input (claim, source)
        │
        ▼
┌──────────────────────┐
│   SEVA Model         │
│   (Qwen2.5-3B/7B)   │
└──────────┬───────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────┐
│Evidence│  │Reasoning │
│Align.  │  │Chain     │
└────────┘  └──────────┘
    ▼             ▼
┌────────┐  ┌──────────┐
│Label + │  │Error     │
│Confid. │  │Diagnosis │
└────────┘  └──────────┘
           │
           ▼
┌──────────────────────┐
│  Process Reward R(v) │
│  R_f(10%) + R_a(30%) │
│  + R_c(30%) + R_l(15%)│
│  + R_d(15%) + R_cal  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  GRPO Optimization   │
│  G=8, T=1.2          │
└──────────────────────┘
```

## Results

### ClearFacts Benchmark

| Model | Size | Output | F1 |
|-------|------|--------|-----|
| Llama-3.1 (0-shot) | 8B | binary | 67.2 |
| MiniCheck | 7B | binary | 81.2 |
| ClearCheck | 8B | binary | ~84 |
| **SEVA-SFT** | **3B** | **structured** | **64.9** |
| **SEVA-GRPO** | **3B** | **structured** | **69.0** |

### Structural Quality (ClearFacts)

| | Alignment | Chain | Format |
|---|-----------|-------|--------|
| SEVA-SFT | 0.917 | 0.917 | 72% |
| SEVA-GRPO | **0.997** | **0.995** | **100%** |

### Multi-Benchmark (Macro F1, 200 samples each)

| | FEVER | TruthfulQA | SciFact | HaluEval | MusiQue | FactScore |
|---|-------|------------|---------|----------|---------|-----------|
| Zero-shot | 63.3 | 43.7 | 78.0 | 29.5 | 22.2 | 82.1 |
| SFT | 76.3 | 72.1 | 59.9 | 42.0 | 22.2 | 82.1 |
| GRPO | **84.9** | **82.7** | 22.1 | 39.4 | 22.2 | 43.5 |

## Training Pipeline

### Phase 1: SFT Data Generation
Generate structured annotations using GPT-4o-mini as teacher:
```bash
python scripts/generate_sft_data.py \
    --input data/attribution/anli_train.jsonl \
    --output data/attribution/seva_sft_train.jsonl
```

### Phase 2: Supervised Fine-Tuning
```bash
# 3B full fine-tuning
CUDA_VISIBLE_DEVICES=0,1 python scripts/train_seva_sft.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --train-file data/attribution/seva_sft_train.jsonl

# 7B with LoRA
CUDA_VISIBLE_DEVICES=0 python scripts/train_seva_sft.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --lora --lora-rank 64 --merge-lora
```

### Phase 3: GRPO with Process Reward
```bash
# Using veRL framework
cd drzero && bash scripts/run_grpo_attribution.sh
```

The process reward is defined in `drzero/verl/custom_reward/seva_reward.py`.

## Self-Evolution Loop

SEVA includes a self-evolution pipeline (VERIFY -> REFLECT -> PROBE -> REFINE):

```bash
python scripts/run_self_evolution.py \
    --model /path/to/checkpoint \
    --rounds 3
```

| Phase | Script | Description |
|-------|--------|-------------|
| VERIFY | `scripts/eval_seva.py` | Evaluate on benchmarks, collect predictions |
| REFLECT | `scripts/analyze_failures.py`, `scripts/extract_rules.py` | Build weakness profile, extract rules to ReasoningBank |
| PROBE | `scripts/generate_adversarial_probes.py` | Generate targeted adversarial examples |
| REFINE | `scripts/run_self_evolution.py` | Additional GRPO on hard examples + rule injection |

## Evaluation

```bash
# Evaluate on ClearFacts
python scripts/eval_seva.py \
    --model /path/to/checkpoint \
    --benchmarks clearfacts

# Evaluate on all benchmarks
python scripts/eval_seva.py \
    --model /path/to/checkpoint \
    --all

# With ReasoningBank rules
python scripts/eval_seva.py \
    --model /path/to/checkpoint \
    --rules-file reasoning_bank/rules_prompt.txt
```

## Process Reward Function

The reward decomposes verification quality into 5 components:

```
R = 0.10*R_f + 0.30*R_a + 0.30*R_c + 0.15*R_l + 0.15*R_d + R_cal
```

| Component | Weight | Description |
|-----------|--------|-------------|
| R_f (Format) | 0.10 | Valid JSON with required fields |
| R_a (Alignment) | 0.30 | Evidence span extraction quality |
| R_c (Chain) | 0.30 | Reasoning step quality |
| R_l (Label) | 0.15 | Label accuracy |
| R_d (Diagnosis) | 0.15 | Error type + fix suggestion |
| R_cal | +/- | Calibration bonus/penalty |

Process components (R_f + R_a + R_c) = **70%**, Outcome components (R_l + R_d) = **30%**.

## Error Taxonomy

| Error Type | Description |
|------------|-------------|
| `numerical_exaggeration` | Number inflated/deflated |
| `negation_flip` | Negation added/removed |
| `scope_inflation` | Specific claim overgeneralized |
| `temporal_shift` | Time qualifier altered |
| `entity_substitution` | Named entity swapped |
| `fabrication` | Information absent from source |

## Project Structure

```
├── src/
│   ├── verifier/              # Core verification pipeline
│   │   ├── seva_format.py     # Structured output schema + prompts
│   │   ├── verifier.py        # Verification orchestrator
│   │   ├── decomposer.py      # Claim decomposition
│   │   ├── evidence_matcher.py # Evidence matching
│   │   ├── scorer.py          # Reliability scoring
│   │   └── calibration.py     # Prediction calibration
│   ├── proposer/              # Safety probe generation
│   ├── evolution/             # Self-evolution loop
│   │   ├── evolver.py         # Failure analysis + strategy update
│   │   ├── reasoning_bank.py  # Verification rule accumulation
│   │   └── failure_extractor.py
│   ├── baselines/             # Baseline methods (SelfCheck, CoVe, SAFE, etc.)
│   ├── benchmarks/            # Benchmark loaders (FEVER, TruthfulQA, etc.)
│   ├── llm/                   # LLM providers (OpenAI, Anthropic, vLLM)
│   └── tools/                 # External tools (web search, calculator, etc.)
├── drzero/                    # veRL-based GRPO training framework
│   ├── verl/custom_reward/
│   │   └── seva_reward.py     # Process reward implementation
│   ├── config/                # Training configs
│   └── scripts/               # Training launch scripts
├── scripts/
│   ├── train_seva_sft.py      # SFT training (full + LoRA)
│   ├── eval_seva.py           # Structured evaluation
│   ├── analyze_failures.py    # VERIFY/REFLECT phase
│   ├── extract_rules.py       # ReasoningBank rule extraction
│   ├── generate_adversarial_probes.py  # PROBE phase
│   ├── run_self_evolution.py  # Full self-evolution orchestrator
│   ├── generate_sft_data.py   # SFT data generation
│   └── run_fair_comparison.py # Baseline comparisons
├── configs/                   # Experiment configurations
├── data/attribution/          # Training data (SFT + GRPO)
├── tests/                     # Unit tests
├── paper/arr2026/             # Paper (ACL ARR 2026)
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Justin0504/Verifiable_agent.git
cd Verifiable_agent
pip install -r requirements.txt
```

### Requirements
- Python >= 3.10
- PyTorch >= 2.1
- transformers >= 4.40
- peft >= 0.10 (for LoRA)
- veRL (included in `drzero/`)

### Environment Variables
```bash
export OPENAI_API_KEY="your-key"      # For SFT data generation
export ANTHROPIC_API_KEY="your-key"   # Optional: Claude baselines
```

## Citation

```bibtex
@inproceedings{yuan2026seva,
  title={{SEVA}: Self-Evolving Verification Agent with Process Reward for Fact Attribution},
  author={Yuan, Aojie and Nian, Yi and Zhang, Haiyue and Su, Zijian and Zhao, Yue},
  booktitle={ICML 2026 Workshop on Trustworthy AI for Good (AI4GOOD)},
  year={2026}
}
```

## License

MIT
