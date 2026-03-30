# Verifiable Agent — Prompt Version

A three-stage pipeline for automated LLM hallucination detection and reliability evaluation, with a self-evolving feedback loop.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Proposer    │────▶│  Responder   │────▶│  Verifier   │
│  (Stage 1)   │     │  (Stage 2)   │     │  (Stage 3)  │
│              │     │              │     │             │
│ Generate     │  x  │ Target LLM   │  α  │ Decompose → │
│ safety       │────▶│ generates    │────▶│ Match     → │
│ probes       │     │ response     │     │ Score       │
└──────┬───────┘     └──────────────┘     └──────┬──────┘
       │                                         │
       │         ┌──────────────────┐            │
       └─────────│  Self-Evolution  │◀───────────┘
                 │  Failure → Novel │
                 │  patterns → Πₑ₊₁ │
                 └──────────────────┘
```

### Stage 1: Proposer
Generates **safety probes** targeting 4 risk categories:
- **Missing Evidence** — topics with scarce public information
- **Multi-Hop Reasoning** — questions requiring chained factual steps
- **Pressure/Presupposition** — questions with false or disputed premises
- **Unanswerable** — fundamentally unanswerable questions

### Stage 2: Responder
Queries the target model(s) under evaluation and records raw responses.

### Stage 3: Verifier
1. **Claim Decomposition** — breaks responses into atomic claims
2. **Evidence Matching** — labels each claim as Supported (S), Contradicted (C), or Not Mentioned (N)
3. **Scoring** — aggregates claim labels into a response-level reliability score

### Self-Evolution Loop
Extracts informative failures (false positives, false negatives, boundary cases) and feeds them back into the Proposer for the next epoch.

## Setup

```bash
cd Verifiable_agent
pip install -r requirements.txt
```

Set API keys as environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # if using Claude
```

## Usage

### Run experiment
```bash
# Single default experiment
python scripts/run_experiment.py --config configs/default.yaml

# Multi-model comparison
python scripts/run_experiment.py --config configs/experiments/multi_model.yaml
```

### Analyze results
```bash
python scripts/analyze_results.py --results results/<run_dir>/all_results.json
```

This generates:
- **Table 1**: Hallucination rate per model per risk type (CSV)
- **Table 2**: S/C/N distribution per model (CSV)
- **Figure 1**: Radar chart — model reliability across risk types
- **Figure 2**: Evolution curve — hallucination rate across epochs
- **Figure 3**: Heatmap — contradiction rate by model × risk type

### Build knowledge base
```bash
python scripts/build_knowledge_base.py --path knowledge_base/documents
```

### Run tests
```bash
pytest tests/ -v
```

## Project Structure

```
├── configs/               # Experiment configurations (YAML)
│   ├── default.yaml
│   ├── models/            # Per-model configs
│   └── experiments/       # Multi-model experiment configs
├── src/
│   ├── llm/               # LLM provider abstraction (OpenAI, Anthropic, vLLM)
│   ├── proposer/          # Stage 1: safety probe generation
│   ├── responder/         # Stage 2: target model querying
│   ├── verifier/          # Stage 3: claim decomposition + evidence matching + scoring
│   ├── evolution/         # Self-evolving loop (failure extraction + strategy update)
│   ├── data/              # Pydantic data schemas
│   └── utils/             # Logging + metrics
├── knowledge_base/        # Pre-built evidence documents (JSONL)
├── scripts/               # Experiment runner + analysis
├── results/               # Output (gitignored)
└── tests/                 # Unit tests
```

## Configuration

All experiment parameters are defined in YAML config files. Key settings:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_epochs` | Self-evolution epochs | 3 |
| `num_probes_per_type` | Probes per risk category | 10 |
| `proposer_llm` | LLM for probe generation | gpt-4o |
| `verifier_llm` | LLM for verification | gpt-4o |
| `responder_models` | Models to evaluate | gpt-4o, claude-sonnet, gpt-4o-mini |
| `scoring.contradicted_weight` | Penalty for contradictions | -2.0 |

## Supported LLM Providers

| Provider | Models | Config key |
|----------|--------|-----------|
| OpenAI | GPT-4o, GPT-4o-mini, etc. | `openai` |
| Anthropic | Claude Sonnet, Opus, Haiku | `anthropic` |
| vLLM | Llama-3, Mistral, etc. | `vllm` |
