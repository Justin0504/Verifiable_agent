"""SEVA Self-Evolution Training Loop (v2 — fully upgraded).

Implements the PROBE → REFLECT → REFINE → VERIFY cycle for attribution verification.

Upgrades over v1:
  - Boundary-optimal reward (group-level, Dr.Zero)
  - Anti-collapse: replay buffer + easy anchors + hard validation gate + checkpoint rollback
  - Contrastive probe pairs (minimal-edit negatives)
  - Rollout consistency signal for prioritized rule distillation
  - Dynamic reward scheduling (accuracy-heavy → calibration-heavy)
  - Difficulty curriculum (warm start → full adversarial)

Usage:
    python scripts/train_seva.py --model /path/to/sft_checkpoint --epochs 5
    python scripts/train_seva.py --model /path/to/sft_checkpoint --proposer-llm openai
"""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "drzero" / "verl" / "custom_reward"))

from src.proposer.attribution_proposer import (
    AttributionProbe,
    AttributionProposer,
    AMBIGUITY_TYPES,
)
from src.evolution.reasoning_bank import ReasoningBank
from attribution_reward import (
    extract_json_from_response,
    normalize_label,
    get_reward_weights,
    compute_score,
)


# ============================================================
# Config
# ============================================================
DEFAULT_MODEL = os.environ.get(
    "MODEL_PATH",
    "/home/yinian/verifiable_agent/checkpoints/sft_attribution/final",
)
DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    "/home/yinian/verifiable_agent/data/attribution",
))
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "/home/yinian/verifiable_agent/checkpoints/seva_attribution",
))

SYSTEM_PROMPT = (
    "You are a fact attribution verifier. Given a claim and a source document, "
    "determine whether the claim is attributable to (supported by) the source.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"label": "Attributable" or "Not Attributable", '
    '"confidence": 0.0-1.0, '
    '"reasoning": "brief explanation"}'
)

USER_TEMPLATE = (
    "Claim: {claim}\n\n"
    "Source: {source}\n\n"
    "Is this claim attributable to the source? Respond with JSON only."
)

# Anti-collapse: regression threshold per benchmark
REGRESSION_THRESHOLD = 0.05  # 5% drop → rollback


# ============================================================
# Model utilities
# ============================================================
def load_model(model_path: str, device: str = "auto"):
    """Load model and tokenizer for inference."""
    print(f"  Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_response(
    model, tokenizer, claim: str, source: str,
    rules_text: str = "", max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Generate attribution verification response."""
    system = SYSTEM_PROMPT
    if rules_text:
        system += rules_text

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(
            claim=claim, source=source[:2000],
        )},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


def generate_rollouts(
    model, tokenizer, claim: str, source: str,
    rules_text: str = "", n: int = 5,
) -> list[dict]:
    """Generate n rollouts with temperature=0.7 for consistency analysis."""
    rollouts = []
    for _ in range(n):
        response = generate_response(
            model, tokenizer, claim, source,
            rules_text=rules_text, temperature=0.7,
        )
        parsed = extract_json_from_response(response)
        if parsed is None:
            label = "Not Attributable"
            confidence = 0.5
            reasoning = ""
        else:
            label = normalize_label(parsed.get("label", "")) or "Not Attributable"
            confidence = parsed.get("confidence", 0.5)
            reasoning = parsed.get("reasoning", "")
        rollouts.append({
            "response": response,
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
        })
    return rollouts


# ============================================================
# Replay Buffer & Easy Anchors
# ============================================================
class ReplayBuffer:
    """Stores hard examples from previous epochs for anti-collapse."""

    def __init__(self, max_size: int = 500):
        self.buffer: list[dict] = []
        self.max_size = max_size

    def add(self, samples: list[dict]):
        """Add hard samples (incorrect or low-consistency)."""
        self.buffer.extend(samples)
        if len(self.buffer) > self.max_size:
            # Keep the most recent + hardest
            self.buffer.sort(key=lambda x: x.get("consistency", 1.0))
            self.buffer = self.buffer[:self.max_size]

    def sample(self, n: int) -> list[dict]:
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.buffer, f, indent=2, default=str)

    def load(self, path: Path):
        if path.exists():
            with open(path) as f:
                self.buffer = json.load(f)


def load_easy_anchors(data_dir: Path, n: int = 100) -> list[dict]:
    """Load easy samples from SFT training data as anti-collapse anchors."""
    sft_path = data_dir / "sft_val.jsonl"
    if not sft_path.exists():
        return []

    samples = []
    with open(sft_path) as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get("messages", [])
            if len(msgs) < 3:
                continue
            # Extract claim and source from user message
            user_msg = msgs[1]["content"]
            assistant_msg = msgs[2]["content"]
            # Parse the assistant response for label
            parsed = extract_json_from_response(assistant_msg)
            if parsed is None:
                continue
            label = normalize_label(parsed.get("label", ""))
            if label is None:
                continue
            # Extract claim/source from user template
            claim = ""
            source = ""
            if "Claim:" in user_msg and "Source:" in user_msg:
                parts = user_msg.split("Source:")
                claim = parts[0].replace("Claim:", "").strip()
                source = parts[1].strip().rstrip("Is this claim attributable to the source? Respond with JSON only.").strip()

            if claim and source:
                samples.append({
                    "claim": claim,
                    "source": source,
                    "gold_label": label,
                    "ambiguity_type": "easy_anchor",
                })

    random.shuffle(samples)
    return samples[:n]


# ============================================================
# Phase 1: PROBE — Adversarial probe generation with curriculum
# ============================================================
def probe_phase(
    proposer: AttributionProposer,
    epoch: int,
    total_epochs: int,
    n_probes: int = 40,
    easy_anchors: list[dict] | None = None,
    replay_buffer: ReplayBuffer | None = None,
) -> list[AttributionProbe]:
    """Generate adversarial probes with difficulty curriculum.

    Curriculum:
      Epoch 1: 60% probes + 20% easy anchors + 20% replay
      Epoch 2: 70% probes + 10% easy anchors + 20% replay
      Epoch 3+: 70% probes (incl. contrastive) + 10% easy + 20% replay
    """
    print("\n[PROBE] Generating adversarial probes...")

    # --- Difficulty curriculum ---
    progress = min((epoch - 1) / max(total_epochs - 1, 1), 1.0)
    n_easy = max(2, int(n_probes * (0.20 - 0.10 * progress)))   # 20% → 10%
    n_replay = max(2, int(n_probes * 0.20))                      # constant 20%
    n_new = n_probes - n_easy - n_replay

    # Generate new adversarial probes
    adaptive = (epoch > 1)
    new_probes = proposer.generate_all(
        n_total=n_new,
        adaptive=adaptive,
        filter_quality=True,
    )

    # Generate contrastive pairs from epoch 2 onward
    if epoch >= 2:
        contrastive = proposer.generate_contrastive_pairs(
            new_probes, n_per_probe=1,
        )
        print(f"  Generated {len(contrastive)} contrastive pairs")
        new_probes.extend(contrastive)

    # Add easy anchors (anti-collapse)
    if easy_anchors:
        anchors_sample = random.sample(easy_anchors, min(n_easy, len(easy_anchors)))
        for a in anchors_sample:
            new_probes.append(AttributionProbe(
                id=f"anchor_{random.randint(0, 99999):05d}",
                claim=a["claim"],
                source=a["source"],
                gold_label=a["gold_label"],
                ambiguity_type="easy_anchor",
            ))

    # Add replay buffer (hard cases from previous epochs)
    if replay_buffer:
        replay_samples = replay_buffer.sample(n_replay)
        for r in replay_samples:
            new_probes.append(AttributionProbe(
                id=f"replay_{random.randint(0, 99999):05d}",
                claim=r["claim"],
                source=r["source"],
                gold_label=r["gold_label"],
                ambiguity_type=r.get("ambiguity_type", "replay"),
            ))

    # Stats
    type_counts = Counter(p.ambiguity_type for p in new_probes)
    label_counts = Counter(p.gold_label for p in new_probes)
    print(f"  Total probes: {len(new_probes)}")
    print(f"  By type: {dict(type_counts)}")
    print(f"  By label: {dict(label_counts)}")

    random.shuffle(new_probes)
    return new_probes


# ============================================================
# Phase 2: REFLECT — Rollout consistency + rule distillation
# ============================================================
def reflect_phase(
    model,
    tokenizer,
    probes: list[AttributionProbe],
    reasoning_bank: ReasoningBank,
    replay_buffer: ReplayBuffer,
    epoch: int,
    n_rollouts: int = 5,
) -> tuple[dict, list[dict]]:
    """Run verifier on probes with multi-rollout consistency analysis.

    Returns:
        (metrics_dict, trajectories_list)
    """
    print(f"\n[REFLECT] Running verifier on {len(probes)} probes "
          f"({n_rollouts} rollouts each)...")

    trajectories = []
    correct = 0
    total = 0
    type_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, probe in enumerate(probes):
        relevant_rules = reasoning_bank.retrieve_relevant(probe.claim)
        rules_text = reasoning_bank.format_for_prompt(relevant_rules)

        # Multi-rollout for consistency analysis
        rollouts = generate_rollouts(
            model, tokenizer,
            claim=probe.claim,
            source=probe.source,
            rules_text=rules_text,
            n=n_rollouts,
        )

        # Consistency analysis
        label_counts = Counter(r["label"] for r in rollouts)
        majority_label = label_counts.most_common(1)[0][0]
        consistency = label_counts.most_common(1)[0][1] / n_rollouts
        avg_confidence = sum(r["confidence"] for r in rollouts) / n_rollouts

        # Use majority vote as prediction
        is_correct = majority_label == probe.gold_label
        if is_correct:
            correct += 1
        total += 1

        type_results[probe.ambiguity_type]["total"] += 1
        if is_correct:
            type_results[probe.ambiguity_type]["correct"] += 1

        # Record rule usage
        for rule in relevant_rules:
            reasoning_bank.record_usage(rule.id, was_helpful=is_correct)

        trajectory = {
            "claim": probe.claim,
            "evidence": probe.source,
            "predicted_label": majority_label,
            "gold_label": probe.gold_label,
            "confidence": avg_confidence,
            "consistency": consistency,
            "label_distribution": dict(label_counts),
            "context": f"ambiguity_type={probe.ambiguity_type}",
            "correct": is_correct,
            "ambiguity_type": probe.ambiguity_type,
            "reasoning": rollouts[0]["reasoning"],
        }
        trajectories.append(trajectory)

        # Add to replay buffer: incorrect OR low-consistency
        if not is_correct or consistency < 0.6:
            replay_buffer.add([{
                "claim": probe.claim,
                "source": probe.source,
                "gold_label": probe.gold_label,
                "ambiguity_type": probe.ambiguity_type,
                "consistency": consistency,
            }])

        if (i + 1) % 20 == 0:
            acc = correct / total
            print(f"  [{i+1}/{len(probes)}] Accuracy: {acc:.1%}")

    overall_acc = correct / max(total, 1)
    print(f"  Overall accuracy: {overall_acc:.1%}")

    # Per-type accuracy
    type_accuracies = {}
    for atype, res in sorted(type_results.items()):
        acc = res["correct"] / max(res["total"], 1)
        type_accuracies[atype] = acc
        print(f"    {atype}: {acc:.1%} ({res['correct']}/{res['total']})")

    # Consistency stats
    consistencies = [t["consistency"] for t in trajectories]
    avg_consistency = sum(consistencies) / max(len(consistencies), 1)
    boundary_count = sum(1 for c in consistencies if c < 0.8)
    print(f"  Avg consistency: {avg_consistency:.2f}")
    print(f"  Boundary samples (consistency < 0.8): {boundary_count}/{len(trajectories)}")

    # --- Prioritized rule distillation ---
    # Priority 1: Low consistency + incorrect (model confused AND wrong)
    priority_high = [t for t in trajectories if not t["correct"] and t["consistency"] < 0.6]
    # Priority 2: Incorrect but consistent (systematic error — most valuable)
    priority_med = [t for t in trajectories if not t["correct"] and t["consistency"] >= 0.6]
    # Priority 3: Correct + high confidence (capture good strategies)
    priority_low = [t for t in trajectories if t["correct"] and t["confidence"] >= 0.8]

    to_distill = priority_high[:8] + priority_med[:8] + priority_low[:4]
    if to_distill and reasoning_bank.llm:
        new_rules = reasoning_bank.distill_batch(to_distill, epoch=epoch)
        print(f"  Distilled {len(new_rules)} rules "
              f"(from {len(priority_high)} confused + {len(priority_med)} systematic "
              f"+ {min(len(priority_low), 4)} correct)")
    else:
        print("  No LLM for distillation, skipping rule extraction")

    reasoning_bank.save()

    metrics = {
        "overall_accuracy": overall_acc,
        "type_accuracies": type_accuracies,
        "avg_consistency": avg_consistency,
        "boundary_count": boundary_count,
        "n_trajectories": len(trajectories),
        "n_failures": sum(1 for t in trajectories if not t["correct"]),
        "bank_stats": reasoning_bank.stats(),
    }
    return metrics, trajectories


# ============================================================
# Phase 3: REFINE — GRPO with data mixing
# ============================================================
def refine_phase(
    probes: list[AttributionProbe],
    trajectories: list[dict],
    base_data_dir: Path,
    model_path: str,
    output_path: Path,
    epoch: int,
    total_epochs: int,
    reasoning_bank: ReasoningBank,
    replay_buffer: ReplayBuffer,
    grpo_script: str | None = None,
) -> str:
    """Prepare mixed training data and launch GRPO training.

    Data composition:
      60% new probes (from PROBE phase)
      20% replay buffer (hard cases from previous epochs)
      20% easy anchors (from original SFT data)
    """
    print("\n[REFINE] Preparing GRPO training data...")

    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []

    # --- 1. New probes (includes contrastive pairs) ---
    for probe in probes:
        relevant_rules = reasoning_bank.retrieve_relevant(probe.claim, top_k=5)
        rules_text = reasoning_bank.format_for_prompt(relevant_rules)

        system_prompt = SYSTEM_PROMPT
        if rules_text:
            system_prompt += rules_text

        records.append({
            "data_source": "seva_probe",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TEMPLATE.format(
                    claim=probe.claim, source=probe.source[:2000],
                )},
            ],
            "ability": "attribution_verification",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": probe.gold_label},
            },
            "extra_info": {
                "ambiguity_type": probe.ambiguity_type,
                "epoch": epoch,
                "total_epochs": total_epochs,
            },
        })

    # --- 2. Mix in base GRPO training data (easy anchors + diversity) ---
    base_grpo = base_data_dir / "grpo_train.parquet"
    if base_grpo.exists():
        existing_table = pq.read_table(base_grpo)
        n_existing = len(existing_table)
        n_mix = min(int(len(records) * 0.3), n_existing)  # Up to 30%
        if n_mix > 0:
            indices = random.sample(range(n_existing), n_mix)
            for idx in indices:
                row = existing_table.slice(idx, 1)
                records.append({
                    "data_source": row.column("data_source")[0].as_py(),
                    "prompt": json.loads(row.column("prompt")[0].as_py()),
                    "ability": row.column("ability")[0].as_py(),
                    "reward_model": json.loads(row.column("reward_model")[0].as_py()),
                    "extra_info": {
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "source": "base_data",
                    },
                })
            print(f"  Mixed in {n_mix} base training samples")

    random.shuffle(records)

    # Save as parquet
    epoch_data_dir = output_path / f"epoch_{epoch}" / "data"
    epoch_data_dir.mkdir(parents=True, exist_ok=True)

    prompt_strs = [json.dumps(r["prompt"]) for r in records]
    reward_strs = [json.dumps(r["reward_model"]) for r in records]
    extra_strs = [json.dumps(r["extra_info"]) for r in records]

    table = pa.table({
        "data_source": [r["data_source"] for r in records],
        "prompt": prompt_strs,
        "ability": [r["ability"] for r in records],
        "reward_model": reward_strs,
        "extra_info": extra_strs,
    })

    train_path = epoch_data_dir / "grpo_train.parquet"
    pq.write_table(table, train_path)
    print(f"  Saved {len(records)} training samples to {train_path}")

    # Launch GRPO training
    epoch_output = output_path / f"epoch_{epoch}" / "checkpoint"
    epoch_output.mkdir(parents=True, exist_ok=True)

    if grpo_script:
        print("  Launching GRPO training...")
        env = os.environ.copy()
        env["TRAIN_DATA"] = str(train_path)
        env["MODEL_PATH"] = model_path
        result = subprocess.run(
            ["bash", grpo_script], env=env,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: GRPO training failed: {result.stderr[:500]}")
        else:
            print("  GRPO training completed")
    else:
        print("  GRPO script not specified — skipping training step")
        print(f"  To train manually:")
        print(f"    MODEL_PATH={model_path} TRAIN_DATA={train_path} "
              f"bash drzero/train_attribution.sh")

    return str(epoch_output)


# ============================================================
# Phase 4: VERIFY — Hard validation gate with rollback
# ============================================================
def verify_phase(
    model,
    tokenizer,
    eval_paths: list[Path],
    reasoning_bank: ReasoningBank,
    max_samples_per_bench: int = 200,
) -> dict:
    """Evaluate on multiple benchmarks. Returns per-benchmark metrics."""
    print("\n[VERIFY] Multi-benchmark evaluation...")

    all_results = {}
    for eval_path in eval_paths:
        if not eval_path.exists():
            continue

        bench_name = eval_path.stem
        samples = []
        with open(eval_path) as f:
            for line in f:
                samples.append(json.loads(line))

        if max_samples_per_bench and len(samples) > max_samples_per_bench:
            random.seed(42)  # deterministic subset
            samples = random.sample(samples, max_samples_per_bench)

        correct = 0
        total = 0
        preds = []
        golds = []

        for sample in samples:
            claim = sample.get("claim", "")
            source = sample.get("source", "")
            gold = sample.get("gold_label", "Not Attributable")
            if not claim or not source:
                continue

            relevant_rules = reasoning_bank.retrieve_relevant(claim)
            rules_text = reasoning_bank.format_for_prompt(relevant_rules)
            response = generate_response(
                model, tokenizer, claim, source, rules_text=rules_text,
            )

            parsed = extract_json_from_response(response)
            pred = "Not Attributable"
            if parsed:
                pred = normalize_label(parsed.get("label", "")) or "Not Attributable"

            preds.append(pred)
            golds.append(gold)
            if pred == gold:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        all_results[bench_name] = {
            "accuracy": accuracy,
            "n_samples": total,
            "n_correct": correct,
            "pred_distribution": dict(Counter(preds)),
            "gold_distribution": dict(Counter(golds)),
        }
        print(f"  {bench_name}: {accuracy:.1%} ({correct}/{total})")

    # Compute overall weighted accuracy
    total_correct = sum(r["n_correct"] for r in all_results.values())
    total_samples = sum(r["n_samples"] for r in all_results.values())
    overall = total_correct / max(total_samples, 1)

    all_results["_overall"] = {
        "accuracy": overall,
        "n_samples": total_samples,
        "n_correct": total_correct,
    }
    print(f"  Overall: {overall:.1%} ({total_correct}/{total_samples})")

    return all_results


def check_validation_gate(
    current_results: dict,
    best_results: dict,
    threshold: float = REGRESSION_THRESHOLD,
) -> tuple[bool, list[str]]:
    """Check if any benchmark regressed beyond threshold.

    Returns:
        (passed: bool, regressed_benchmarks: list[str])
    """
    if not best_results:
        return True, []

    regressed = []
    for bench, metrics in current_results.items():
        if bench.startswith("_"):
            continue
        if bench in best_results:
            best_acc = best_results[bench]["accuracy"]
            curr_acc = metrics["accuracy"]
            if curr_acc < best_acc - threshold:
                regressed.append(
                    f"{bench}: {best_acc:.1%} → {curr_acc:.1%} "
                    f"(Δ={curr_acc - best_acc:+.1%})"
                )

    passed = len(regressed) == 0
    return passed, regressed


# ============================================================
# Main: Self-Evolution Loop
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SEVA Self-Evolution Training v2")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--n-probes", type=int, default=50)
    parser.add_argument("--n-rollouts", type=int, default=5,
                        help="Rollouts per probe for consistency analysis")
    parser.add_argument("--eval-data", nargs="+",
                        default=[str(DATA_DIR / "clearfacts.jsonl")])
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--grpo-script", default=None)
    parser.add_argument("--proposer-model", default="gpt-4o-mini")
    parser.add_argument("--proposer-provider", default="openai",
                        choices=["openai", "anthropic"])
    parser.add_argument("--bank-path", default=None)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEVA: Self-Evolving Verification Agent (v2)")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Epochs:      {args.epochs}")
    print(f"Probes/ep:   {args.n_probes}")
    print(f"Rollouts:    {args.n_rollouts}")
    print(f"Eval data:   {args.eval_data}")
    print(f"Output:      {args.output}")
    rw = get_reward_weights(1, args.epochs)
    print(f"Reward (ep1): {rw}")
    rw = get_reward_weights(args.epochs, args.epochs)
    print(f"Reward (ep{args.epochs}): {rw}")
    print("=" * 70)

    # --- Initialize components ---

    # Proposer LLM
    proposer_llm = None
    try:
        if args.proposer_provider == "openai":
            from src.llm.openai_llm import OpenAILLM
            proposer_llm = OpenAILLM(model=args.proposer_model, temperature=0.8)
        elif args.proposer_provider == "anthropic":
            from src.llm.anthropic_llm import AnthropicLLM
            proposer_llm = AnthropicLLM(model=args.proposer_model, temperature=0.8)
        print(f"Proposer LLM: {args.proposer_provider} {args.proposer_model}")
    except Exception as e:
        print(f"WARNING: Could not load proposer LLM: {e}")

    # ReasoningBank
    bank_path = args.bank_path or str(output_dir / "reasoning_bank.json")
    reasoning_bank = ReasoningBank(path=bank_path, llm=proposer_llm)
    print(f"ReasoningBank: {reasoning_bank.stats()}")

    # Proposer
    proposer = AttributionProposer(llm=proposer_llm, seed=args.seed) if proposer_llm else None

    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=500)
    replay_path = output_dir / "replay_buffer.json"
    replay_buffer.load(replay_path)
    print(f"Replay buffer: {len(replay_buffer.buffer)} samples")

    # Easy anchors (from SFT data — anti-collapse)
    easy_anchors = load_easy_anchors(DATA_DIR, n=200)
    print(f"Easy anchors: {len(easy_anchors)} samples")

    # Verifier model
    model, tokenizer = load_model(args.model)
    current_model_path = args.model

    # Eval paths
    eval_paths = [Path(p) for p in args.eval_data]

    # Tracking
    epoch_metrics = []
    best_results = {}
    best_model_path = args.model
    best_overall_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")
        epoch_start = time.time()

        # === Phase 1: PROBE ===
        probes = []
        if not args.skip_probe and proposer is not None:
            probes = probe_phase(
                proposer,
                epoch=epoch,
                total_epochs=args.epochs,
                n_probes=args.n_probes,
                easy_anchors=easy_anchors,
                replay_buffer=replay_buffer,
            )
        else:
            print("  [PROBE] Skipped")

        # === Phase 2: REFLECT ===
        reflect_results = {}
        trajectories = []
        if probes:
            reflect_results, trajectories = reflect_phase(
                model, tokenizer, probes, reasoning_bank,
                replay_buffer, epoch,
                n_rollouts=args.n_rollouts,
            )
            # Update proposer with per-type accuracies
            if proposer is not None:
                proposer.update_type_accuracies(reflect_results["type_accuracies"])
                # Feed failure patterns to proposer
                failures = [
                    {
                        "ambiguity_type": t["ambiguity_type"],
                        "pattern": f"Failed: {t['claim'][:80]} "
                                   f"(pred={t['predicted_label']}, "
                                   f"gold={t['gold_label']}, "
                                   f"consistency={t['consistency']:.2f})",
                        "claim": t["claim"],
                        "consistency": t["consistency"],
                    }
                    for t in trajectories if not t["correct"]
                ]
                proposer.update_memory(failures[:20])
        else:
            print("  [REFLECT] Skipped (no probes)")

        # === Phase 3: REFINE ===
        if probes and not args.skip_refine:
            new_model_path = refine_phase(
                probes=probes,
                trajectories=trajectories,
                base_data_dir=DATA_DIR,
                model_path=current_model_path,
                output_path=output_dir,
                epoch=epoch,
                total_epochs=args.epochs,
                reasoning_bank=reasoning_bank,
                replay_buffer=replay_buffer,
                grpo_script=args.grpo_script,
            )
            # Reload if new checkpoint exists
            checkpoint_dir = output_dir / f"epoch_{epoch}" / "checkpoint"
            if (checkpoint_dir / "config.json").exists():
                print(f"  Reloading model from {checkpoint_dir}")
                del model
                torch.cuda.empty_cache()
                model, tokenizer = load_model(str(checkpoint_dir))
                current_model_path = str(checkpoint_dir)
        else:
            print("  [REFINE] Skipped")

        # === Phase 4: VERIFY (hard gate) ===
        verify_results = verify_phase(
            model, tokenizer, eval_paths,
            reasoning_bank,
            max_samples_per_bench=args.eval_samples,
        )

        # --- Validation gate ---
        gate_passed, regressed = check_validation_gate(
            verify_results, best_results, REGRESSION_THRESHOLD,
        )

        if gate_passed:
            overall_acc = verify_results["_overall"]["accuracy"]
            if overall_acc > best_overall_acc:
                best_overall_acc = overall_acc
                best_results = copy.deepcopy(verify_results)
                best_model_path = current_model_path
                print(f"  ✓ New best model: {overall_acc:.1%}")
            else:
                print(f"  ✓ Gate passed (no regression), "
                      f"but not new best ({overall_acc:.1%} vs {best_overall_acc:.1%})")
        else:
            print(f"  ✗ VALIDATION GATE FAILED — rollback to best checkpoint")
            for r in regressed:
                print(f"    Regressed: {r}")
            # Rollback
            if best_model_path != current_model_path:
                print(f"  Rolling back to: {best_model_path}")
                del model
                torch.cuda.empty_cache()
                model, tokenizer = load_model(best_model_path)
                current_model_path = best_model_path

        # Save replay buffer
        replay_buffer.save(replay_path)

        # Epoch metrics
        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch,
            "probe_accuracy": reflect_results.get("overall_accuracy", 0.0),
            "avg_consistency": reflect_results.get("avg_consistency", 0.0),
            "type_accuracies": reflect_results.get("type_accuracies", {}),
            "eval_results": {
                k: v["accuracy"] for k, v in verify_results.items()
            },
            "gate_passed": gate_passed,
            "regressed_benchmarks": regressed,
            "n_probes": len(probes),
            "bank_stats": reasoning_bank.stats(),
            "replay_buffer_size": len(replay_buffer.buffer),
            "reward_weights": get_reward_weights(epoch, args.epochs),
            "time_seconds": epoch_time,
            "model_path": current_model_path,
        }
        epoch_metrics.append(metrics)

        # Save epoch results
        epoch_dir = output_dir / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        with open(epoch_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Probe accuracy:  {metrics['probe_accuracy']:.1%}")
        print(f"    Avg consistency: {metrics['avg_consistency']:.2f}")
        print(f"    Eval accuracy:   {verify_results['_overall']['accuracy']:.1%}")
        print(f"    Gate:            {'PASS' if gate_passed else 'FAIL (rollback)'}")
        print(f"    Rules in bank:   {reasoning_bank.stats()['total_rules']}")
        print(f"    Replay buffer:   {len(replay_buffer.buffer)}")
        print(f"    Time:            {epoch_time:.0f}s")

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SEVA TRAINING COMPLETE")
    print("=" * 70)
    header = f"{'Ep':<4} {'Probe':>7} {'Consist':>8} {'Eval':>7} {'Gate':>6} {'Rules':>6} {'Replay':>7} {'Time':>6}"
    print(header)
    print("-" * 70)
    for m in epoch_metrics:
        eval_acc = m["eval_results"].get("_overall", 0.0)
        print(
            f"{m['epoch']:<4} {m['probe_accuracy']:>6.1%} "
            f"{m['avg_consistency']:>7.2f} {eval_acc:>6.1%} "
            f"{'OK' if m['gate_passed'] else 'FAIL':>6} "
            f"{m['bank_stats']['total_rules']:>6} "
            f"{m['replay_buffer_size']:>7} "
            f"{m['time_seconds']:>5.0f}s"
        )
    print("=" * 70)
    print(f"\nBest model: {best_model_path} ({best_overall_acc:.1%})")

    # Save full training log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "config": vars(args),
            "epochs": epoch_metrics,
            "best_model": best_model_path,
            "best_accuracy": best_overall_acc,
            "final_bank_stats": reasoning_bank.stats(),
        }, f, indent=2, default=str)

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
