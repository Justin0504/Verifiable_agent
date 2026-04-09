"""Evaluate a model on fact attribution benchmarks.

Supports: ClearFacts, GrayFacts, LLM-AggreFact subsets, CoverBench, SciFact, HoVer

Usage:
    python scripts/eval_attribution.py --model /path/to/checkpoint
    python scripts/eval_attribution.py --model /path/to/checkpoint --benchmarks clearfacts grayfacts
    python scripts/eval_attribution.py --model /path/to/checkpoint --all
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Reuse reward function's parser
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "drzero" / "verl" / "custom_reward"))
from attribution_reward import extract_json_from_response, normalize_label


# ============================================================
# Config
# ============================================================
DATA_DIR = Path("/Users/justin/Verifiable_agent/data/attribution")
RESULTS_DIR = Path("/Users/justin/Verifiable_agent/results/attribution_eval")

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


# ============================================================
# Model loading
# ============================================================
def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, claim: str, source: str,
                      max_new_tokens: int = 256) -> str:
    """Generate attribution verification response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            claim=claim, source=source[:2000]  # truncate long sources
        )}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
    return response


# ============================================================
# Evaluation
# ============================================================
def load_benchmark(name: str) -> list:
    """Load a benchmark JSONL file."""
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def evaluate_benchmark(model, tokenizer, samples: list,
                       benchmark_name: str) -> dict:
    """Run evaluation on a benchmark."""
    print(f"\nEvaluating on {benchmark_name} ({len(samples)} samples)...")

    preds = []
    golds = []
    confidences = []
    format_errors = 0
    details = []

    for i, sample in enumerate(samples):
        claim = sample.get("claim", "")
        source = sample.get("source", "")
        gold = sample.get("gold_label", "Not Attributable")

        if not claim or not source:
            continue

        response = generate_response(model, tokenizer, claim, source)

        parsed = extract_json_from_response(response)
        if parsed is None:
            pred = "Not Attributable"  # default on parse failure
            conf = 0.5
            format_errors += 1
        else:
            pred = normalize_label(parsed.get("label", "")) or "Not Attributable"
            conf = parsed.get("confidence", 0.5)

        preds.append(pred)
        golds.append(gold)
        confidences.append(conf)

        details.append({
            "claim": claim[:200],
            "gold": gold,
            "pred": pred,
            "confidence": conf,
            "correct": pred == gold,
            "response": response[:500],
        })

        if (i + 1) % 50 == 0:
            acc_so_far = sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)
            print(f"  [{i+1}/{len(samples)}] Accuracy so far: {acc_so_far:.1%}")

    # Compute metrics
    acc = accuracy_score(golds, preds)
    labels = ["Attributable", "Not Attributable"]
    f1_macro = f1_score(golds, preds, labels=labels, average="macro", zero_division=0)
    f1_per_class = {
        l: f1_score(golds, preds, labels=[l], average="macro", zero_division=0)
        for l in labels
    }

    # Confusion matrix
    cm = confusion_matrix(golds, preds, labels=labels)

    # Calibration: average confidence when correct vs incorrect
    correct_confs = [c for c, p, g in zip(confidences, preds, golds) if p == g]
    incorrect_confs = [c for c, p, g in zip(confidences, preds, golds) if p != g]

    results = {
        "benchmark": benchmark_name,
        "n_samples": len(preds),
        "accuracy": acc,
        "macro_f1": f1_macro,
        "f1_per_class": f1_per_class,
        "format_error_rate": format_errors / max(len(preds), 1),
        "confusion_matrix": cm.tolist(),
        "avg_confidence_correct": sum(correct_confs) / max(len(correct_confs), 1),
        "avg_confidence_incorrect": sum(incorrect_confs) / max(len(incorrect_confs), 1),
        "details": details,
    }

    print(f"  Accuracy: {acc:.1%}")
    print(f"  Macro F1: {f1_macro:.3f}")
    print(f"  F1 Attributable: {f1_per_class['Attributable']:.3f}")
    print(f"  F1 Not Attributable: {f1_per_class['Not Attributable']:.3f}")
    print(f"  Format errors: {format_errors}/{len(preds)}")
    print(f"  Confusion matrix:\n    {cm}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["clearfacts"],
                        help="Which benchmarks to evaluate on")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate on all available benchmarks")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per benchmark (for quick testing)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Determine output directory
    model_name = Path(args.model).name
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model)

    # Determine benchmarks
    if args.all:
        benchmark_files = sorted(DATA_DIR.glob("*.jsonl"))
        benchmarks = [f.stem for f in benchmark_files]
    else:
        benchmarks = args.benchmarks

    # Evaluate each benchmark
    all_results = {}
    for bench_name in benchmarks:
        samples = load_benchmark(bench_name)
        if not samples:
            continue

        if args.max_samples:
            samples = samples[:args.max_samples]

        results = evaluate_benchmark(model, tokenizer, samples, bench_name)
        all_results[bench_name] = results

        # Save per-benchmark results
        with open(output_dir / f"{bench_name}_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Save summary
    summary = {
        "model": args.model,
        "results": {
            name: {
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "n_samples": r["n_samples"],
            }
            for name, r in all_results.items()
        },
        "overall_accuracy": (
            sum(r["accuracy"] * r["n_samples"] for r in all_results.values()) /
            max(sum(r["n_samples"] for r in all_results.values()), 1)
        ),
        "overall_macro_f1": (
            sum(r["macro_f1"] for r in all_results.values()) /
            max(len(all_results), 1)
        ),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY — {model_name}")
    print("=" * 70)
    print(f"{'Benchmark':<30} {'N':>6} {'Acc':>8} {'F1':>8}")
    print("-" * 70)
    for name, r in sorted(all_results.items()):
        print(f"{name:<30} {r['n_samples']:>6} {r['accuracy']:>7.1%} {r['macro_f1']:>7.3f}")
    print("-" * 70)
    print(f"{'OVERALL':<30} {'':>6} {summary['overall_accuracy']:>7.1%} {summary['overall_macro_f1']:>7.3f}")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
