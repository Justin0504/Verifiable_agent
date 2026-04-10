"""Evaluate SEVA v2 model on fact attribution benchmarks.

Extends eval_attribution.py with structured output evaluation:
  - Standard metrics: accuracy, macro F1
  - Alignment quality: are evidence spans actually grounded?
  - Chain quality: are reasoning steps non-trivial and coherent?
  - Diagnosis accuracy: correct error types for Not Attributable?
  - Calibration: ECE (Expected Calibration Error)

Usage:
    python scripts/eval_seva.py --model /path/to/checkpoint
    python scripts/eval_seva.py --model /path/to/checkpoint --benchmarks clearfacts grayfacts
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verifier.seva_format import SEVA_SYSTEM_PROMPT, SEVA_USER_TEMPLATE, ERROR_TYPES

# Import SEVA v2 reward components for scoring
sys.path.insert(0, str(Path(__file__).parent.parent / "drzero" / "verl" / "custom_reward"))
from seva_reward import (
    extract_json_from_response,
    normalize_label,
    score_alignment,
    score_chain,
    score_diagnosis,
)

# ============================================================
# Config
# ============================================================
DATA_DIR = Path("/Users/justin/Verifiable_agent/data/attribution")
RESULTS_DIR = Path("/Users/justin/Verifiable_agent/results/seva_eval")


# ============================================================
# Model
# ============================================================
def load_model(model_path: str):
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
                      max_new_tokens: int = 512) -> str:
    """Generate SEVA v2 structured verification response."""
    messages = [
        {"role": "system", "content": SEVA_SYSTEM_PROMPT},
        {"role": "user", "content": SEVA_USER_TEMPLATE.format(
            claim=claim, source=source[:2000]
        )},
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
# Structured output metrics
# ============================================================
def compute_alignment_groundedness(parsed: dict, claim: str, source: str) -> float:
    """Check if extracted spans actually appear in claim/source text."""
    alignment = parsed.get("evidence_alignment", [])
    if not alignment:
        return 0.0

    grounded = 0
    total = 0
    for entry in alignment:
        if not isinstance(entry, dict):
            continue
        total += 1
        claim_span = entry.get("claim_span", "")
        source_span = entry.get("source_span", "")

        claim_ok = claim_span.lower() in claim.lower() if claim_span else False
        source_ok = (
            source_span == "NOT_FOUND"
            or (source_span.lower() in source.lower() if source_span else False)
        )

        if claim_ok and source_ok:
            grounded += 1

    return grounded / max(total, 1)


def compute_chain_consistency(parsed: dict) -> float:
    """Check if reasoning chain judgments are consistent with final label."""
    chain = parsed.get("reasoning_chain", [])
    label = normalize_label(parsed.get("label", ""))
    if not chain or not label:
        return 0.0

    supported_count = sum(
        1 for s in chain
        if isinstance(s, dict) and s.get("judgment") == "supported"
    )
    total = sum(1 for s in chain if isinstance(s, dict) and s.get("judgment"))

    if total == 0:
        return 0.0

    support_rate = supported_count / total

    if label == "Attributable":
        # Should have mostly "supported" judgments
        return support_rate
    else:
        # Should have at least one "not_supported"
        return 1.0 - support_rate


def compute_ece(confidences: list[float], correct: list[bool],
                n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if not confidences:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_correct = defaultdict(list)
    bin_conf = defaultdict(list)

    for conf, corr in zip(confidences, correct):
        for i in range(n_bins):
            if bin_boundaries[i] <= conf < bin_boundaries[i + 1]:
                bin_correct[i].append(int(corr))
                bin_conf[i].append(conf)
                break
        else:
            # conf == 1.0
            bin_correct[n_bins - 1].append(int(corr))
            bin_conf[n_bins - 1].append(conf)

    ece = 0.0
    total = len(confidences)
    for i in range(n_bins):
        if bin_correct[i]:
            avg_conf = sum(bin_conf[i]) / len(bin_conf[i])
            avg_acc = sum(bin_correct[i]) / len(bin_correct[i])
            ece += abs(avg_conf - avg_acc) * len(bin_correct[i]) / total

    return ece


# ============================================================
# Evaluation
# ============================================================
def load_benchmark(name: str) -> list:
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
    """Run SEVA v2 evaluation with structured output metrics."""
    print(f"\nEvaluating on {benchmark_name} ({len(samples)} samples)...")

    preds, golds = [], []
    confidences, correct_list = [], []
    alignment_scores, chain_scores, diagnosis_scores = [], [], []
    groundedness_scores, consistency_scores = [], []
    format_errors = 0
    error_type_preds = []
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
            pred = "Not Attributable"
            conf = 0.5
            format_errors += 1
            a_score = c_score = d_score = g_score = cons_score = 0.0
        else:
            pred = normalize_label(parsed.get("label", "")) or "Not Attributable"
            conf = min(max(parsed.get("confidence", 0.5), 0.0), 1.0)

            a_score = score_alignment(parsed)
            c_score = score_chain(parsed)
            d_score = score_diagnosis(parsed, gold)
            g_score = compute_alignment_groundedness(parsed, claim, source)
            cons_score = compute_chain_consistency(parsed)

            et = parsed.get("error_type", "")
            if et:
                error_type_preds.append(et)

        is_correct = pred == gold
        preds.append(pred)
        golds.append(gold)
        confidences.append(conf)
        correct_list.append(is_correct)
        alignment_scores.append(a_score)
        chain_scores.append(c_score)
        diagnosis_scores.append(d_score)
        groundedness_scores.append(g_score)
        consistency_scores.append(cons_score)

        details.append({
            "claim": claim[:200],
            "gold": gold,
            "pred": pred,
            "confidence": conf,
            "correct": is_correct,
            "alignment_score": round(a_score, 3),
            "chain_score": round(c_score, 3),
            "groundedness": round(g_score, 3),
            "consistency": round(cons_score, 3),
            "response": response[:800],
        })

        if (i + 1) % 50 == 0:
            acc_so_far = sum(correct_list) / len(correct_list)
            print(f"  [{i+1}/{len(samples)}] Acc={acc_so_far:.1%}")

    # Standard metrics
    acc = accuracy_score(golds, preds)
    labels = ["Attributable", "Not Attributable"]
    f1_macro = f1_score(golds, preds, labels=labels, average="macro", zero_division=0)
    f1_per_class = {
        l: f1_score(golds, preds, labels=[l], average="macro", zero_division=0)
        for l in labels
    }
    cm = confusion_matrix(golds, preds, labels=labels)

    # Calibration
    ece = compute_ece(confidences, correct_list)

    # Structured output metrics
    avg = lambda xs: sum(xs) / max(len(xs), 1)

    results = {
        "benchmark": benchmark_name,
        "n_samples": len(preds),
        # Standard
        "accuracy": acc,
        "macro_f1": f1_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm.tolist(),
        "format_error_rate": format_errors / max(len(preds), 1),
        # Calibration
        "ece": ece,
        "avg_confidence_correct": avg([c for c, ok in zip(confidences, correct_list) if ok]),
        "avg_confidence_incorrect": avg([c for c, ok in zip(confidences, correct_list) if not ok]),
        # SEVA v2 structured metrics
        "avg_alignment_quality": avg(alignment_scores),
        "avg_chain_quality": avg(chain_scores),
        "avg_diagnosis_quality": avg(diagnosis_scores),
        "avg_groundedness": avg(groundedness_scores),
        "avg_chain_consistency": avg(consistency_scores),
        # Error type distribution
        "error_type_distribution": {
            et: error_type_preds.count(et)
            for et in set(error_type_preds)
        },
        "details": details,
    }

    # Print results
    print(f"  Accuracy:           {acc:.1%}")
    print(f"  Macro F1:           {f1_macro:.3f}")
    print(f"  F1 Attr:            {f1_per_class['Attributable']:.3f}")
    print(f"  F1 Not Attr:        {f1_per_class['Not Attributable']:.3f}")
    print(f"  ECE:                {ece:.3f}")
    print(f"  Format errors:      {format_errors}/{len(preds)}")
    print(f"  --- Structured Output Metrics ---")
    print(f"  Alignment quality:  {avg(alignment_scores):.3f}")
    print(f"  Chain quality:      {avg(chain_scores):.3f}")
    print(f"  Groundedness:       {avg(groundedness_scores):.3f}")
    print(f"  Chain consistency:  {avg(consistency_scores):.3f}")
    print(f"  Diagnosis quality:  {avg(diagnosis_scores):.3f}")
    print(f"  Confusion matrix:\n    {cm}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["clearfacts"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    model_name = Path(args.model).name
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model)

    if args.all:
        benchmark_files = sorted(DATA_DIR.glob("*.jsonl"))
        benchmarks = [f.stem for f in benchmark_files]
    else:
        benchmarks = args.benchmarks

    all_results = {}
    for bench_name in benchmarks:
        samples = load_benchmark(bench_name)
        if not samples:
            continue
        if args.max_samples:
            samples = samples[:args.max_samples]

        results = evaluate_benchmark(model, tokenizer, samples, bench_name)
        all_results[bench_name] = results

        with open(output_dir / f"{bench_name}_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Summary
    summary = {
        "model": args.model,
        "format": "seva_v2",
        "results": {
            name: {
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "ece": r["ece"],
                "alignment_quality": r["avg_alignment_quality"],
                "chain_quality": r["avg_chain_quality"],
                "groundedness": r["avg_groundedness"],
                "n_samples": r["n_samples"],
            }
            for name, r in all_results.items()
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SEVA v2 EVALUATION SUMMARY — {model_name}")
    print(f"{'='*80}")
    print(f"{'Benchmark':<20} {'N':>5} {'Acc':>7} {'F1':>7} {'ECE':>7} {'Align':>7} {'Chain':>7} {'Ground':>7}")
    print("-" * 80)
    for name, r in sorted(all_results.items()):
        print(f"{name:<20} {r['n_samples']:>5} "
              f"{r['accuracy']:>6.1%} {r['macro_f1']:>6.3f} {r['ece']:>6.3f} "
              f"{r['avg_alignment_quality']:>6.3f} {r['avg_chain_quality']:>6.3f} "
              f"{r['avg_groundedness']:>6.3f}")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
