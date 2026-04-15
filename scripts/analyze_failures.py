"""Analyze SEVA model failures by error type and pattern.

Reads evaluation results JSON and produces a weakness profile:
  - Per-error-type accuracy
  - Confusion matrix by error type
  - Hardest samples (for replay buffer)
  - Weakness profile JSON (input to adversarial generation)

Usage:
    python scripts/analyze_failures.py \
        --results-file results/seva_eval/seva_grpo_step350/clearfacts_results.json \
        --output-dir results/seva_eval/seva_grpo_step350/analysis
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "drzero" / "verl" / "custom_reward"))

from seva_reward import extract_json_from_response, normalize_label, VALID_ERROR_TYPES


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyze_by_error_type(details: list) -> dict:
    """Analyze accuracy broken down by predicted/gold error type."""
    # For incorrect predictions, what error types are involved?
    error_type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "samples": []})

    for i, d in enumerate(details):
        response = d.get("response", "")
        parsed = extract_json_from_response(response)
        gold = d["gold"]
        pred = d["pred"]
        correct = d["correct"]

        # Extract error type from model output
        error_type = None
        if parsed:
            error_type = parsed.get("error_type", "")
            if error_type and error_type not in VALID_ERROR_TYPES:
                error_type = f"other:{error_type}"

        # Categorize by what KIND of sample this is
        if gold == "Not Attributable":
            # For NA samples, we want to know which error types the model struggles with
            category = error_type if error_type else "no_error_type_predicted"
        else:
            category = "attributable"

        error_type_stats[category]["total"] += 1
        if correct:
            error_type_stats[category]["correct"] += 1
        else:
            error_type_stats[category]["samples"].append({
                "index": i,
                "claim": d.get("claim", ""),
                "gold": gold,
                "pred": pred,
                "confidence": d.get("confidence", 0),
                "error_type": error_type,
            })

    return dict(error_type_stats)


def analyze_confusion_patterns(details: list) -> dict:
    """Analyze false positive vs false negative patterns."""
    fp = []  # predicted Attributable, gold Not Attributable
    fn = []  # predicted Not Attributable, gold Attributable
    tp = 0
    tn = 0

    for i, d in enumerate(details):
        gold = d["gold"]
        pred = d["pred"]
        response = d.get("response", "")
        parsed = extract_json_from_response(response)

        if gold == "Attributable" and pred == "Attributable":
            tp += 1
        elif gold == "Not Attributable" and pred == "Not Attributable":
            tn += 1
        elif gold == "Not Attributable" and pred == "Attributable":
            fp.append({
                "index": i,
                "claim": d.get("claim", ""),
                "confidence": d.get("confidence", 0),
                "error_type": parsed.get("error_type", "") if parsed else "",
                "response_snippet": response[:300],
            })
        elif gold == "Attributable" and pred == "Not Attributable":
            fn.append({
                "index": i,
                "claim": d.get("claim", ""),
                "confidence": d.get("confidence", 0),
                "error_type": parsed.get("error_type", "") if parsed else "",
                "response_snippet": response[:300],
            })

    return {
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "fp_rate": len(fp) / max(len(fp) + tn, 1),
        "fn_rate": len(fn) / max(len(fn) + tp, 1),
        "fp_samples": fp[:20],  # top 20 for inspection
        "fn_samples": fn[:20],
    }


def analyze_confidence(details: list) -> dict:
    """Analyze confidence calibration by correctness."""
    correct_confs = []
    incorrect_confs = []
    high_conf_wrong = []

    for i, d in enumerate(details):
        conf = d.get("confidence", 0.5)
        if d["correct"]:
            correct_confs.append(conf)
        else:
            incorrect_confs.append(conf)
            if conf > 0.8:
                high_conf_wrong.append({
                    "index": i,
                    "claim": d.get("claim", ""),
                    "gold": d["gold"],
                    "pred": d["pred"],
                    "confidence": conf,
                })

    avg = lambda xs: sum(xs) / max(len(xs), 1)

    return {
        "avg_confidence_correct": round(avg(correct_confs), 3),
        "avg_confidence_incorrect": round(avg(incorrect_confs), 3),
        "high_confidence_wrong_count": len(high_conf_wrong),
        "high_confidence_wrong_samples": high_conf_wrong[:10],
    }


def analyze_structural_failures(details: list) -> dict:
    """Analyze where structural output quality fails."""
    low_alignment = []
    low_chain = []
    inconsistent = []
    format_errors = 0

    for i, d in enumerate(details):
        response = d.get("response", "")
        parsed = extract_json_from_response(response)

        if parsed is None:
            format_errors += 1
            continue

        a = d.get("alignment_score", 0)
        c = d.get("chain_score", 0)
        cons = d.get("consistency", 0)

        if a < 0.5:
            low_alignment.append({"index": i, "score": a, "claim": d.get("claim", "")[:100]})
        if c < 0.5:
            low_chain.append({"index": i, "score": c, "claim": d.get("claim", "")[:100]})
        if cons < 0.3:
            inconsistent.append({"index": i, "score": cons, "claim": d.get("claim", "")[:100],
                                 "gold": d["gold"], "pred": d["pred"]})

    return {
        "format_errors": format_errors,
        "low_alignment_count": len(low_alignment),
        "low_chain_count": len(low_chain),
        "inconsistent_count": len(inconsistent),
        "low_alignment_samples": low_alignment[:10],
        "low_chain_samples": low_chain[:10],
        "inconsistent_samples": inconsistent[:10],
    }


def build_weakness_profile(error_type_stats: dict, confusion: dict,
                           confidence: dict, structural: dict) -> dict:
    """Build a weakness profile for adversarial data generation.

    This is the KEY output: it tells the Proposer what to target.
    """
    # Per error type: accuracy and priority
    error_type_accuracy = {}
    for etype, stats in error_type_stats.items():
        if etype in ("attributable", "no_error_type_predicted"):
            continue
        if stats["total"] >= 3:  # need at least 3 samples
            acc = stats["correct"] / stats["total"]
            error_type_accuracy[etype] = {
                "accuracy": round(acc, 3),
                "total": stats["total"],
                "priority": round(1.0 - acc, 3),  # lower accuracy = higher priority
            }

    # Sort by priority (worst first)
    sorted_types = sorted(error_type_accuracy.items(),
                          key=lambda x: x[1]["priority"], reverse=True)

    # Overall weakness summary
    profile = {
        "overall_accuracy": None,  # filled by caller
        "overall_f1": None,

        # Error type weaknesses (sorted worst first)
        "error_type_weaknesses": dict(sorted_types),

        # Confusion pattern
        "bias": "false_positive" if confusion["fp_rate"] > confusion["fn_rate"] else "false_negative",
        "fp_rate": confusion["fp_rate"],
        "fn_rate": confusion["fn_rate"],

        # Calibration issues
        "overconfident_wrong": confidence["high_confidence_wrong_count"],
        "avg_conf_gap": round(
            confidence["avg_confidence_correct"] - confidence["avg_confidence_incorrect"], 3
        ),

        # Structural issues
        "format_error_rate": structural["format_errors"],
        "structural_issues": {
            "low_alignment": structural["low_alignment_count"],
            "low_chain": structural["low_chain_count"],
            "inconsistent": structural["inconsistent_count"],
        },

        # Targeting weights for adversarial generation
        "targeting_weights": {},
    }

    # Compute targeting weights: how much of the next adversarial batch
    # should focus on each error type
    total_priority = sum(v["priority"] for v in error_type_accuracy.values()) or 1.0
    for etype, stats in sorted_types:
        profile["targeting_weights"][etype] = round(
            stats["priority"] / total_priority, 3
        )

    # Add weights for FP/FN balance
    if confusion["fp_rate"] > confusion["fn_rate"]:
        profile["targeting_weights"]["safe_attribution"] = round(
            min(confusion["fp_rate"], 0.3), 3
        )
    else:
        profile["targeting_weights"]["subtle_non_attribution"] = round(
            min(confusion["fn_rate"], 0.3), 3
        )

    return profile


def extract_hard_samples(details: list, top_k: int = 200) -> list:
    """Extract hardest samples for replay buffer.

    Hard = incorrect AND high confidence (model was confidently wrong).
    """
    hard = []
    for i, d in enumerate(details):
        if not d["correct"]:
            hard.append({
                "index": i,
                "claim": d.get("claim", ""),
                "gold": d["gold"],
                "pred": d["pred"],
                "confidence": d.get("confidence", 0.5),
                "difficulty_score": d.get("confidence", 0.5),  # higher conf = harder
            })

    hard.sort(key=lambda x: x["difficulty_score"], reverse=True)
    return hard[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, required=True,
                        help="Path to *_results.json from eval_seva.py")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for analysis")
    parser.add_argument("--top-hard", type=int, default=200,
                        help="Number of hard samples for replay buffer")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    details = results.get("details", [])
    print(f"  {len(details)} samples loaded")

    if not details:
        print("ERROR: No details found in results file.")
        print("  Make sure eval_seva.py was run with detail logging.")
        sys.exit(1)

    # Run all analyses
    print("\n1. Analyzing by error type...")
    error_type_stats = analyze_by_error_type(details)
    for etype, stats in sorted(error_type_stats.items(),
                                key=lambda x: x[1]["total"], reverse=True):
        acc = stats["correct"] / max(stats["total"], 1)
        print(f"  {etype:30s}  {stats['correct']:>3d}/{stats['total']:>3d}  ({acc:.1%})")

    print("\n2. Analyzing confusion patterns...")
    confusion = analyze_confusion_patterns(details)
    print(f"  TP={confusion['true_positives']:>4d}  FP={confusion['false_positives']:>4d}")
    print(f"  FN={confusion['false_negatives']:>4d}  TN={confusion['true_negatives']:>4d}")
    print(f"  FP rate: {confusion['fp_rate']:.1%}  FN rate: {confusion['fn_rate']:.1%}")
    print(f"  Bias: {('FP-heavy' if confusion['fp_rate'] > confusion['fn_rate'] else 'FN-heavy')}")

    print("\n3. Analyzing confidence calibration...")
    confidence = analyze_confidence(details)
    print(f"  Avg confidence (correct):   {confidence['avg_confidence_correct']}")
    print(f"  Avg confidence (incorrect): {confidence['avg_confidence_incorrect']}")
    print(f"  High-conf wrong: {confidence['high_confidence_wrong_count']}")

    print("\n4. Analyzing structural quality...")
    structural = analyze_structural_failures(details)
    print(f"  Format errors:  {structural['format_errors']}")
    print(f"  Low alignment:  {structural['low_alignment_count']}")
    print(f"  Low chain:      {structural['low_chain_count']}")
    print(f"  Inconsistent:   {structural['inconsistent_count']}")

    print("\n5. Building weakness profile...")
    profile = build_weakness_profile(error_type_stats, confusion, confidence, structural)
    profile["overall_accuracy"] = results.get("accuracy")
    profile["overall_f1"] = results.get("macro_f1")
    print(f"  Overall: acc={profile['overall_accuracy']}, F1={profile['overall_f1']}")
    print(f"  Bias: {profile['bias']}")
    print(f"  Targeting weights:")
    for etype, w in sorted(profile["targeting_weights"].items(),
                           key=lambda x: x[1], reverse=True):
        print(f"    {etype:30s}  {w:.3f}")

    print("\n6. Extracting hard samples...")
    hard_samples = extract_hard_samples(details, top_k=args.top_hard)
    print(f"  {len(hard_samples)} hard samples extracted")

    # Save outputs
    with open(output_dir / "error_type_analysis.json", "w") as f:
        # Remove sample lists for clean summary
        summary = {k: {"total": v["total"], "correct": v["correct"],
                        "accuracy": round(v["correct"] / max(v["total"], 1), 3),
                        "n_failures": len(v["samples"])}
                   for k, v in error_type_stats.items()}
        json.dump(summary, f, indent=2)

    with open(output_dir / "confusion_analysis.json", "w") as f:
        json.dump(confusion, f, indent=2, default=str)

    with open(output_dir / "confidence_analysis.json", "w") as f:
        json.dump(confidence, f, indent=2, default=str)

    with open(output_dir / "structural_analysis.json", "w") as f:
        json.dump(structural, f, indent=2, default=str)

    with open(output_dir / "weakness_profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    with open(output_dir / "hard_samples.json", "w") as f:
        json.dump(hard_samples, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Analysis saved to: {output_dir}")
    print(f"  - error_type_analysis.json")
    print(f"  - confusion_analysis.json")
    print(f"  - confidence_analysis.json")
    print(f"  - structural_analysis.json")
    print(f"  - weakness_profile.json    <-- input for generate_adversarial.py")
    print(f"  - hard_samples.json        <-- replay buffer for GRPO")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
