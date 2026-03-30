"""Unified benchmark evaluation script.

Run the Verifier on any supported benchmark and compute standard metrics.

Usage:
    # Run on TruthfulQA with manual sample (no download)
    python scripts/run_benchmark.py --benchmark truthfulqa --manual --limit 30

    # Run on FEVER full dataset
    python scripts/run_benchmark.py --benchmark fever --split validation --limit 500

    # Run on all benchmarks with manual samples
    python scripts/run_benchmark.py --benchmark all --manual

    # Run on SciFact with a specific verifier model
    python scripts/run_benchmark.py --benchmark scifact --config configs/default.yaml

Supported benchmarks: truthfulqa, factscore, halueval, musique, scifact, fever, all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.benchmarks import (
    BenchmarkSample,
    FActScoreLoader,
    FEVERLoader,
    HaluEvalLoader,
    MuSiQueLoader,
    SciFactLoader,
    TruthfulQALoader,
)
from src.data.schema import AtomicClaim, ClaimLabel, Probe, Response, RiskType
from src.llm import create_llm
from src.tools.registry import ToolRegistry
from src.verifier.knowledge_base import KnowledgeBase
from src.verifier.scorer import Scorer
from src.verifier.verifier import Verifier

BENCHMARK_LOADERS = {
    "truthfulqa": TruthfulQALoader,
    "factscore": FActScoreLoader,
    "halueval": HaluEvalLoader,
    "musique": MuSiQueLoader,
    "scifact": SciFactLoader,
    "fever": FEVERLoader,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def samples_to_probes(samples: list[BenchmarkSample]) -> list[Probe]:
    """Convert benchmark samples to Probe objects."""
    probes = []
    for s in samples:
        risk_type = s.metadata.get("risk_type", "missing_evidence")
        try:
            rt = RiskType(risk_type)
        except ValueError:
            rt = RiskType.MISSING_EVIDENCE

        ground_truth = s.reference_answer
        if s.evidence:
            ground_truth = s.evidence[0] if not ground_truth else ground_truth

        probes.append(Probe(
            id=s.id,
            question=s.question,
            risk_type=rt,
            ground_truth=ground_truth or None,
            metadata={
                "benchmark": s.metadata.get("benchmark", ""),
                "gold_label": s.gold_label,
                **{k: v for k, v in s.metadata.items() if k not in ("benchmark", "gold_label")},
            },
        ))
    return probes


def evaluate_claim_level(
    samples: list[BenchmarkSample],
    predictions: list[list[dict]],
) -> dict:
    """Evaluate claim-level predictions against gold labels.

    For FActScore-style benchmarks with per-claim labels.
    """
    total = 0
    correct = 0
    confusion = Counter()

    for sample, preds in zip(samples, predictions):
        per_claim_labels = sample.metadata.get("per_claim_labels", [])
        if not per_claim_labels:
            continue

        for j, gold in enumerate(per_claim_labels):
            if j >= len(preds):
                break
            pred = preds[j].get("label", "N")
            confusion[(gold, pred)] += 1
            total += 1
            if gold == pred:
                correct += 1

    accuracy = correct / total if total else 0.0
    return {
        "claim_level_accuracy": accuracy,
        "claim_level_total": total,
        "claim_level_correct": correct,
        "confusion": dict(confusion),
    }


def evaluate_sample_level(
    samples: list[BenchmarkSample],
    sample_predictions: list[str],
) -> dict:
    """Evaluate sample-level predictions (majority vote of claims → single label)."""
    total = 0
    correct = 0
    per_label = {"S": {"tp": 0, "fp": 0, "fn": 0}, "C": {"tp": 0, "fp": 0, "fn": 0}, "N": {"tp": 0, "fp": 0, "fn": 0}}
    confusion = Counter()

    for sample, pred in zip(samples, sample_predictions):
        gold = sample.gold_label
        if gold == "mixed":
            continue  # Skip mixed-label samples for sample-level eval

        total += 1
        confusion[(gold, pred)] += 1

        if gold == pred:
            correct += 1
            per_label[gold]["tp"] += 1
        else:
            per_label[pred]["fp"] += 1
            per_label[gold]["fn"] += 1

    accuracy = correct / total if total else 0.0

    # Per-label P/R/F1
    label_metrics = {}
    for label, counts in per_label.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        label_metrics[label] = {"precision": precision, "recall": recall, "f1": f1}

    # Macro F1
    macro_f1 = sum(m["f1"] for m in label_metrics.values()) / len(label_metrics)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "total": total,
        "correct": correct,
        "per_label": label_metrics,
        "confusion": {f"{g}->{p}": c for (g, p), c in confusion.items()},
    }


def run_benchmark_eval(
    benchmark_name: str,
    samples: list[BenchmarkSample],
    verifier: Verifier,
    responder_llm=None,
) -> dict:
    """Run verifier on benchmark samples and compute metrics."""
    from src.responder.responder import Responder

    probes = samples_to_probes(samples)

    # If we have a responder, generate responses; otherwise use reference answers
    if responder_llm:
        responder = Responder(responder_llm)
        responses = responder.respond_batch(probes)
    else:
        # Use reference answers as responses (for direct verification)
        responses = []
        for s, p in zip(samples, probes):
            text = s.reference_answer
            if not text and s.claims:
                text = ". ".join(s.claims)
            if not text:
                text = s.question  # Fallback: verify the claim itself
            responses.append(Response(
                probe_id=p.id,
                model_name="benchmark_reference",
                text=text,
            ))

    # Run verification
    results = verifier.verify_batch(probes, responses, show_progress=True)

    # Collect predictions
    sample_predictions = []
    claim_predictions = []

    for result in results:
        # Sample-level: majority vote of claim labels
        label_counts = Counter()
        claim_preds = []
        for claim in result.claims:
            lbl = claim.label.value if claim.label else "N"
            label_counts[lbl] += 1
            claim_preds.append({"label": lbl, "confidence": claim.confidence})

        claim_predictions.append(claim_preds)

        if label_counts:
            majority = label_counts.most_common(1)[0][0]
        else:
            majority = "N"
        sample_predictions.append(majority)

    # Compute metrics
    sample_metrics = evaluate_sample_level(samples, sample_predictions)

    # Check if we have claim-level gold labels (FActScore style)
    has_claim_labels = any(s.metadata.get("per_claim_labels") for s in samples)
    claim_metrics = {}
    if has_claim_labels:
        claim_metrics = evaluate_claim_level(samples, claim_predictions)

    return {
        "benchmark": benchmark_name,
        "num_samples": len(samples),
        "sample_level": sample_metrics,
        "claim_level": claim_metrics if claim_metrics else None,
        "verification_results": [
            {
                "id": r.probe.id,
                "question": r.probe.question,
                "gold_label": samples[i].gold_label,
                "predicted_label": sample_predictions[i],
                "score": r.score,
                "num_claims": len(r.claims),
                "num_supported": r.num_supported,
                "num_contradicted": r.num_contradicted,
                "num_not_mentioned": r.num_not_mentioned,
            }
            for i, r in enumerate(results)
        ],
    }


def format_results(results: dict) -> str:
    """Format benchmark results for display."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Benchmark: {results['benchmark']}  ({results['num_samples']} samples)")
    lines.append(f"{'='*60}")

    sm = results["sample_level"]
    lines.append(f"\nSample-Level Metrics:")
    lines.append(f"  Accuracy:  {sm['accuracy']:.3f}")
    lines.append(f"  Macro F1:  {sm['macro_f1']:.3f}")
    lines.append(f"  Correct:   {sm['correct']}/{sm['total']}")

    lines.append(f"\n  Per-Label P/R/F1:")
    for label in ["S", "C", "N"]:
        m = sm["per_label"].get(label, {})
        lines.append(f"    {label}: P={m.get('precision', 0):.3f}  R={m.get('recall', 0):.3f}  F1={m.get('f1', 0):.3f}")

    lines.append(f"\n  Confusion Matrix:")
    for key, count in sorted(sm.get("confusion", {}).items()):
        lines.append(f"    {key}: {count}")

    if results.get("claim_level"):
        cm = results["claim_level"]
        lines.append(f"\nClaim-Level Metrics:")
        lines.append(f"  Accuracy:  {cm['claim_level_accuracy']:.3f}")
        lines.append(f"  Total:     {cm['claim_level_total']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument("--benchmark", required=True,
                        choices=list(BENCHMARK_LOADERS.keys()) + ["all"],
                        help="Benchmark to evaluate")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config file for verifier LLM")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Max samples")
    parser.add_argument("--manual", action="store_true",
                        help="Use manual curated samples (no download)")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--with-responder", action="store_true",
                        help="Generate responses with responder LLM instead of using reference answers")
    args = parser.parse_args()

    config = load_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or f"results/benchmark_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize verifier
    verifier_llm = create_llm(config["verifier_llm"])
    kb = KnowledgeBase(
        path=config.get("knowledge_base", {}).get("path", "knowledge_base/documents"),
    )
    kb.load()
    scorer = Scorer()

    tools_config = config.get("tools", {})
    tool_registry = None
    if tools_config.get("enabled", True):
        tool_registry = ToolRegistry(
            enable_web=tools_config.get("web_search", True),
            enable_wikidata=tools_config.get("wikidata", True),
            enable_calculator=tools_config.get("calculator", True),
            enable_wikipedia=tools_config.get("wikipedia", True),
            enable_scholar=tools_config.get("semantic_scholar", True),
            enable_code_executor=tools_config.get("code_executor", True),
        )

    verifier = Verifier(verifier_llm, kb, scorer, tool_registry=tool_registry)

    responder_llm = None
    if args.with_responder and config.get("responder_models"):
        responder_llm = create_llm(config["responder_models"][0])

    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmarks = list(BENCHMARK_LOADERS.keys())
    else:
        benchmarks = [args.benchmark]

    all_results = {}

    for bm_name in benchmarks:
        print(f"\n{'#'*60}")
        print(f"# Running benchmark: {bm_name}")
        print(f"{'#'*60}")

        loader_cls = BENCHMARK_LOADERS[bm_name]
        loader = loader_cls()

        if args.manual:
            samples = loader.load_manual_sample(limit=args.limit or 50)
        else:
            try:
                samples = loader.load(split=args.split, limit=args.limit)
            except Exception as e:
                print(f"  Failed to load from HuggingFace ({e}), falling back to manual...")
                samples = loader.load_manual_sample(limit=args.limit or 50)

        print(f"  Loaded {len(samples)} samples")

        # Label distribution
        label_dist = Counter(s.gold_label for s in samples)
        print(f"  Label distribution: {dict(label_dist)}")

        results = run_benchmark_eval(bm_name, samples, verifier, responder_llm)
        all_results[bm_name] = results

        print(format_results(results))

        # Save per-benchmark results
        with open(f"{output_dir}/{bm_name}_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Save combined results
    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Benchmark':<15} {'Samples':>8} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'-'*15} {'-'*8} {'-'*10} {'-'*10}")
    for bm_name, results in all_results.items():
        sm = results["sample_level"]
        n = results["num_samples"]
        print(f"{bm_name:<15} {n:>8} {sm['accuracy']:>10.3f} {sm['macro_f1']:>10.3f}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
