"""Run our Verifier Agent on benchmark data for direct comparison with baselines.

This bridges the gap between:
- run_baselines.py (6 static baselines on benchmark data)
- run_experiment.py (full self-evolution pipeline with self-generated probes)

Here we run the Verifier on the SAME benchmark samples used by baselines,
so accuracy is directly comparable. Supports:
1. Single-epoch mode: one-shot verification (like a baseline)
2. Multi-epoch mode: evolve across epochs, measure accuracy after each

Usage:
    # Single epoch on FEVER
    python scripts/run_verifier_on_benchmarks.py --benchmarks fever --limit 100

    # Multi-epoch evolution with accuracy curve
    python scripts/run_verifier_on_benchmarks.py --benchmarks fever --epochs 5 --limit 200

    # Full comparison (same config as baselines)
    python scripts/run_verifier_on_benchmarks.py --benchmarks fever,truthfulqa --epochs 3 --limit 500

    # With majority voting (R-Zero)
    python scripts/run_verifier_on_benchmarks.py --benchmarks fever --majority-vote 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from src.baselines.base import BaseBaseline, BaselineResult
from src.benchmarks import (
    FActScoreLoader,
    FEVERLoader,
    HaluEvalLoader,
    MuSiQueLoader,
    SciFactLoader,
    TruthfulQALoader,
)
from src.benchmarks.base import BenchmarkSample
from src.data.schema import AtomicClaim, ClaimLabel, Probe, Response, RiskType, VerificationResult
from src.evolution.failure_extractor import FailureExtractor, InformativeFailure
from src.evolution.memory_store import MemoryStore
from src.evolution.reasoning_bank import ReasoningBank
from src.llm.openai_llm import OpenAILLM
from src.verifier.decomposer import Decomposer
from src.verifier.evidence_matcher import EvidenceMatcher
from src.verifier.knowledge_base import KnowledgeBase
from src.verifier.scorer import Scorer
from src.verifier.verifier import Verifier

BENCHMARK_LOADERS = {
    "fever": FEVERLoader,
    "truthfulqa": TruthfulQALoader,
    "scifact": SciFactLoader,
    "halueval": HaluEvalLoader,
    "musique": MuSiQueLoader,
    "factscore": FActScoreLoader,
}


def load_samples(benchmark_name: str, manual: bool, limit: int | None, split: str) -> list[BenchmarkSample]:
    loader_cls = BENCHMARK_LOADERS[benchmark_name]
    loader = loader_cls()
    if manual:
        return loader.load_manual_sample(limit=limit or 50)
    try:
        return loader.load(split=split, limit=limit)
    except Exception as e:
        print(f"  HuggingFace load failed ({e}), falling back to manual...")
        return loader.load_manual_sample(limit=limit or 50)


def sample_to_probe_response(sample: BenchmarkSample) -> tuple[Probe, Response, str]:
    """Convert a BenchmarkSample into Probe + Response for our Verifier.

    For benchmark evaluation, the "response" IS the claim text itself
    (or the reference answer), and evidence comes from the sample.
    """
    # Build evidence string
    evidence_text = "\n\n".join(sample.evidence[:5]) if sample.evidence else ""

    # The "question" is the claim or question from the benchmark
    probe = Probe(
        id=sample.id,
        question=sample.question,
        risk_type=RiskType.MISSING_EVIDENCE,  # Default; doesn't affect verification
        ground_truth=sample.reference_answer or None,
        metadata={"benchmark_gold": sample.gold_label},
    )

    # For NLI-style benchmarks (FEVER), the claim IS the thing to verify
    # The "response" from the model is effectively the claim itself
    response_text = sample.question
    if sample.claims:
        response_text = ". ".join(sample.claims)

    response = Response(
        probe_id=sample.id,
        model_name="benchmark_sample",
        text=response_text,
    )

    return probe, response, evidence_text


def verifier_predict(
    verifier: Verifier,
    sample: BenchmarkSample,
    evidence_override: str = "",
) -> BaselineResult:
    """Run our Verifier on a single benchmark sample and return a BaselineResult."""
    probe, response, sample_evidence = sample_to_probe_response(sample)

    # Use sample evidence (same as baselines for fairness)
    if evidence_override:
        evidence = evidence_override
    elif sample_evidence:
        evidence = sample_evidence
    else:
        evidence = ""

    # Decompose
    claims_text = sample.claims if sample.claims else [sample.question]
    claims = [AtomicClaim(id=f"{sample.id}_c{i}", text=c) for i, c in enumerate(claims_text)]

    # If we have ReasoningBank rules, inject them
    if verifier.reasoning_bank:
        for claim in claims:
            relevant_rules = verifier.reasoning_bank.retrieve_relevant(claim.text)
            if relevant_rules:
                rules_text = verifier.reasoning_bank.format_for_prompt(relevant_rules)
                if not claim.metadata:
                    claim.metadata = {}
                claim.metadata["reasoning_rules"] = rules_text
                claim.metadata["rule_ids"] = [r.id for r in relevant_rules]

    # Verify each claim — use majority vote if configured
    if verifier.majority_vote_n > 1:
        verified = verifier._verify_majority_vote(claims, evidence)
    else:
        verified = verifier.matcher.match_batch(claims, evidence)

    # Aggregate to sample-level prediction
    claim_labels = []
    claim_confidences = []
    for c in verified:
        label = c.label.value if c.label else "N"
        claim_labels.append(label)
        claim_confidences.append(c.confidence)

    # Aggregate: if any C → C, if all S → S, else N
    if "C" in claim_labels:
        predicted = "C"
    elif all(l == "S" for l in claim_labels):
        predicted = "S"
    else:
        predicted = "N"

    avg_conf = sum(claim_confidences) / len(claim_confidences) if claim_confidences else 0.5

    per_claim_golds = sample.metadata.get("per_claim_labels", [])
    claim_gold_labels = per_claim_golds if per_claim_golds else [sample.gold_label] * len(claims_text)

    # Record rule usage feedback using gold labels (not just confidence)
    if verifier.reasoning_bank:
        for claim_obj, pred_label in zip(verified, claim_labels):
            if claim_obj.metadata:
                rule_ids = claim_obj.metadata.get("rule_ids", [])
                was_correct = pred_label == sample.gold_label
                for rid in rule_ids:
                    verifier.reasoning_bank.record_usage(rid, was_correct)

    return BaselineResult(
        sample_id=sample.id,
        predicted_label=predicted,
        gold_label=sample.gold_label,
        confidence=avg_conf,
        claims=[{"claim": c.text, "label": l} for c, l in zip(verified, claim_labels)],
        claim_labels=claim_labels,
        claim_gold_labels=claim_gold_labels,
        metadata={"method": "verifier_agent"},
    )


def run_single_epoch(
    verifier: Verifier,
    samples: list[BenchmarkSample],
    epoch: int,
) -> tuple[list[BaselineResult], dict]:
    """Run Verifier on all samples for one epoch."""
    results = []
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(samples)}]...")
        result = verifier_predict(verifier, sample)
        results.append(result)

    metrics = BaseBaseline.compute_metrics(results)
    return results, metrics


def extract_failures_from_results(
    results: list[BaselineResult],
    samples: list[BenchmarkSample],
) -> list[InformativeFailure]:
    """Extract failure cases for evolution (using gold labels from benchmark)."""
    failures = []
    for result, sample in zip(results, samples):
        if result.predicted_label != result.gold_label:
            # Wrong prediction = informative failure
            if result.predicted_label == "C" and result.gold_label == "S":
                ftype = "false_positive"
            elif result.predicted_label == "S" and result.gold_label == "C":
                ftype = "false_negative"
            else:
                ftype = "boundary"

            failures.append(InformativeFailure(
                failure_type=ftype,
                probe_question=sample.question,
                response_text=sample.question[:200],
                claim_text=sample.claims[0] if sample.claims else sample.question,
                label=result.predicted_label,
                confidence=result.confidence,
                pattern=f"Predicted {result.predicted_label} but gold is {result.gold_label}",
                risk_type="benchmark",
            ))
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Verifier Agent on benchmarks")
    parser.add_argument("--benchmarks", default="fever", help="Comma-separated benchmark names")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of self-evolution epochs")
    parser.add_argument("--majority-vote", type=int, default=1, help="Majority vote N (1=off, 3=triple)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fresh", action="store_true", help="Start with fresh memory")
    args = parser.parse_args()

    benchmark_names = [b.strip() for b in args.benchmarks.split(",")]

    # Create LLM
    llm = OpenAILLM(model=args.model, temperature=0.2)
    print(f"Using {args.provider}/{args.model}")
    print(f"Epochs: {args.epochs}, Majority vote: {args.majority_vote}x\n")

    # Initialize components
    memory_dir = "memory/benchmark_eval"
    memory = MemoryStore(path=memory_dir)
    if args.fresh:
        memory.reset()
        # Also clear ReasoningBank file for clean evolution curve
        rb_path = Path(f"{memory_dir}/reasoning_bank.json")
        if rb_path.exists():
            rb_path.unlink()
        print("Fresh start: cleared all memory\n")

    kb = KnowledgeBase(path="knowledge_base/documents")
    kb.load()
    print(f"KB: {len(kb)} documents\n")

    scorer = Scorer()
    reasoning_bank = ReasoningBank(path=f"{memory_dir}/reasoning_bank.json", llm=llm)
    print(f"ReasoningBank: {reasoning_bank.stats()}\n")

    verifier = Verifier(
        llm, kb, scorer,
        memory=memory,
        reasoning_bank=reasoning_bank,
        majority_vote_n=args.majority_vote,
    )

    # Load saved evolution state
    if not args.fresh:
        saved_matcher = memory.load_matcher_few_shots()
        saved_decomposer = memory.load_decomposer_few_shots()
        if saved_matcher:
            verifier.matcher.few_shot_examples = saved_matcher
            print(f"Loaded {len(saved_matcher)} matcher few-shots")
        if saved_decomposer:
            verifier.decomposer.few_shot_examples = saved_decomposer
            print(f"Loaded {len(saved_decomposer)} decomposer few-shots")
        saved_cal = memory.load_calibration_correction()
        if saved_cal:
            verifier.matcher.calibration_correction = saved_cal
            print("Loaded calibration correction")

    failure_extractor = FailureExtractor()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or f"results/verifier_agent_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for bench_name in benchmark_names:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")

        samples = load_samples(bench_name, args.manual, args.limit, args.split)
        # Filter to S/C/N labels only
        samples = [s for s in samples if s.gold_label in ("S", "C", "N")]
        print(f"  Loaded {len(samples)} samples")

        if not samples:
            print("  No samples, skipping...")
            continue

        bench_results = {"epochs": []}

        # Split: use 80% for evolution, 20% for held-out evaluation
        if args.epochs > 1 and len(samples) > 20:
            n_eval = max(int(len(samples) * 0.2), 10)
            eval_samples = samples[:n_eval]
            train_samples = samples[n_eval:]
            print(f"  Split: {len(train_samples)} train, {len(eval_samples)} eval")
        else:
            train_samples = samples
            eval_samples = samples

        for epoch in range(args.epochs):
            print(f"\n  --- Epoch {epoch + 1}/{args.epochs} ---")
            verifier._epoch = epoch

            # Run on training samples
            print(f"  Verifying {len(train_samples)} samples...")
            t0 = time.time()
            results, metrics = run_single_epoch(verifier, train_samples, epoch)
            elapsed = time.time() - t0

            print(f"    Accuracy: {metrics['accuracy']:.1%}")
            print(f"    Macro F1: {metrics['macro_f1']:.3f}")
            print(f"    Time: {elapsed:.1f}s")

            # Run on held-out eval samples (if multi-epoch)
            eval_metrics = metrics
            if eval_samples is not train_samples:
                print(f"  Evaluating on {len(eval_samples)} held-out samples...")
                eval_results, eval_metrics = run_single_epoch(verifier, eval_samples, epoch)
                print(f"    Eval Accuracy: {eval_metrics['accuracy']:.1%}")
                print(f"    Eval Macro F1: {eval_metrics['macro_f1']:.3f}")

            # Collect evolution state for this epoch
            rb_stats = reasoning_bank.stats() if reasoning_bank else {}
            epoch_record = {
                "epoch": epoch,
                "train_accuracy": metrics["accuracy"],
                "train_macro_f1": metrics["macro_f1"],
                "eval_accuracy": eval_metrics["accuracy"],
                "eval_macro_f1": eval_metrics["macro_f1"],
                "n_train": len(train_samples),
                "n_eval": len(eval_samples),
                "elapsed_s": elapsed,
                # Evolution state tracking
                "reasoning_bank_rules": rb_stats.get("total_rules", 0),
                "reasoning_bank_success": rb_stats.get("success_rules", 0),
                "reasoning_bank_failure": rb_stats.get("failure_rules", 0),
                "reasoning_bank_linked": rb_stats.get("linked_rules", 0),
                "reasoning_bank_effectiveness": rb_stats.get("avg_effectiveness", 0),
                "matcher_few_shots": len(verifier.matcher.few_shot_examples),
                "decomposer_few_shots": len(verifier.decomposer.few_shot_examples),
                "kb_size": len(kb),
                "confusion_matrix": metrics.get("confusion_matrix", {}),
                "per_label": metrics.get("per_label", {}),
            }
            bench_results["epochs"].append(epoch_record)

            # Self-evolution (if not last epoch)
            if epoch < args.epochs - 1:
                # Extract failures using gold labels
                failures = extract_failures_from_results(results, train_samples)
                print(f"  Failures extracted: {len(failures)}")

                if failures:
                    # Evolve verifier (few-shots + ReasoningBank rules)
                    n_evol = verifier.evolve(failures)
                    print(f"  Evolution: {n_evol} corrections/rules added")

                    # Calibrate
                    if len(kb) >= 3:
                        cal = verifier.calibrate(n_per_label=5, seed=epoch)
                        print(f"  Calibration accuracy: {cal.get('accuracy', 0):.1%}")

                    # Evict low-effectiveness rules before persisting
                    if verifier.reasoning_bank:
                        verifier.reasoning_bank._evict()

                    # Persist
                    memory.save_matcher_few_shots(verifier.matcher.few_shot_examples)
                    memory.save_decomposer_few_shots(verifier.decomposer.few_shot_examples)
                    if verifier.reasoning_bank:
                        verifier.reasoning_bank.save()

                    rb_stats = reasoning_bank.stats()
                    print(f"  ReasoningBank: {rb_stats['total_rules']} rules "
                          f"({rb_stats['success_rules']} success, {rb_stats['failure_rules']} failure)")

        # Store final metrics
        final = bench_results["epochs"][-1]
        all_results[bench_name] = {
            "final_accuracy": final["eval_accuracy"],
            "final_macro_f1": final["eval_macro_f1"],
            "evolution_curve": [
                {"epoch": e["epoch"], "accuracy": e["eval_accuracy"], "f1": e["eval_macro_f1"]}
                for e in bench_results["epochs"]
            ],
        }

        # Save detailed results
        detail_path = Path(output_dir) / f"{bench_name}_verifier_agent.json"
        with open(detail_path, "w") as f:
            json.dump(bench_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("VERIFIER AGENT RESULTS")
    print(f"{'='*60}")
    for bench, data in all_results.items():
        print(f"\n  {bench}:")
        print(f"    Final accuracy: {data['final_accuracy']:.1%}")
        print(f"    Final macro F1: {data['final_macro_f1']:.3f}")
        if len(data["evolution_curve"]) > 1:
            first = data["evolution_curve"][0]["accuracy"]
            last = data["evolution_curve"][-1]["accuracy"]
            delta = last - first
            print(f"    Evolution: {first:.1%} → {last:.1%} ({delta:+.1%})")

    # Save combined results
    combined_path = Path(output_dir) / "verifier_agent_summary.json"
    with open(combined_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": args.model,
                "epochs": args.epochs,
                "majority_vote": args.majority_vote,
                "benchmarks": benchmark_names,
            },
            "results": all_results,
            "reasoning_bank_stats": reasoning_bank.stats(),
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
