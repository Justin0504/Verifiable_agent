"""Phase 1 A/B test: PredicateMatcher vs EvidenceMatcher vs baselines.

Runs on the same test set to measure the impact of predicate-based hybrid verification.

Usage:
    python scripts/run_predicate_comparison.py --benchmark fever --test-limit 200
    python scripts/run_predicate_comparison.py --benchmark truthfulqa --test-limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from src.baselines.base import BaseBaseline, BaselineResult
from src.baselines.direct_prompting import DirectPromptingBaseline
from src.baselines.retrieve_nli import RetrieveNLIBaseline
from src.benchmarks import (
    FEVERLoader, HaluEvalLoader, MuSiQueLoader, SciFactLoader, TruthfulQALoader,
)
from src.benchmarks.base import BenchmarkSample
from src.data.schema import AtomicClaim, ClaimLabel
from src.llm.openai_llm import OpenAILLM
from src.verifier.evidence_matcher import EvidenceMatcher
from src.verifier.predicate_matcher import PredicateMatcher

BENCHMARK_LOADERS = {
    "fever": FEVERLoader, "truthfulqa": TruthfulQALoader,
    "scifact": SciFactLoader, "halueval": HaluEvalLoader, "musique": MuSiQueLoader,
}


def load_samples(name, limit, split):
    loader = BENCHMARK_LOADERS[name]()
    try:
        samples = loader.load(split=split, limit=limit)
    except Exception as e:
        print(f"  HuggingFace failed ({e}), manual fallback...")
        samples = loader.load_manual_sample(limit=limit or 50)
    return [s for s in samples if s.gold_label in ("S", "C", "N")]


def aggregate_claim_labels(claim_labels, claim_confidences):
    """Confidence-weighted majority vote across claims."""
    if not claim_labels:
        return "N", 0.5

    if "C" in claim_labels:
        c_confs = [c for l, c in zip(claim_labels, claim_confidences) if l == "C"]
        if max(c_confs) >= 0.75:
            return "C", max(c_confs)

    label_weight = {}
    for label, conf in zip(claim_labels, claim_confidences):
        label_weight[label] = label_weight.get(label, 0) + conf

    priority = {"C": 2, "N": 1, "S": 0}
    predicted = max(label_weight, key=lambda l: (label_weight[l], priority.get(l, 0)))
    avg_conf = sum(claim_confidences) / len(claim_confidences)
    return predicted, avg_conf


def predict_with_matcher(matcher, sample):
    """Predict using any matcher (EvidenceMatcher or PredicateMatcher)."""
    evidence_text = "\n\n".join(sample.evidence) if sample.evidence else ""
    claims_text = sample.claims if sample.claims else [sample.question]
    claims = [AtomicClaim(id=f"{sample.id}_c{i}", text=c) for i, c in enumerate(claims_text)]

    verified = [matcher.match(c, evidence_text) for c in claims]
    claim_labels = [c.label.value if c.label else "N" for c in verified]
    claim_confs = [c.confidence for c in verified]

    predicted, avg_conf = aggregate_claim_labels(claim_labels, claim_confs)

    return BaselineResult(
        sample_id=sample.id, predicted_label=predicted, gold_label=sample.gold_label,
        confidence=avg_conf,
        claims=[{"claim": c.text, "label": l} for c, l in zip(verified, claim_labels)],
        claim_labels=claim_labels,
        claim_gold_labels=[sample.gold_label] * len(claims_text),
        metadata={"method": "predicate_matcher" if isinstance(matcher, PredicateMatcher) else "evidence_matcher"},
    )


def run_matcher_eval(matcher, test_samples, name, n_votes=1):
    """Run evaluation with optional majority voting."""
    print(f"\n  {name}...")
    t0 = time.time()
    results = []

    for i, s in enumerate(test_samples):
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(test_samples)}]...")

        if n_votes > 1:
            votes, confs = [], []
            for _ in range(n_votes):
                r = predict_with_matcher(matcher, s)
                votes.append(r.predicted_label)
                confs.append(r.confidence)
            label_conf = {}
            for v, c in zip(votes, confs):
                label_conf[v] = label_conf.get(v, 0) + c
            majority = max(label_conf, key=label_conf.get)
            results.append(BaselineResult(
                sample_id=s.id, predicted_label=majority, gold_label=s.gold_label,
                confidence=sum(confs) / len(confs),
                metadata={"method": name, "votes": dict(Counter(votes))},
            ))
        else:
            r = predict_with_matcher(matcher, s)
            results.append(r)

    metrics = BaseBaseline.compute_metrics(results)
    elapsed = time.time() - t0
    print(f"    {metrics['accuracy']:.1%} (F1={metrics['macro_f1']:.3f}, {elapsed:.0f}s)")
    return metrics, elapsed


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Predicate vs Evidence Matcher comparison")
    parser.add_argument("--benchmark", default="fever")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--cal-ratio", type=float, default=0.2)
    parser.add_argument("--majority-votes", type=int, default=1)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"=== Phase 1 A/B Test: {args.benchmark} ===")
    print(f"PredicateMatcher (hybrid) vs EvidenceMatcher (LLM-only) vs baselines")
    print()

    # Load data
    all_samples = load_samples(args.benchmark, args.test_limit, args.split)
    random.Random(42).shuffle(all_samples)

    n_cal = max(10, int(len(all_samples) * args.cal_ratio))
    test_samples = all_samples[n_cal:]
    test_dist = Counter(s.gold_label for s in test_samples)
    print(f"Test: {len(test_samples)} samples {dict(test_dist)}")
    print()

    llm = OpenAILLM(model=args.model, temperature=0.2)
    output_dir = args.output or f"results/phase1_{args.benchmark}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    # ============================================================
    # 1. PredicateMatcher (new, hybrid)
    # ============================================================
    pred_matcher = PredicateMatcher(llm)
    metrics, elapsed = run_matcher_eval(
        pred_matcher, test_samples, "predicate_matcher", args.majority_votes)
    all_metrics["predicate_matcher"] = {**metrics, "time": elapsed}

    # Print predicate stats
    stats = pred_matcher.get_stats()
    print(f"    Predicate stats: {stats}")
    print(f"    Deterministic ratio: {stats['deterministic_ratio']:.1%}")

    # ============================================================
    # 2. EvidenceMatcher (old, LLM-only)
    # ============================================================
    ev_matcher = EvidenceMatcher(llm)
    metrics, elapsed = run_matcher_eval(
        ev_matcher, test_samples, "evidence_matcher", args.majority_votes)
    all_metrics["evidence_matcher"] = {**metrics, "time": elapsed}

    # ============================================================
    # 3. Key baselines
    # ============================================================
    baselines = {
        "direct_prompting": DirectPromptingBaseline(llm),
        "retrieve_nli": RetrieveNLIBaseline(llm),
    }
    for bl_name, bl in baselines.items():
        print(f"\n  {bl_name}...")
        t0 = time.time()
        bl_results = [bl.verify_sample(s) for s in test_samples]
        bm = BaseBaseline.compute_metrics(bl_results)
        elapsed = time.time() - t0
        print(f"    {bm['accuracy']:.1%} (F1={bm['macro_f1']:.3f}, {elapsed:.0f}s)")
        all_metrics[bl_name] = {**bm, "time": elapsed}

    # ============================================================
    # Results
    # ============================================================
    print("\n" + "=" * 70)
    print(f"PHASE 1 RESULTS: {args.benchmark} | {len(test_samples)} test samples")
    print("=" * 70)

    ranked = sorted(all_metrics.items(), key=lambda x: -x[1]["accuracy"])
    print(f"\n{'Method':<25} {'Accuracy':>10} {'F1':>8} {'Time':>8} {'Rank':>6}")
    print("-" * 61)
    for rank, (name, m) in enumerate(ranked, 1):
        marker = " ***" if name == "predicate_matcher" else ""
        t = m.get("time", 0)
        print(f"{name:<25} {m['accuracy']:>9.1%} {m['macro_f1']:>8.3f} {t:>7.0f}s {rank:>5}{marker}")

    # Delta
    pred_acc = all_metrics["predicate_matcher"]["accuracy"]
    ev_acc = all_metrics["evidence_matcher"]["accuracy"]
    delta = pred_acc - ev_acc
    print(f"\n  PredicateMatcher vs EvidenceMatcher: {delta:+.1%}")

    if "predicate_matcher" in all_metrics:
        pm = all_metrics["predicate_matcher"]
        cm = pm.get("confusion_matrix", {})
        if cm:
            print(f"\n  PredicateMatcher confusion matrix:")
            for gold, preds in cm.items():
                print(f"    gold={gold}: {preds}")

    # Save
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": args.benchmark,
        "test_samples": len(test_samples),
        "test_dist": dict(test_dist),
        "model": args.model,
        "majority_votes": args.majority_votes,
        "metrics": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                         "time": v.get("time", 0),
                         "confusion_matrix": v.get("confusion_matrix", {})}
                    for k, v in all_metrics.items()},
        "ranking": [(n, m["accuracy"]) for n, m in ranked],
        "delta_pred_vs_ev": delta,
    }
    if hasattr(pred_matcher, 'get_stats'):
        results_data["predicate_stats"] = pred_matcher.get_stats()

    out_path = Path(output_dir) / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
