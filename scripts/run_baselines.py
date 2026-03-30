"""Run all baselines on all benchmarks and produce comparison results.

Usage:
    # Run all baselines on all benchmarks (manual samples, no API needed for dry run)
    python scripts/run_baselines.py --manual --dry-run

    # Run specific baseline on specific benchmark
    python scripts/run_baselines.py --baselines selfcheck_gpt,factscore --benchmarks fever,truthfulqa

    # Run with specific LLM provider
    python scripts/run_baselines.py --provider openai --model gpt-4o --manual

    # Run our pipeline as well for comparison
    python scripts/run_baselines.py --manual --include-ours

    # Full run on all benchmarks
    python scripts/run_baselines.py --provider openai --model gpt-4o --benchmarks all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import (
    BaseBaseline,
    BaselineResult,
    CoVeBaseline,
    FActScoreBaseline,
    RetrieveNLIBaseline,
    SAFEBaseline,
    SelfCheckGPTBaseline,
)
from src.benchmarks import (
    FActScoreLoader,
    FEVERLoader,
    HaluEvalLoader,
    MuSiQueLoader,
    SciFactLoader,
    TruthfulQALoader,
)
from src.benchmarks.base import BenchmarkSample

BENCHMARK_LOADERS = {
    "fever": FEVERLoader,
    "truthfulqa": TruthfulQALoader,
    "scifact": SciFactLoader,
    "halueval": HaluEvalLoader,
    "musique": MuSiQueLoader,
    "factscore": FActScoreLoader,
}

BASELINE_REGISTRY = {
    "selfcheck_gpt": SelfCheckGPTBaseline,
    "factscore": FActScoreBaseline,
    "safe": SAFEBaseline,
    "cove": CoVeBaseline,
    "retrieve_nli": RetrieveNLIBaseline,
}


def create_llm(provider: str, model: str, api_key: str | None = None):
    """Create an LLM instance based on provider."""
    if provider == "openai":
        from src.llm.openai_llm import OpenAILLM
        return OpenAILLM(model=model, temperature=0.0)
    elif provider == "anthropic":
        from src.llm.anthropic_llm import AnthropicLLM
        return AnthropicLLM(model=model, temperature=0.0)
    elif provider == "vllm":
        from src.llm.vllm_llm import VLLMLlm
        return VLLMLlm(model=model, temperature=0.0)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class DryRunLLM:
    """Mock LLM for dry-run mode — returns deterministic labels based on heuristics."""

    def __init__(self):
        self.temperature = 0.0
        self._call_count = 0

    def generate(self, prompt: str, system: str | None = None):
        """Return a mock response with heuristic-based labels."""
        self._call_count += 1

        # Heuristic: detect what kind of prompt this is and return appropriate mock
        prompt_lower = prompt.lower()

        if "search quer" in prompt_lower or "generate" in prompt_lower:
            return _MockResponse('["query 1", "query 2"]')

        if "verification question" in prompt_lower:
            return _MockResponse('["Is this claim accurate?", "What are the key facts?"]')

        if "answer the following" in prompt_lower:
            return _MockResponse("Based on my knowledge, this appears to be accurate.")

        # For NLI / verification / consistency prompts
        # Use simple keyword matching to simulate a baseline
        if "contradict" in prompt_lower and "refute" in prompt_lower:
            label = "C"
        elif "not enough" in prompt_lower or "no evidence" in prompt_lower:
            label = "N"
        else:
            # Alternate labels for variety
            labels = ["S", "S", "C", "N", "S"]
            label = labels[self._call_count % len(labels)]

        response_json = json.dumps({
            "label": label,
            "confidence": 0.7,
            "reasoning": "Dry run heuristic",
            "consistent": label == "S",
            "occurrences": 3 if label == "S" else 1,
            "contradictions": 1 if label == "C" else 0,
            "relationship": {"S": "entailment", "C": "contradiction", "N": "neutral"}[label],
            "key_evidence": "Mock evidence",
            "inconsistencies": ["Mock inconsistency"] if label == "C" else [],
        })
        return _MockResponse(response_json)


class _MockResponse:
    def __init__(self, text: str):
        self.text = text
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0


def create_baseline(name: str, llm, kb=None) -> BaseBaseline:
    """Create a baseline instance."""
    if name == "selfcheck_gpt":
        return SelfCheckGPTBaseline(llm=llm, n_samples=3)
    elif name == "factscore":
        return FActScoreBaseline(llm=llm, knowledge_base=kb)
    elif name == "safe":
        return SAFEBaseline(llm=llm, knowledge_base=kb)
    elif name == "cove":
        return CoVeBaseline(llm=llm, n_questions=2)
    elif name == "retrieve_nli":
        return RetrieveNLIBaseline(llm=llm, knowledge_base=kb, top_k=3)
    else:
        raise ValueError(f"Unknown baseline: {name}")


def load_samples(
    benchmark_name: str, manual: bool, limit: int | None, split: str
) -> list[BenchmarkSample]:
    """Load benchmark samples."""
    loader_cls = BENCHMARK_LOADERS[benchmark_name]
    loader = loader_cls()

    if manual:
        return loader.load_manual_sample(limit=limit or 50)

    try:
        return loader.load(split=split, limit=limit)
    except Exception as e:
        print(f"  HuggingFace load failed ({e}), falling back to manual...")
        return loader.load_manual_sample(limit=limit or 50)


def filter_samples_for_baselines(samples: list[BenchmarkSample]) -> list[BenchmarkSample]:
    """Filter out 'mixed' label samples for sample-level evaluation.

    FActScore-style samples with gold_label='mixed' need special handling.
    We expand them into per-claim samples for fair comparison.
    """
    filtered = []
    for s in samples:
        if s.gold_label == "mixed" and s.claims:
            per_claim_labels = s.metadata.get("per_claim_labels", [])
            # Keep as-is for claim-level evaluation
            filtered.append(s)
        elif s.gold_label in ("S", "C", "N"):
            filtered.append(s)
    return filtered


def format_results_table(all_results: dict[str, dict[str, dict]]) -> str:
    """Format results as a readable comparison table."""
    lines = []
    lines.append("=" * 90)
    lines.append("BASELINE COMPARISON RESULTS")
    lines.append("=" * 90)

    # Collect all baselines and benchmarks
    baselines = sorted(set(b for benchmarks in all_results.values() for b in benchmarks))
    benchmarks = sorted(all_results.keys())

    # Header
    header = f"{'Benchmark':<15}"
    for b in baselines:
        header += f" | {b:<15}"
    lines.append(header)
    lines.append("-" * len(header))

    # Accuracy rows
    lines.append("\nAccuracy:")
    for bench in benchmarks:
        row = f"  {bench:<13}"
        for baseline in baselines:
            metrics = all_results[bench].get(baseline, {})
            acc = metrics.get("accuracy", 0.0)
            row += f" | {acc:>13.1%}"
        lines.append(row)

    # Macro F1 rows
    lines.append("\nMacro F1:")
    for bench in benchmarks:
        row = f"  {bench:<13}"
        for baseline in baselines:
            metrics = all_results[bench].get(baseline, {})
            f1 = metrics.get("macro_f1", 0.0)
            row += f" | {f1:>13.3f}"
        lines.append(row)

    # Claim-level accuracy (if available)
    has_claims = any(
        all_results[bench].get(baseline, {}).get("total_claims", 0) > 0
        for bench in benchmarks
        for baseline in baselines
    )
    if has_claims:
        lines.append("\nClaim Accuracy:")
        for bench in benchmarks:
            row = f"  {bench:<13}"
            for baseline in baselines:
                metrics = all_results[bench].get(baseline, {})
                ca = metrics.get("claim_accuracy", 0.0)
                tc = metrics.get("total_claims", 0)
                if tc > 0:
                    row += f" | {ca:>13.1%}"
                else:
                    row += f" | {'N/A':>13}"
            lines.append(row)

    lines.append("\n" + "=" * 90)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baselines on benchmarks")
    parser.add_argument("--baselines", default="all",
                        help="Comma-separated baseline names, or 'all'")
    parser.add_argument("--benchmarks", default="all",
                        help="Comma-separated benchmark names, or 'all'")
    parser.add_argument("--provider", default="openai",
                        help="LLM provider: openai, anthropic, vllm")
    parser.add_argument("--model", default="gpt-4o",
                        help="Model name")
    parser.add_argument("--manual", action="store_true",
                        help="Use manual samples (no HuggingFace download)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock LLM for testing (no API calls)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per benchmark")
    parser.add_argument("--split", default="validation",
                        help="Dataset split")
    parser.add_argument("--output", default=None,
                        help="Output directory for results")
    parser.add_argument("--include-ours", action="store_true",
                        help="Include our Verifier pipeline for comparison")
    args = parser.parse_args()

    # Determine baselines
    if args.baselines == "all":
        baseline_names = list(BASELINE_REGISTRY.keys())
    else:
        baseline_names = [b.strip() for b in args.baselines.split(",")]

    # Determine benchmarks
    if args.benchmarks == "all":
        benchmark_names = list(BENCHMARK_LOADERS.keys())
    else:
        benchmark_names = [b.strip() for b in args.benchmarks.split(",")]

    # Create LLM
    if args.dry_run:
        llm = DryRunLLM()
        print("DRY RUN MODE — using mock LLM (no API calls)\n")
    else:
        llm = create_llm(args.provider, args.model)
        print(f"Using {args.provider}/{args.model}\n")

    # Optional knowledge base
    kb = None
    kb_path = Path("knowledge_base/documents")
    if kb_path.exists():
        try:
            from src.verifier.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(str(kb_path))
            kb.load()
            print(f"Loaded knowledge base: {kb.size()} documents\n")
        except Exception as e:
            print(f"Knowledge base load failed: {e}\n")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or f"results/baselines_{timestamp}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run all combinations
    all_results: dict[str, dict[str, dict]] = {}
    all_raw_results: dict[str, dict[str, list]] = {}

    for bench_name in benchmark_names:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")

        samples = load_samples(bench_name, args.manual, args.limit, args.split)
        samples = filter_samples_for_baselines(samples)
        print(f"  Loaded {len(samples)} samples")

        if not samples:
            print("  No samples, skipping...")
            continue

        all_results[bench_name] = {}
        all_raw_results[bench_name] = {}

        for bl_name in baseline_names:
            print(f"\n  Running: {bl_name}...")
            baseline = create_baseline(bl_name, llm, kb)
            results = baseline.verify_batch(samples)
            metrics = BaseBaseline.compute_metrics(results)

            all_results[bench_name][bl_name] = metrics
            all_raw_results[bench_name][bl_name] = [
                {
                    "sample_id": r.sample_id,
                    "predicted": r.predicted_label,
                    "gold": r.gold_label,
                    "confidence": r.confidence,
                }
                for r in results
            ]

            print(f"    Accuracy: {metrics['accuracy']:.1%}")
            print(f"    Macro F1: {metrics['macro_f1']:.3f}")
            print(f"    Confusion: {metrics['confusion_matrix']}")

    # Print comparison table
    print("\n")
    print(format_results_table(all_results))

    # Save results
    results_path = Path(output_dir) / "baseline_comparison.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "baselines": baseline_names,
                "benchmarks": benchmark_names,
                "provider": args.provider if not args.dry_run else "dry_run",
                "model": args.model if not args.dry_run else "mock",
                "manual": args.manual,
            },
            "metrics": all_results,
            "raw_results": all_raw_results,
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save per-baseline detail files
    for bench_name, baselines in all_raw_results.items():
        for bl_name, raw in baselines.items():
            detail_path = Path(output_dir) / f"{bench_name}_{bl_name}.jsonl"
            with open(detail_path, "w") as f:
                for r in raw:
                    f.write(json.dumps(r) + "\n")

    print(f"Detail files saved to: {output_dir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
