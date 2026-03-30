"""Generate AdversarialFACT datasets from existing benchmarks.

Applies 6 adversarial strategies to produce harder benchmark variants:
1. Numerical perturbation — subtle number changes → contradictions
2. Multi-hop grafting — compound claims requiring multi-step reasoning
3. False presupposition — inject incorrect premises
4. Unanswerable wrapping — transform into questions lacking evidence
5. Paraphrase — surface-form changes to test robustness
6. Entity confusion — swap attributes between similar entities

Usage:
    # Generate from FEVER manual sample
    python scripts/generate_adversarial.py --source fever --manual --output data/adversarial/fever_hard.jsonl

    # Generate from TruthfulQA + SciFact combined
    python scripts/generate_adversarial.py --source truthfulqa,scifact --manual --output data/adversarial/combined_hard.jsonl

    # Generate balanced dataset of specific size
    python scripts/generate_adversarial.py --source fever --manual --target-size 300 --balance

    # Generate from HuggingFace data with custom strategies
    python scripts/generate_adversarial.py --source fever --limit 1000 --strategies numerical_perturb,entity_confusion,presupposition

    # Generate all (FEVER + TruthfulQA + SciFact + HaluEval + MuSiQue)
    python scripts/generate_adversarial.py --source all --manual --output data/adversarial/adversarial_fact_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.adversarial import (
    AdversarialGenerator,
    EntityConfusionStrategy,
    MultiHopGraftStrategy,
    NumericalPerturbStrategy,
    ParaphraseStrategy,
    PresuppositionStrategy,
    QualityFilter,
    UnanswerableWrapStrategy,
)
from src.benchmarks import (
    FActScoreLoader,
    FEVERLoader,
    HaluEvalLoader,
    MuSiQueLoader,
    SciFactLoader,
    TruthfulQALoader,
)

BENCHMARK_LOADERS = {
    "fever": FEVERLoader,
    "truthfulqa": TruthfulQALoader,
    "scifact": SciFactLoader,
    "halueval": HaluEvalLoader,
    "musique": MuSiQueLoader,
    "factscore": FActScoreLoader,
}

STRATEGY_MAP = {
    "numerical_perturb": NumericalPerturbStrategy,
    "multi_hop_graft": MultiHopGraftStrategy,
    "presupposition": PresuppositionStrategy,
    "unanswerable_wrap": UnanswerableWrapStrategy,
    "paraphrase": ParaphraseStrategy,
    "entity_confusion": EntityConfusionStrategy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AdversarialFACT datasets")
    parser.add_argument("--source", required=True,
                        help="Source benchmark(s), comma-separated. 'all' for everything.")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: data/adversarial/<source>_hard.jsonl)")
    parser.add_argument("--manual", action="store_true",
                        help="Use manual curated samples (no HuggingFace download)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max source samples to load per benchmark")
    parser.add_argument("--target-size", type=int, default=None,
                        help="Target output dataset size")
    parser.add_argument("--balance", action="store_true",
                        help="Balance S/C/N labels in output")
    parser.add_argument("--strategies", default=None,
                        help="Comma-separated strategy names to use (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", default="validation", help="Dataset split to load")
    args = parser.parse_args()

    # Determine sources
    if args.source == "all":
        sources = list(BENCHMARK_LOADERS.keys())
    else:
        sources = [s.strip() for s in args.source.split(",")]
        for s in sources:
            if s not in BENCHMARK_LOADERS:
                print(f"Unknown benchmark: {s}. Available: {list(BENCHMARK_LOADERS.keys())}")
                sys.exit(1)

    # Determine strategies
    if args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(",")]
        strategies = []
        for name in strategy_names:
            if name not in STRATEGY_MAP:
                print(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
                sys.exit(1)
            strategies.append(STRATEGY_MAP[name]())
    else:
        strategies = None  # Use all

    # Output path
    if args.output:
        output_path = args.output
    else:
        source_tag = "_".join(sources) if len(sources) <= 3 else "combined"
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = f"data/adversarial/{source_tag}_hard_{timestamp}.jsonl"

    # Default strategy weights: downweight paraphrase, boost hard strategies
    default_weights = {
        "numerical_perturb": 1.0,
        "multi_hop_graft": 1.0,
        "presupposition": 1.0,
        "unanswerable_wrap": 1.0,
        "paraphrase": 0.3,  # Reduce paraphrase dominance
        "entity_confusion": 1.0,
    }

    # Initialize generator
    quality_filter = QualityFilter(
        min_length=15,
        max_length=500,
        min_edit_distance=0.05,
        dedup_threshold=0.90,
    )

    generator = AdversarialGenerator(
        strategies=strategies,
        quality_filter=quality_filter,
        seed=args.seed,
    )

    # Load source data and generate
    all_adversarial = []

    for source_name in sources:
        print(f"\n{'='*50}")
        print(f"Source: {source_name}")
        print(f"{'='*50}")

        loader_cls = BENCHMARK_LOADERS[source_name]
        loader = loader_cls()

        if args.manual:
            samples = loader.load_manual_sample(limit=args.limit or 50)
        else:
            try:
                samples = loader.load(split=args.split, limit=args.limit)
            except Exception as e:
                print(f"  HuggingFace load failed ({e}), falling back to manual...")
                samples = loader.load_manual_sample(limit=args.limit or 50)

        print(f"  Loaded {len(samples)} source samples")

        # Generate adversarial variants
        if args.target_size and args.balance:
            per_source = args.target_size // len(sources)
            adversarial = generator.generate_dataset(
                samples, source_name,
                target_size=per_source,
                balance_labels=True,
                strategy_weights=default_weights,
            )
        else:
            adversarial = generator.generate_from_benchmark(
                samples, source_name,
                strategy_weights=default_weights,
            )

        print(f"  Generated {len(adversarial)} adversarial samples")

        # Per-source stats
        stats = AdversarialGenerator.dataset_stats(adversarial)
        print(f"  Labels: {stats['label_distribution']}")
        print(f"  Strategies: {stats['strategy_distribution']}")
        print(f"  Difficulty: {stats['difficulty_distribution']}")

        all_adversarial.extend(adversarial)

    # Apply target size if set (without balance, just truncate)
    if args.target_size and not args.balance and len(all_adversarial) > args.target_size:
        import random
        rng = random.Random(args.seed)
        rng.shuffle(all_adversarial)
        all_adversarial = all_adversarial[:args.target_size]

    # Save
    generator.save_dataset(all_adversarial, output_path)

    # Final stats
    final_stats = AdversarialGenerator.dataset_stats(all_adversarial)

    print(f"\n{'='*60}")
    print(f"AdversarialFACT Dataset Generated")
    print(f"{'='*60}")
    print(f"  Output: {output_path}")
    print(f"  Total samples: {final_stats['total']}")
    print(f"  Label distribution: {final_stats['label_distribution']}")
    print(f"  Strategy distribution: {final_stats['strategy_distribution']}")
    print(f"  Difficulty distribution: {final_stats['difficulty_distribution']}")
    print(f"  Avg claim length: {final_stats['avg_claim_length']:.0f} chars")

    # Also save metadata
    meta_path = output_path.replace(".jsonl", "_meta.json")
    metadata = {
        "name": "AdversarialFACT",
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "sources": sources,
        "strategies": [s.name for s in (strategies or [cls() for cls in STRATEGY_MAP.values()])],
        "seed": args.seed,
        "stats": final_stats,
        "description": (
            "Adversarially harder fact verification dataset generated from "
            f"{', '.join(sources)} using rule-based perturbation strategies."
        ),
    }
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata: {meta_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
