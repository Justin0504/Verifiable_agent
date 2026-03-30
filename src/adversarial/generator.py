"""AdversarialFACT generator — orchestrates strategies to produce harder datasets.

Takes benchmark samples as input, applies adversarial strategies, filters
for quality, and outputs a new harder dataset with provenance metadata.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from src.benchmarks.base import BenchmarkSample

from .quality_filter import QualityFilter
from .strategies import (
    ALL_STRATEGIES,
    AdversarialSample,
    AdversarialStrategy,
    EntityConfusionStrategy,
    MultiHopGraftStrategy,
    NumericalPerturbStrategy,
    ParaphraseStrategy,
    PresuppositionStrategy,
    UnanswerableWrapStrategy,
)


class AdversarialGenerator:
    """Generate adversarially harder benchmark data from existing samples.

    Applies multiple perturbation strategies to produce a dataset that
    is systematically harder than the original benchmark.
    """

    def __init__(
        self,
        strategies: list[AdversarialStrategy] | None = None,
        quality_filter: QualityFilter | None = None,
        seed: int = 42,
        samples_per_strategy: int = 1,
    ):
        """
        Args:
            strategies: List of strategies to apply. Defaults to all.
            quality_filter: Filter for output quality. Defaults to standard filter.
            seed: Random seed for reproducibility.
            samples_per_strategy: How many variants to attempt per strategy per input.
        """
        if strategies is None:
            strategies = [cls() for cls in ALL_STRATEGIES]
        self.strategies = strategies
        self.filter = quality_filter or QualityFilter()
        self.rng = random.Random(seed)
        self.samples_per_strategy = samples_per_strategy
        self._counter = 0

    def generate_from_benchmark(
        self,
        samples: list[BenchmarkSample],
        source_name: str = "unknown",
        strategy_weights: dict[str, float] | None = None,
    ) -> list[AdversarialSample]:
        """Generate adversarial variants from benchmark samples.

        Args:
            samples: Input benchmark samples.
            source_name: Name of the source benchmark (e.g., "fever", "truthfulqa").
            strategy_weights: Optional {strategy_name: weight} for sampling.
                Strategies not in the dict are included with weight 1.0.

        Returns:
            Quality-filtered adversarial samples.
        """
        # Determine active strategies with weights
        active = self.strategies
        if strategy_weights:
            weighted = []
            for s in active:
                w = strategy_weights.get(s.name, 1.0)
                if w > 0:
                    weighted.append((s, w))
            active_with_weights = weighted
        else:
            active_with_weights = [(s, 1.0) for s in active]

        all_generated: list[AdversarialSample] = []

        for sample in samples:
            # For FActScore-style samples with per-claim labels,
            # expand individual claims as separate inputs
            if sample.gold_label == "mixed" and sample.claims:
                per_claim_labels = sample.metadata.get("per_claim_labels", [])
                claim_inputs = []
                for j, c in enumerate(sample.claims):
                    lbl = per_claim_labels[j] if j < len(per_claim_labels) else "S"
                    claim_inputs.append((c, lbl))
            else:
                claim_inputs = [(sample.question, sample.gold_label)]

            for claim_text, claim_label in claim_inputs:
                for strategy, weight in active_with_weights:
                    if weight < 1.0 and self.rng.random() > weight:
                        continue

                    for _ in range(self.samples_per_strategy):
                        variants = strategy.generate(
                            claim=claim_text,
                            label=claim_label,
                            evidence=sample.evidence,
                            metadata=sample.metadata,
                            rng=self.rng,
                        )

                        for v in variants:
                            self._counter += 1
                            v.id = f"advfact_{source_name}_{self._counter:06d}"
                            v.original_id = sample.id
                            v.metadata["source_benchmark"] = source_name
                            v.metadata["original_gold_label"] = sample.gold_label
                            all_generated.append(v)

        # Quality filter
        filtered = self.filter.filter_batch(all_generated)

        return filtered

    def generate_dataset(
        self,
        samples: list[BenchmarkSample],
        source_name: str = "unknown",
        target_size: int | None = None,
        balance_labels: bool = True,
        strategy_weights: dict[str, float] | None = None,
    ) -> list[AdversarialSample]:
        """Generate a balanced adversarial dataset.

        Args:
            samples: Input benchmark samples.
            source_name: Source benchmark name.
            target_size: Desired output size (None = all that pass filter).
            balance_labels: If True, balance S/C/N labels in output.
            strategy_weights: Optional {strategy_name: weight} for sampling.

        Returns:
            Balanced, filtered adversarial dataset.
        """
        all_samples = self.generate_from_benchmark(
            samples, source_name, strategy_weights=strategy_weights,
        )

        if not balance_labels or not target_size:
            if target_size and len(all_samples) > target_size:
                self.rng.shuffle(all_samples)
                return all_samples[:target_size]
            return all_samples

        # Balance by label
        by_label: dict[str, list[AdversarialSample]] = {"S": [], "C": [], "N": []}
        for s in all_samples:
            if s.gold_label in by_label:
                by_label[s.gold_label].append(s)

        per_label = target_size // 3
        balanced = []
        for label, items in by_label.items():
            self.rng.shuffle(items)
            balanced.extend(items[:per_label])

        self.rng.shuffle(balanced)
        return balanced

    def save_dataset(
        self,
        samples: list[AdversarialSample],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        """Save adversarial dataset to disk.

        Args:
            samples: Adversarial samples to save.
            output_path: Output file path.
            format: "jsonl" or "json".
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        records = []
        for s in samples:
            records.append({
                "id": s.id,
                "claim": s.claim,
                "gold_label": s.gold_label,
                "strategy": s.strategy,
                "difficulty": s.difficulty,
                "original_id": s.original_id,
                "original_claim": s.original_claim,
                "evidence": s.evidence,
                "explanation": s.explanation,
                "metadata": s.metadata,
            })

        if format == "jsonl":
            with open(output_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_dataset(path: str) -> list[AdversarialSample]:
        """Load a saved adversarial dataset."""
        samples = []
        p = Path(path)

        if p.suffix == ".jsonl":
            with open(p) as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(AdversarialSample(**{
                        k: v for k, v in data.items()
                        if k in AdversarialSample.__dataclass_fields__
                    }))
        else:
            with open(p) as f:
                records = json.load(f)
            for data in records:
                samples.append(AdversarialSample(**{
                    k: v for k, v in data.items()
                    if k in AdversarialSample.__dataclass_fields__
                }))

        return samples

    @staticmethod
    def dataset_stats(samples: list[AdversarialSample]) -> dict:
        """Compute statistics for an adversarial dataset."""
        label_dist = Counter(s.gold_label for s in samples)
        strategy_dist = Counter(s.strategy for s in samples)
        difficulty_dist = Counter(s.difficulty for s in samples)

        return {
            "total": len(samples),
            "label_distribution": dict(label_dist),
            "strategy_distribution": dict(strategy_dist),
            "difficulty_distribution": dict(difficulty_dist),
            "avg_claim_length": sum(len(s.claim) for s in samples) / len(samples) if samples else 0,
        }
