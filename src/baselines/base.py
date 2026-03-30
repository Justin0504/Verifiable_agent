"""Base class for all baseline methods."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.benchmarks.base import BenchmarkSample


@dataclass
class BaselineResult:
    """Result from a baseline method for one sample."""

    sample_id: str
    predicted_label: str  # "S", "C", or "N"
    gold_label: str
    confidence: float = 0.5
    claims: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # For claim-level evaluation (FActScore-style)
    claim_labels: list[str] = field(default_factory=list)
    claim_gold_labels: list[str] = field(default_factory=list)


class BaseBaseline(ABC):
    """Abstract base class for baseline fact verification methods."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def verify_sample(self, sample: BenchmarkSample) -> BaselineResult:
        """Verify a single benchmark sample.

        Args:
            sample: A BenchmarkSample with question, claims, evidence, gold_label.

        Returns:
            BaselineResult with predicted label and metadata.
        """

    def verify_batch(self, samples: list[BenchmarkSample]) -> list[BaselineResult]:
        """Verify a batch of samples. Override for batched implementations."""
        return [self.verify_sample(s) for s in samples]

    @staticmethod
    def compute_metrics(results: list[BaselineResult]) -> dict:
        """Compute accuracy, per-label precision/recall/F1, and confusion matrix."""
        if not results:
            return {"accuracy": 0.0}

        correct = sum(1 for r in results if r.predicted_label == r.gold_label)
        total = len(results)

        # Per-label metrics
        labels = ["S", "C", "N"]
        per_label = {}
        for label in labels:
            tp = sum(1 for r in results if r.predicted_label == label and r.gold_label == label)
            fp = sum(1 for r in results if r.predicted_label == label and r.gold_label != label)
            fn = sum(1 for r in results if r.predicted_label != label and r.gold_label == label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

        macro_f1 = sum(per_label[l]["f1"] for l in labels if per_label[l]["support"] > 0)
        active_labels = sum(1 for l in labels if per_label[l]["support"] > 0)
        macro_f1 = macro_f1 / active_labels if active_labels > 0 else 0.0

        # Confusion matrix
        confusion = {gl: {pl: 0 for pl in labels} for gl in labels}
        for r in results:
            if r.gold_label in confusion and r.predicted_label in labels:
                confusion[r.gold_label][r.predicted_label] += 1

        # Claim-level metrics (for FActScore-style)
        all_claim_preds = []
        all_claim_golds = []
        for r in results:
            all_claim_preds.extend(r.claim_labels)
            all_claim_golds.extend(r.claim_gold_labels)

        claim_accuracy = 0.0
        if all_claim_preds:
            claim_accuracy = sum(
                1 for p, g in zip(all_claim_preds, all_claim_golds) if p == g
            ) / len(all_claim_preds)

        # Bootstrap 95% confidence intervals
        acc_ci = BaseBaseline._bootstrap_ci(
            [1 if r.predicted_label == r.gold_label else 0 for r in results]
        )
        f1_ci = BaseBaseline._bootstrap_ci_f1(results)

        return {
            "accuracy": correct / total,
            "accuracy_ci_95": acc_ci,
            "macro_f1": macro_f1,
            "macro_f1_ci_95": f1_ci,
            "per_label": per_label,
            "confusion_matrix": confusion,
            "total": total,
            "claim_accuracy": claim_accuracy,
            "total_claims": len(all_claim_preds),
        }

    @staticmethod
    def _bootstrap_ci(
        binary_results: list[int],
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for accuracy."""
        if len(binary_results) < 5:
            mean = sum(binary_results) / len(binary_results) if binary_results else 0
            return (mean, mean)

        rng = random.Random(seed)
        n = len(binary_results)
        means = []
        for _ in range(n_bootstrap):
            sample = [rng.choice(binary_results) for _ in range(n)]
            means.append(sum(sample) / n)

        means.sort()
        alpha = (1 - confidence) / 2
        lo = means[int(alpha * n_bootstrap)]
        hi = means[int((1 - alpha) * n_bootstrap)]
        return (round(lo, 4), round(hi, 4))

    @staticmethod
    def _bootstrap_ci_f1(
        results: list["BaselineResult"],
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Compute bootstrap CI for macro F1."""
        if len(results) < 5:
            return (0.0, 0.0)

        rng = random.Random(seed)
        n = len(results)
        labels = ["S", "C", "N"]
        f1_scores = []

        for _ in range(n_bootstrap):
            sample = [rng.choice(results) for _ in range(n)]
            per_label_f1 = []
            for label in labels:
                tp = sum(1 for r in sample if r.predicted_label == label and r.gold_label == label)
                fp = sum(1 for r in sample if r.predicted_label == label and r.gold_label != label)
                fn = sum(1 for r in sample if r.predicted_label != label and r.gold_label == label)
                support = tp + fn
                if support == 0:
                    continue
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                per_label_f1.append(f1)
            macro = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0
            f1_scores.append(macro)

        f1_scores.sort()
        lo = f1_scores[int(0.025 * n_bootstrap)]
        hi = f1_scores[int(0.975 * n_bootstrap)]
        return (round(lo, 4), round(hi, 4))
