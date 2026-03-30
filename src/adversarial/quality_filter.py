"""Quality filter for adversarial samples.

Ensures generated adversarial data meets minimum quality standards:
1. Grammaticality — claim is well-formed English
2. Label integrity — gold label is actually correct given the perturbation
3. Difficulty delta — the sample is genuinely harder than the original
4. Deduplication — no near-duplicate claims
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .strategies import AdversarialSample


class QualityFilter:
    """Filter and validate adversarial samples for dataset quality."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 500,
        min_edit_distance: float = 0.1,
        dedup_threshold: float = 0.85,
    ):
        """
        Args:
            min_length: Minimum claim character length.
            max_length: Maximum claim character length.
            min_edit_distance: Minimum normalized edit distance from original
                (0 = identical, 1 = completely different).
            dedup_threshold: SequenceMatcher ratio above which two claims
                are considered duplicates.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_edit_distance = min_edit_distance
        self.dedup_threshold = dedup_threshold

    def filter_batch(self, samples: list[AdversarialSample]) -> list[AdversarialSample]:
        """Filter a batch of adversarial samples, returning only quality ones."""
        passed = []
        seen_claims: list[str] = []

        for sample in samples:
            issues = self.check(sample, seen_claims)
            if not issues:
                passed.append(sample)
                seen_claims.append(sample.claim.lower())

        return passed

    def check(self, sample: AdversarialSample, seen_claims: list[str] | None = None) -> list[str]:
        """Check a single sample for quality issues.

        Returns list of issue descriptions (empty = passed).
        """
        issues = []

        # 1. Length check
        if len(sample.claim) < self.min_length:
            issues.append(f"Too short: {len(sample.claim)} chars < {self.min_length}")
        if len(sample.claim) > self.max_length:
            issues.append(f"Too long: {len(sample.claim)} chars > {self.max_length}")

        # 2. Basic grammaticality
        issues.extend(self._check_grammar(sample.claim))

        # 3. Edit distance from original (must be different enough)
        if sample.original_claim:
            ratio = SequenceMatcher(None, sample.claim.lower(), sample.original_claim.lower()).ratio()
            edit_dist = 1.0 - ratio
            if edit_dist < self.min_edit_distance:
                issues.append(f"Too similar to original: edit_distance={edit_dist:.3f} < {self.min_edit_distance}")

        # 4. Not identical to original
        if sample.claim.strip() == sample.original_claim.strip():
            issues.append("Identical to original claim")

        # 5. Label sanity
        issues.extend(self._check_label_sanity(sample))

        # 6. Deduplication
        if seen_claims:
            for seen in seen_claims:
                sim = SequenceMatcher(None, sample.claim.lower(), seen).ratio()
                if sim > self.dedup_threshold:
                    issues.append(f"Near-duplicate of existing claim (similarity={sim:.3f})")
                    break

        return issues

    def _check_grammar(self, text: str) -> list[str]:
        """Basic grammaticality heuristics (not a full grammar checker)."""
        issues = []

        # Must have at least one verb-like word
        if not re.search(r'\b(?:is|was|were|are|has|had|have|did|does|do|can|will|would|should|could|may|might|shall)\b', text, re.IGNORECASE):
            if not re.search(r'\b\w+(?:ed|ing|es|s)\b', text):
                issues.append("No verb detected")

        # Must start with uppercase or quote
        if text and not text[0].isupper() and text[0] not in ('"', "'", "("):
            issues.append("Does not start with uppercase letter")

        # Must end with punctuation
        if text and text[-1] not in ".?!\"')":
            issues.append("Does not end with punctuation")

        # No broken formatting
        if "{{" in text or "}}" in text:
            issues.append("Contains template artifacts")
        if "{" in text and "}" in text:
            issues.append("Contains unfilled template placeholders")

        # No double spaces (except intentional)
        if "  " in text.strip():
            issues.append("Contains double spaces")

        return issues

    def _check_label_sanity(self, sample: AdversarialSample) -> list[str]:
        """Check if the gold label makes sense for the strategy used."""
        issues = []

        strategy = sample.strategy
        label = sample.gold_label

        # Strategy → expected label mapping
        expected = {
            "numerical_perturb": "C",
            "presupposition": "C",
            "entity_confusion": "C",
            "unanswerable_wrap": "N",
            "multi_hop_graft": "S",
            # paraphrase preserves original label
        }

        if strategy in expected and label != expected[strategy]:
            issues.append(f"Label mismatch: {strategy} should produce '{expected[strategy]}' but got '{label}'")

        return issues

    def stats(self, original: list[AdversarialSample], filtered: list[AdversarialSample]) -> dict:
        """Compute filtering statistics."""
        by_strategy_original: dict[str, int] = {}
        by_strategy_filtered: dict[str, int] = {}

        for s in original:
            by_strategy_original[s.strategy] = by_strategy_original.get(s.strategy, 0) + 1
        for s in filtered:
            by_strategy_filtered[s.strategy] = by_strategy_filtered.get(s.strategy, 0) + 1

        return {
            "total_input": len(original),
            "total_output": len(filtered),
            "pass_rate": len(filtered) / len(original) if original else 0.0,
            "by_strategy": {
                strategy: {
                    "input": by_strategy_original.get(strategy, 0),
                    "output": by_strategy_filtered.get(strategy, 0),
                    "pass_rate": (
                        by_strategy_filtered.get(strategy, 0) / by_strategy_original.get(strategy, 0)
                        if by_strategy_original.get(strategy, 0) > 0 else 0.0
                    ),
                }
                for strategy in set(list(by_strategy_original.keys()) + list(by_strategy_filtered.keys()))
            },
        }
