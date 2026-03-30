"""Scoring function: aggregate claim-level labels into a response-level score."""

from __future__ import annotations

from src.data.schema import AtomicClaim, ClaimLabel, VerificationResult


class Scorer:
    """Compute response-level reliability scores from claim labels."""

    def __init__(
        self,
        supported_weight: float = 1.0,
        contradicted_weight: float = -2.0,
        not_mentioned_weight: float = -0.5,
    ):
        self.w_s = supported_weight
        self.w_c = contradicted_weight
        self.w_n = not_mentioned_weight

    def score(self, result: VerificationResult) -> VerificationResult:
        """Compute and attach the aggregated score to a VerificationResult."""
        claims = result.claims
        if not claims:
            result.score = 0.0
            return result

        n_s = sum(1 for c in claims if c.label == ClaimLabel.SUPPORTED)
        n_c = sum(1 for c in claims if c.label == ClaimLabel.CONTRADICTED)
        n_n = sum(1 for c in claims if c.label == ClaimLabel.NOT_MENTIONED)

        result.num_supported = n_s
        result.num_contradicted = n_c
        result.num_not_mentioned = n_n

        total = len(claims)
        raw_score = (n_s * self.w_s + n_c * self.w_c + n_n * self.w_n) / total

        # Normalize to [0, 1] range
        # Worst case: all contradicted → w_c; Best case: all supported → w_s
        min_score = self.w_c
        max_score = self.w_s
        if max_score != min_score:
            result.score = (raw_score - min_score) / (max_score - min_score)
        else:
            result.score = 0.5

        result.score = max(0.0, min(1.0, result.score))
        return result

    def compute_hallucination_rate(self, results: list[VerificationResult]) -> float:
        """Fraction of responses that contain at least one contradicted claim."""
        if not results:
            return 0.0
        hallucinated = sum(1 for r in results if r.num_contradicted > 0)
        return hallucinated / len(results)
