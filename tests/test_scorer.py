"""Tests for the Scorer module."""

import pytest

from src.data.schema import (
    AtomicClaim,
    ClaimLabel,
    Probe,
    Response,
    RiskType,
    VerificationResult,
)
from src.verifier.scorer import Scorer


def _make_result(labels: list[ClaimLabel]) -> VerificationResult:
    claims = [
        AtomicClaim(id=str(i), text=f"Claim {i}", label=label, confidence=0.9)
        for i, label in enumerate(labels)
    ]
    return VerificationResult(
        probe=Probe(id="p1", question="test?", risk_type=RiskType.MULTI_HOP),
        response=Response(probe_id="p1", model_name="test", text="test response"),
        claims=claims,
    )


class TestScorer:
    def test_all_supported(self):
        scorer = Scorer()
        result = _make_result([ClaimLabel.SUPPORTED] * 5)
        scored = scorer.score(result)
        assert scored.score == 1.0
        assert scored.num_supported == 5
        assert scored.num_contradicted == 0

    def test_all_contradicted(self):
        scorer = Scorer()
        result = _make_result([ClaimLabel.CONTRADICTED] * 3)
        scored = scorer.score(result)
        assert scored.score == 0.0
        assert scored.num_contradicted == 3

    def test_mixed(self):
        scorer = Scorer()
        result = _make_result([
            ClaimLabel.SUPPORTED,
            ClaimLabel.SUPPORTED,
            ClaimLabel.CONTRADICTED,
            ClaimLabel.NOT_MENTIONED,
        ])
        scored = scorer.score(result)
        assert 0.0 < scored.score < 1.0
        assert scored.num_supported == 2
        assert scored.num_contradicted == 1
        assert scored.num_not_mentioned == 1

    def test_empty_claims(self):
        scorer = Scorer()
        result = _make_result([])
        scored = scorer.score(result)
        assert scored.score == 0.0

    def test_hallucination_rate(self):
        scorer = Scorer()
        results = [
            _make_result([ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED]),  # has C
            _make_result([ClaimLabel.SUPPORTED, ClaimLabel.SUPPORTED]),     # no C
            _make_result([ClaimLabel.CONTRADICTED]),                         # has C
        ]
        for r in results:
            scorer.score(r)
        rate = scorer.compute_hallucination_rate(results)
        assert rate == pytest.approx(2 / 3)

    def test_custom_weights(self):
        scorer = Scorer(supported_weight=1.0, contradicted_weight=-1.0, not_mentioned_weight=0.0)
        result = _make_result([ClaimLabel.SUPPORTED, ClaimLabel.CONTRADICTED])
        scored = scorer.score(result)
        # raw = (1*1.0 + 1*(-1.0)) / 2 = 0.0
        # normalized = (0.0 - (-1.0)) / (1.0 - (-1.0)) = 0.5
        assert scored.score == pytest.approx(0.5)
