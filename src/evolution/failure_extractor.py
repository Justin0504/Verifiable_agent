"""Extract informative failures from verification results for evolution."""

from __future__ import annotations

from dataclasses import dataclass

from src.data.schema import ClaimLabel, VerificationResult


@dataclass
class InformativeFailure:
    """A failure case that carries signal for improving both Proposer and Verifier."""

    failure_type: str  # "false_positive" | "false_negative" | "boundary"
    probe_question: str
    response_text: str
    claim_text: str
    label: str
    confidence: float
    pattern: str  # Human-readable description of the failure pattern
    risk_type: str


class FailureExtractor:
    """Extract informative failure cases from verification results.

    Failure types:
    - False Positive: claim labeled Contradicted but confidence is low (potential Verifier error)
    - False Negative: claim labeled Supported but response seems suspicious
    - Boundary Case: claims near the S/C/N decision boundary (low confidence)
    """

    def __init__(self, confidence_threshold: float = 0.7, boundary_band: float = 0.15):
        self.confidence_threshold = confidence_threshold
        self.boundary_band = boundary_band

    def extract(self, results: list[VerificationResult]) -> list[InformativeFailure]:
        """Extract all informative failures from a batch of results."""
        failures: list[InformativeFailure] = []
        for result in results:
            failures.extend(self._extract_from_result(result))
        return failures

    def _extract_from_result(self, result: VerificationResult) -> list[InformativeFailure]:
        failures: list[InformativeFailure] = []

        for claim in result.claims:
            if claim.label is None:
                continue

            # Boundary case: confidence near the threshold
            if abs(claim.confidence - 0.5) < self.boundary_band:
                failures.append(
                    InformativeFailure(
                        failure_type="boundary",
                        probe_question=result.probe.question,
                        response_text=result.response.text[:200],
                        claim_text=claim.text,
                        label=claim.label.value,
                        confidence=claim.confidence,
                        pattern=f"Boundary case: claim '{claim.text[:80]}...' has confidence {claim.confidence:.2f}",
                        risk_type=result.probe.risk_type.value,
                    )
                )

            # False positive candidate: Contradicted with low confidence
            if claim.label == ClaimLabel.CONTRADICTED and claim.confidence < self.confidence_threshold:
                failures.append(
                    InformativeFailure(
                        failure_type="false_positive",
                        probe_question=result.probe.question,
                        response_text=result.response.text[:200],
                        claim_text=claim.text,
                        label=claim.label.value,
                        confidence=claim.confidence,
                        pattern=f"Low-confidence contradiction: '{claim.text[:80]}...' (conf={claim.confidence:.2f})",
                        risk_type=result.probe.risk_type.value,
                    )
                )

            # False negative candidate: Supported for unanswerable probe
            if (
                claim.label == ClaimLabel.SUPPORTED
                and result.probe.risk_type.value == "unanswerable"
            ):
                failures.append(
                    InformativeFailure(
                        failure_type="false_negative",
                        probe_question=result.probe.question,
                        response_text=result.response.text[:200],
                        claim_text=claim.text,
                        label=claim.label.value,
                        confidence=claim.confidence,
                        pattern=f"Supported claim for unanswerable probe: '{claim.text[:80]}...'",
                        risk_type=result.probe.risk_type.value,
                    )
                )

        return failures

    def summarize(self, failures: list[InformativeFailure]) -> dict:
        """Summarize failure distribution."""
        summary: dict = {"total": len(failures), "by_type": {}, "by_risk": {}}
        for f in failures:
            summary["by_type"][f.failure_type] = summary["by_type"].get(f.failure_type, 0) + 1
            summary["by_risk"][f.risk_type] = summary["by_risk"].get(f.risk_type, 0) + 1
        return summary
