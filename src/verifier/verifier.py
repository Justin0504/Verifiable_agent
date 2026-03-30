"""Verifier: orchestrates claim decomposition, evidence matching, and scoring.

Supports self-evolution: the Verifier accumulates corrective few-shot examples
from past failures, so both its Decomposer and EvidenceMatcher improve across epochs.
"""

from __future__ import annotations

from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.schema import AtomicClaim, ClaimLabel, Probe, Response, VerificationResult
from src.evolution.failure_extractor import InformativeFailure
from src.evolution.memory_store import MemoryStore
from src.llm.base import BaseLLM
from src.tools.registry import ToolRegistry

from .calibration import (
    bias_to_prompt_correction,
    evaluate_verifier_accuracy,
    generate_calibration_set,
)
from .decomposer import Decomposer
from .evidence_matcher import EvidenceMatcher
from .knowledge_base import KnowledgeBase
from .scorer import Scorer


class Verifier:
    """Stage 3 — full verification pipeline with self-evolution and external tools."""

    def __init__(
        self,
        llm: BaseLLM,
        knowledge_base: KnowledgeBase,
        scorer: Scorer | None = None,
        tool_registry: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
    ):
        self.llm = llm
        self.decomposer = Decomposer(llm)
        self.matcher = EvidenceMatcher(llm)
        self.kb = knowledge_base
        self.scorer = scorer or Scorer()
        self.tools = tool_registry
        self.memory = memory
        # Runtime context set by the experiment runner
        self._epoch: int = 0
        self._experiment_id: str = ""
        self._model_name: str = ""

    def evolve(self, failures: list[InformativeFailure]) -> int:
        """Incorporate past failures to improve verification in the next epoch.

        Converts InformativeFailures into few-shot corrections for:
        - EvidenceMatcher: false_positive / false_negative → corrected label examples
        - Decomposer: boundary cases → decomposition refinement hints

        Returns the number of new few-shot examples added.
        """
        matcher_examples: list[dict] = []
        decomposer_examples: list[dict] = []

        for f in failures:
            if f.failure_type == "false_positive":
                # Verifier labeled C, but low confidence → likely should be S or N
                matcher_examples.append({
                    "claim": f.claim_text,
                    "evidence": f.response_text[:200],
                    "wrong_label": "C",
                    "correct_label": "S or N",
                    "reasoning": (
                        f"This claim was labeled Contradicted with low confidence "
                        f"({f.confidence:.2f}). Low-confidence contradictions often "
                        f"indicate the evidence is ambiguous, not truly contradicting. "
                        f"Be more conservative before assigning C."
                    ),
                })

            elif f.failure_type == "false_negative":
                # Verifier labeled S for an unanswerable probe → should flag it
                matcher_examples.append({
                    "claim": f.claim_text,
                    "evidence": f.response_text[:200],
                    "wrong_label": "S",
                    "correct_label": "N or C",
                    "reasoning": (
                        f"This claim was labeled Supported for a question that is "
                        f"fundamentally unanswerable (risk_type={f.risk_type}). "
                        f"When a question is unanswerable, specific factual claims "
                        f"in the response are likely fabricated. Scrutinize more carefully."
                    ),
                })

            elif f.failure_type == "boundary":
                # Near decision boundary → hint for both decomposer and matcher
                decomposer_examples.append({
                    "response": f.response_text[:200],
                    "issue": f"Claim '{f.claim_text[:100]}' was ambiguous (conf={f.confidence:.2f})",
                    "reasoning": (
                        "When decomposing, ensure each claim is maximally atomic "
                        "and self-contained. Ambiguous claims often result from "
                        "merging multiple facts into one claim."
                    ),
                })

        if matcher_examples:
            self.matcher.update_few_shots(matcher_examples)
        if decomposer_examples:
            self.decomposer.update_few_shots(decomposer_examples)

        return len(matcher_examples) + len(decomposer_examples)

    def calibrate(self, n_per_label: int = 15, seed: int = 42) -> dict:
        """Run calibration: test verifier on synthetic data with known labels.

        1. Generate synthetic claim-evidence pairs with gold S/C/N labels
        2. Run the EvidenceMatcher on each pair
        3. Compare predictions to gold labels → accuracy + bias detection
        4. Inject bias corrections into the matcher's system prompt

        Returns calibration results (accuracy, confusion matrix, biases).
        """
        # Step 1: Generate calibration set from KB
        cal_data = generate_calibration_set(self.kb, n_per_label=n_per_label, seed=seed)
        if not cal_data:
            return {"accuracy": 0.0, "error": "Not enough KB data for calibration"}

        # Step 2: Run verifier on each synthetic pair
        predictions = []
        for item in cal_data:
            claim = AtomicClaim(id=item["id"], text=item["claim"])
            matched = self.matcher.match(claim, item["evidence"])
            predicted = matched.label.value if matched.label else "N"
            predictions.append({
                "gold_label": item["gold_label"],
                "predicted_label": predicted,
                "claim": item["claim"],
                "corruption_type": item["corruption_type"],
            })

        # Step 3: Evaluate accuracy
        results = evaluate_verifier_accuracy(predictions)

        # Step 4: Inject bias corrections into matcher prompt
        if results.get("biases"):
            correction = bias_to_prompt_correction(results["biases"])
            if correction:
                self.matcher.calibration_correction = correction

        return results

    def verify(self, probe: Probe, response: Response) -> VerificationResult:
        """Run the full verification pipeline on a single probe-response pair."""
        # Step 1: Decompose response into atomic claims
        claims = self.decomposer.decompose(probe.question, response.text)

        # Step 2: Retrieve evidence from KB
        evidence = self.kb.get_evidence_text(probe.question)

        # If the probe has ground truth, prepend it as primary evidence
        if probe.ground_truth:
            evidence = f"[Ground Truth] {probe.ground_truth}\n\n{evidence}"

        # Step 3: For each claim, try external tools first, then LLM matcher
        verified_claims = self._verify_claims_with_tools(claims, evidence)

        # Step 4: Build result and score
        result = VerificationResult(
            probe=probe,
            response=response,
            claims=verified_claims,
        )
        result = self.scorer.score(result)

        # Step 5: KB auto-growth + logging
        for claim in verified_claims:
            # Log verification trace
            if self.memory:
                self.memory.append_verification_trace({
                    "claim_id": claim.id,
                    "claim_text": claim.text,
                    "probe_id": probe.id,
                    "probe_question": probe.question,
                    "predicted_label": claim.label.value if claim.label else None,
                    "confidence": claim.confidence,
                    "evidence_snippet": claim.evidence,
                    "epoch": self._epoch,
                    "experiment_id": self._experiment_id,
                    "model_name": self._model_name,
                })

            # KB auto-growth — high-confidence Supported claims become new evidence
            if (
                claim.label == ClaimLabel.SUPPORTED
                and claim.confidence >= 0.9
                and claim.evidence
            ):
                self.kb.add_verified_evidence(
                    claim=claim.text,
                    evidence=claim.evidence,
                    source=f"verified_epoch_{self._epoch}",
                )
                # Log KB provenance
                if self.memory:
                    self.memory.append_kb_provenance(
                        doc_id=f"auto_{len(self.kb)}",
                        claim_text=claim.text,
                        evidence_text=claim.evidence,
                        source_probe_id=probe.id,
                        epoch=self._epoch,
                        experiment_id=self._experiment_id,
                        confidence=claim.confidence,
                    )

        return result

    def _verify_claims_with_tools(
        self, claims: list[AtomicClaim], kb_evidence: str
    ) -> list[AtomicClaim]:
        """Verify claims using external tools + LLM matcher.

        Strategy:
        1. For each claim, check if a deterministic tool (calculator, Wikidata)
           can provide a hard verdict. If yes, use it directly — no LLM needed.
        2. Otherwise, augment KB evidence with tool evidence (web search, Wikidata
           facts) and pass to the LLM-based matcher.
        """
        if not self.tools:
            # No tools configured, fall back to pure LLM matching
            return self.matcher.match_batch(claims, kb_evidence)

        verified: list[AtomicClaim] = []
        for claim in claims:
            # Try deterministic tools first
            has_verdict, det_evidence = self.tools.has_deterministic_verdict(claim.text)

            if has_verdict and det_evidence:
                # Calculator or Wikidata gave a hard answer
                # Check if the evidence says "CORRECT" or "INCORRECT"
                if "INCORRECT" in det_evidence:
                    claim.label = ClaimLabel.CONTRADICTED
                    claim.confidence = 1.0
                    claim.evidence = det_evidence
                elif "CORRECT" in det_evidence:
                    claim.label = ClaimLabel.SUPPORTED
                    claim.confidence = 1.0
                    claim.evidence = det_evidence
                else:
                    # Deterministic tool returned facts but no verdict;
                    # augment evidence and let LLM decide
                    combined_evidence = f"{det_evidence}\n\n{kb_evidence}"
                    claim = self.matcher.match(claim, combined_evidence)
            else:
                # No deterministic verdict — try soft tools for extra evidence
                tool_evidence = self.tools.get_evidence(claim.text)
                if tool_evidence:
                    combined_evidence = f"{tool_evidence}\n\n{kb_evidence}"
                else:
                    combined_evidence = kb_evidence
                claim = self.matcher.match(claim, combined_evidence)

            verified.append(claim)
        return verified

    def verify_batch(
        self,
        probes: list[Probe],
        responses: list[Response],
        show_progress: bool = True,
    ) -> list[VerificationResult]:
        """Verify a batch of probe-response pairs."""
        # Build lookup from probe_id to probe
        probe_map = {p.id: p for p in probes}
        results: list[VerificationResult] = []

        items = list(zip(probes, responses))
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("Verifying responses...", total=len(items))
                for probe, response in items:
                    result = self.verify(probe, response)
                    results.append(result)
                    progress.advance(task)
        else:
            for probe, response in items:
                results.append(self.verify(probe, response))

        return results
