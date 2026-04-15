"""Verifier: orchestrates claim decomposition, evidence matching, and scoring.

Supports self-evolution via:
1. Few-shot corrections from past failures (original)
2. ReasoningBank structured rules (from ReasoningBank paper 2509.25140)
3. Majority-vote verification (from R-Zero, ICLR 2026)
4. Calibration bias detection and correction
"""

from __future__ import annotations

from collections import Counter

from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.schema import AtomicClaim, ClaimLabel, Probe, Response, VerificationResult
from src.evolution.failure_extractor import InformativeFailure
from src.evolution.memory_store import MemoryStore
from src.evolution.reasoning_bank import ReasoningBank
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
        reasoning_bank: ReasoningBank | None = None,
        majority_vote_n: int = 1,
    ):
        self.llm = llm
        self.decomposer = Decomposer(llm)
        self.matcher = EvidenceMatcher(llm)
        self.kb = knowledge_base
        self.scorer = scorer or Scorer()
        self.tools = tool_registry
        self.memory = memory
        self.reasoning_bank = reasoning_bank
        self.majority_vote_n = majority_vote_n  # R-Zero: 1 = no vote, 3 = triple vote
        # Runtime context set by the experiment runner
        self._epoch: int = 0
        self._experiment_id: str = ""
        self._model_name: str = ""

    def evolve(self, failures: list[InformativeFailure]) -> int:
        """Incorporate past failures to improve verification in the next epoch.

        Evolution channels:
        1. Prompt Refinement: analyze errors → LLM generates improved system prompt
        2. Few-shot corrections: targeted corrections for EvidenceMatcher
        3. ReasoningBank: distill rules from failure trajectories

        Returns the number of improvements applied.
        """
        matcher_examples: list[dict] = []
        rule_trajectories: list[dict] = []

        for f in failures:
            matcher_examples.append({
                "claim": f.claim_text,
                "evidence": f.response_text[:200],
                "wrong_label": f.label,
                "correct_label": f.pattern.split("gold is ")[-1] if "gold is" in f.pattern else "?",
                "failure_type": f.failure_type,
                "confidence": f.confidence,
            })

            rule_trajectories.append({
                "claim": f.claim_text,
                "evidence": f.response_text[:200],
                "predicted_label": f.label,
                "gold_label": None,
                "confidence": f.confidence,
                "context": f"failure_type={f.failure_type}, risk={f.risk_type}, pattern={f.pattern}",
            })

        # Channel 1 (PRIMARY): Prompt Refinement — rewrite the system prompt
        self._refine_prompt(failures)

        # Channel 2: Keep a few targeted corrections
        if matcher_examples:
            self.matcher.update_few_shots(matcher_examples)

        # Channel 3: Distill structured rules into ReasoningBank
        n_rules = 0
        if self.reasoning_bank and rule_trajectories:
            sample = rule_trajectories[:10]
            rules = self.reasoning_bank.distill_batch(sample, epoch=self._epoch)
            n_rules = len(rules)

        total = len(matcher_examples) + n_rules + 1  # +1 for prompt refinement
        return total

    def _refine_prompt(self, failures: list[InformativeFailure]) -> None:
        """Core evolution: analyze error patterns and generate an improved system prompt.

        Instead of appending rules to the prompt (inflation), we REWRITE it.
        The prompt stays compact but gets smarter each epoch.
        """
        # Build error analysis summary
        error_summary = []
        fp_count = sum(1 for f in failures if f.failure_type == "false_positive")
        fn_count = sum(1 for f in failures if f.failure_type == "false_negative")
        boundary_count = sum(1 for f in failures if f.failure_type == "boundary")

        error_summary.append(f"Error breakdown: {fp_count} false positives (predicted C but was S/N), "
                           f"{fn_count} false negatives (predicted S but was C/N), "
                           f"{boundary_count} boundary cases (S↔N confusion)")

        # Sample concrete error examples
        examples_text = ""
        for f in failures[:5]:
            examples_text += (
                f"\n- Claim: \"{f.claim_text[:100]}\"\n"
                f"  Predicted: {f.label}, Correct: {f.pattern.split('gold is ')[-1] if 'gold is' in f.pattern else '?'}\n"
                f"  Type: {f.failure_type}\n"
            )

        current_prompt = self.matcher.evolved_prompt or self.matcher._build_system_prompt()

        refinement_prompt = f"""\
You are optimizing a verification system prompt. The current prompt is used to classify claims as Supported (S), Contradicted (C), or Not Mentioned (N) given evidence.

CURRENT SYSTEM PROMPT:
---
{current_prompt}
---

ERROR ANALYSIS FROM LAST EPOCH:
{chr(10).join(error_summary)}

EXAMPLE ERRORS:
{examples_text}

Based on these errors, write an IMPROVED system prompt that:
1. Stays concise (under 300 words)
2. Fixes the specific error patterns observed
3. Includes clear decision criteria for S vs C vs N
4. Keeps the JSON output format: {{"label": "S"/"C"/"N", "confidence": <float>, "reasoning": "<brief>", "evidence_snippet": "<key quote>"}}

Write ONLY the improved system prompt, nothing else:"""

        result = self.llm.generate(refinement_prompt)
        new_prompt = result.text.strip()

        # Basic validation: must mention S, C, N and JSON
        if '"S"' in new_prompt and '"C"' in new_prompt and '"N"' in new_prompt and "label" in new_prompt:
            self.matcher.evolved_prompt = new_prompt
            self.matcher.epoch = self._epoch + 1

    def get_weakness_profile(self) -> tuple[list[dict], float]:
        """Extract current Verifier weaknesses for Proposer adaptive targeting.

        Returns (weaknesses, boundary_accuracy) for R-Zero style difficulty adaptation.
        """
        weaknesses: list[dict] = []

        # From ReasoningBank: failure rules are weaknesses
        if self.reasoning_bank:
            for rule in self.reasoning_bank.rules:
                if rule.source_type == "failure" and rule.effectiveness < 0.5:
                    weaknesses.append({
                        "title": rule.title,
                        "pattern": rule.description,
                        "tags": rule.tags,
                        "risk_type": "",
                    })

        # From few-shot corrections: each correction = a known weakness
        for ex in self.matcher.few_shot_examples[-10:]:
            weaknesses.append({
                "title": f"Mislabeled {ex.get('wrong_label', '?')} → {ex.get('correct_label', '?')}",
                "pattern": ex.get("reasoning", "")[:100],
                "risk_type": "",
            })

        # Estimate boundary accuracy from recent calibration
        boundary_acc = 0.5
        if self.memory:
            cal_history = self.memory.load_calibration_history()
            if cal_history:
                boundary_acc = cal_history[-1].get("accuracy", 0.5)

        return weaknesses, boundary_acc

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

        # Step 2.5: Inject ReasoningBank rules into matcher context
        if self.reasoning_bank:
            for claim in claims:
                relevant_rules = self.reasoning_bank.retrieve_relevant(claim.text)
                if relevant_rules:
                    rules_text = self.reasoning_bank.format_for_prompt(relevant_rules)
                    # Temporarily inject rules into matcher (per-claim context)
                    if not claim.metadata:
                        claim.metadata = {}
                    claim.metadata["reasoning_rules"] = rules_text
                    claim.metadata["rule_ids"] = [r.id for r in relevant_rules]

        # Step 3: Verify claims — majority vote if configured (R-Zero style)
        if self.majority_vote_n > 1:
            verified_claims = self._verify_majority_vote(claims, evidence)
        else:
            verified_claims = self._verify_claims_with_tools(claims, evidence)

        # Step 4: Build result and score
        result = VerificationResult(
            probe=probe,
            response=response,
            claims=verified_claims,
        )
        result = self.scorer.score(result)

        # Step 5: KB auto-growth + logging + rule feedback
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

            # Record rule usage feedback (ReasoningBank learning)
            if self.reasoning_bank and claim.metadata:
                rule_ids = claim.metadata.get("rule_ids", [])
                was_helpful = claim.confidence >= 0.7  # High-confidence = rule was useful
                for rid in rule_ids:
                    self.reasoning_bank.record_usage(rid, was_helpful)

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

    def _verify_majority_vote(
        self, claims: list[AtomicClaim], evidence: str
    ) -> list[AtomicClaim]:
        """R-Zero inspired: verify each claim N times and take majority vote.

        This replaces gold labels with consensus — high agreement = reliable.
        """
        verified: list[AtomicClaim] = []
        for claim in claims:
            votes: list[str] = []
            confidences: list[float] = []

            for _ in range(self.majority_vote_n):
                # Create a copy for each vote
                claim_copy = AtomicClaim(id=claim.id, text=claim.text)
                if claim.metadata:
                    claim_copy.metadata = claim.metadata

                # Get a verification
                result_claims = self._verify_claims_with_tools([claim_copy], evidence)
                if result_claims:
                    c = result_claims[0]
                    votes.append(c.label.value if c.label else "N")
                    confidences.append(c.confidence)

            if votes:
                # Majority vote
                vote_counts = Counter(votes)
                majority_label, majority_count = vote_counts.most_common(1)[0]
                agreement = majority_count / len(votes)

                # Map back to ClaimLabel
                label_map = {"S": ClaimLabel.SUPPORTED, "C": ClaimLabel.CONTRADICTED, "N": ClaimLabel.NOT_MENTIONED}
                claim.label = label_map.get(majority_label, ClaimLabel.NOT_MENTIONED)
                # Confidence = average confidence * agreement ratio
                claim.confidence = (sum(confidences) / len(confidences)) * agreement
                claim.evidence = f"[Majority vote: {dict(vote_counts)}, agreement={agreement:.0%}]"
            else:
                claim.label = ClaimLabel.NOT_MENTIONED
                claim.confidence = 0.3

            verified.append(claim)
        return verified

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
