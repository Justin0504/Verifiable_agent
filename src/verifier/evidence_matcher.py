"""Evidence Matching: verify each atomic claim against retrieved evidence."""

from __future__ import annotations

import json

from src.data.schema import AtomicClaim, ClaimLabel
from src.llm.base import BaseLLM

MATCHER_SYSTEM = """\
You are a factual verification engine. Given a claim and supporting evidence \
documents, determine whether the claim is:

- **Supported (S)**: The evidence directly or strongly supports this claim.
- **Contradicted (C)**: The evidence directly contradicts this claim.
- **Not Mentioned (N)**: The evidence neither supports nor contradicts; insufficient info.

You must be precise:
- A claim is Supported ONLY if the evidence clearly backs it up
- A claim is Contradicted ONLY if the evidence explicitly conflicts with it
- When in doubt, label as Not Mentioned

You MUST reason step by step before giving a final verdict. Follow these steps:

Step 1 — EXTRACT: Identify the key factual assertions in the claim \
(names, numbers, dates, relationships).
Step 2 — RETRIEVE: Find the specific sentences in the evidence that are \
relevant to these assertions. Quote them exactly.
Step 3 — COMPARE: For each assertion, compare it against the retrieved \
evidence. Note whether it matches, conflicts, or has no corresponding evidence.
Step 4 — JUDGE: Based on the comparison, assign the final label.

Output format: Return a JSON object with:
- "step1_extract": key assertions from the claim
- "step2_retrieve": relevant evidence quotes
- "step3_compare": comparison of each assertion vs evidence
- "label": one of "S", "C", "N"
- "confidence": float between 0.0 and 1.0
- "evidence_snippet": the specific part of evidence that supports your judgment
- "reasoning": brief explanation of your verdict
"""

MATCHER_PROMPT = """\
Claim: {claim}

Evidence:
{evidence}

Verify the claim against the evidence above. Think step by step.
"""


class EvidenceMatcher:
    """Match atomic claims against evidence and assign S/C/N labels.

    Supports self-evolution: accumulated few-shot examples from past
    verification failures improve labeling accuracy across epochs.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.few_shot_examples: list[dict] = []
        self.calibration_correction: str = ""

    def match(self, claim: AtomicClaim, evidence: str) -> AtomicClaim:
        """Verify a single claim against evidence, returning an updated claim."""
        prompt = MATCHER_PROMPT.format(claim=claim.text, evidence=evidence)
        system = self._build_system_prompt()
        result = self.llm.generate(prompt, system=system)
        return self._parse_verdict(claim, result.text)

    def match_batch(
        self, claims: list[AtomicClaim], evidence: str
    ) -> list[AtomicClaim]:
        """Verify a batch of claims against the same evidence."""
        return [self.match(claim, evidence) for claim in claims]

    def update_few_shots(self, examples: list[dict]) -> None:
        """Add corrected examples from past failures to improve future matching.

        Each example should have: claim, evidence, wrong_label, correct_label, reasoning.
        """
        self.few_shot_examples.extend(examples)
        # Keep a bounded window to avoid prompt bloat
        if len(self.few_shot_examples) > 15:
            self.few_shot_examples = self.few_shot_examples[-15:]

    def _build_system_prompt(self) -> str:
        """Build system prompt with three layers of evolution context:
        1. Base rules (static)
        2. Calibration corrections (from synthetic data accuracy measurement)
        3. Few-shot corrections (from past failure cases)
        """
        prompt = MATCHER_SYSTEM

        # Layer 2: Calibration-based bias corrections
        if self.calibration_correction:
            prompt += self.calibration_correction

        # Layer 3: Few-shot corrections from past failures
        if self.few_shot_examples:
            prompt += "\n\nLearn from these past verification errors to avoid repeating them:\n"
            for i, ex in enumerate(self.few_shot_examples, 1):
                prompt += (
                    f"\nCorrection {i}:\n"
                    f"  Claim: {ex.get('claim', '')}\n"
                    f"  Evidence: {ex.get('evidence', '')[:200]}\n"
                    f"  Wrong label: {ex.get('wrong_label', '?')} → Correct label: {ex.get('correct_label', '?')}\n"
                    f"  Lesson: {ex.get('reasoning', '')}\n"
                )

        return prompt

    def _parse_verdict(self, claim: AtomicClaim, text: str) -> AtomicClaim:
        """Parse the LLM verdict and update the claim."""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1

        label = ClaimLabel.NOT_MENTIONED
        confidence = 0.5
        evidence_snippet = ""

        if start != -1 and end > 0:
            try:
                data = json.loads(text[start:end])
                raw_label = data.get("label", "N").upper().strip()
                if raw_label in ("S", "SUPPORTED"):
                    label = ClaimLabel.SUPPORTED
                elif raw_label in ("C", "CONTRADICTED"):
                    label = ClaimLabel.CONTRADICTED
                else:
                    label = ClaimLabel.NOT_MENTIONED
                confidence = float(data.get("confidence", 0.5))
                evidence_snippet = data.get("evidence_snippet", "")
            except (json.JSONDecodeError, ValueError):
                pass

        claim.label = label
        claim.confidence = max(0.0, min(1.0, confidence))
        claim.evidence = evidence_snippet
        return claim
