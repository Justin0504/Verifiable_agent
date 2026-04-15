"""Evidence Matching: verify each atomic claim against retrieved evidence."""

from __future__ import annotations

import json

from src.data.schema import AtomicClaim, ClaimLabel
from src.llm.base import BaseLLM

MATCHER_SYSTEM_CLEAN = ""  # No system prompt at epoch 0 — matches direct prompting

MATCHER_SYSTEM_COT = """\
You are a factual verification engine. Given a claim and evidence, determine:
- "S" (Supported): The evidence directly supports this claim
- "C" (Contradicted): The evidence directly contradicts this claim
- "N" (Not enough info): Insufficient evidence to verify

Think step by step:
1. Identify key assertions in the claim (names, numbers, dates, relationships)
2. Find specific sentences in the evidence relevant to these assertions
3. Compare each assertion against the evidence
4. Assign the final label

Reply with JSON: {"label": "S"/"C"/"N", "confidence": <float 0-1>, "reasoning": "<brief>", "evidence_snippet": "<key quote>"}
"""

MATCHER_PROMPT_CLEAN = """\
Given the following claim and evidence, determine if the claim is:
- "S" (Supported): The evidence supports this claim
- "C" (Contradicted): The evidence contradicts this claim
- "N" (Not enough info): The evidence is insufficient to verify

Claim: {claim}

Evidence:
{evidence}

Reply with JSON: {{"label": "S"/"C"/"N", "confidence": <float 0-1>, "reasoning": "<brief>"}}
"""

MATCHER_PROMPT_EVOLVED = """\
Claim: {claim}

Evidence:
{evidence}

Verify the claim against the evidence. Reply with JSON: {{"label": "S"/"C"/"N", "confidence": <float 0-1>, "reasoning": "<brief>"}}
"""


class EvidenceMatcher:
    """Match atomic claims against evidence and assign S/C/N labels.

    Supports self-evolution via prompt refinement:
    - Epoch 0: clean prompt (matches direct prompting performance)
    - Epoch 1+: refined system prompt based on error analysis
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.few_shot_examples: list[dict] = []
        self.calibration_correction: str = ""
        self.evolved_prompt: str = ""  # Refined prompt from evolution (replaces base)
        self.epoch: int = 0

    def match(self, claim: AtomicClaim, evidence: str) -> AtomicClaim:
        """Verify a single claim against evidence, returning an updated claim."""
        system = self._build_system_prompt()

        # Epoch 0: use clean prompt (all-in-one, no system), matches direct prompting
        if not system:
            prompt = MATCHER_PROMPT_CLEAN.format(claim=claim.text, evidence=evidence)
        else:
            prompt = MATCHER_PROMPT_EVOLVED.format(claim=claim.text, evidence=evidence)
            # Inject per-claim ReasoningBank rules if available
            if claim.metadata and claim.metadata.get("reasoning_rules"):
                system += "\n" + claim.metadata["reasoning_rules"]

        result = self.llm.generate(prompt, system=system if system else None)
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
        """Build system prompt using prompt refinement strategy.

        Epoch 0: Clean, simple prompt (match direct prompting baseline)
        Epoch 1+: Use evolved prompt if available, otherwise CoT prompt + corrections
        """
        # If we have an evolved prompt from self-refinement, use it directly
        if self.evolved_prompt:
            return self.evolved_prompt

        # Epoch 0: clean prompt, no extras — matches direct prompting
        if self.epoch == 0 or (not self.few_shot_examples and not self.calibration_correction):
            return MATCHER_SYSTEM_CLEAN

        # Later epochs without evolved prompt: CoT + targeted corrections
        prompt = MATCHER_SYSTEM_COT

        # Add calibration bias note (one line, not verbose)
        if self.calibration_correction:
            prompt += "\n" + self.calibration_correction

        # Add only the 3 most recent, most specific corrections
        if self.few_shot_examples:
            recent = self.few_shot_examples[-3:]
            prompt += "\n\nCommon mistakes to avoid:\n"
            for ex in recent:
                prompt += (
                    f"- Claim like \"{ex.get('claim', '')[:80]}\" "
                    f"was wrongly labeled {ex.get('wrong_label', '?')}, "
                    f"correct is {ex.get('correct_label', '?')}: "
                    f"{ex.get('reasoning', '')[:100]}\n"
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
