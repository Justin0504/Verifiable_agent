"""Direct Prompting baseline — the simplest possible approach.

Just ask the LLM: "Is this claim supported, contradicted, or not verifiable?"
No decomposition, no retrieval, no tools. Pure LLM parametric knowledge.

This serves as the lower-bound baseline — any method that can't beat this
is not worth the added complexity.
"""

from __future__ import annotations

import json

from src.benchmarks.base import BenchmarkSample
from src.llm.base import BaseLLM

from .base import BaseBaseline, BaselineResult

DIRECT_PROMPT_WITH_EVIDENCE = """\
Given the following claim and evidence, determine if the claim is:
- "S" (Supported): The evidence supports this claim
- "C" (Contradicted): The evidence contradicts this claim
- "N" (Not enough info): The evidence is insufficient to verify

Claim: {claim}

Evidence:
{evidence}

Reply with JSON: {{"label": "S"/"C"/"N", "confidence": <float 0-1>, "reasoning": "<brief>"}}
"""

DIRECT_PROMPT_NO_EVIDENCE = """\
Based on your knowledge, determine if the following claim is:
- "S" (Supported): This is factually correct
- "C" (Contradicted): This is factually incorrect
- "N" (Not enough info): Cannot be determined

Claim: {claim}

Reply with JSON: {{"label": "S"/"C"/"N", "confidence": <float 0-1>, "reasoning": "<brief>"}}
"""


class DirectPromptingBaseline(BaseBaseline):
    """Direct Prompting: ask LLM to classify claims with no pipeline."""

    name = "direct_prompting"
    description = "Directly ask LLM to verify claims (no decomposition/retrieval)"

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def verify_sample(self, sample: BenchmarkSample) -> BaselineResult:
        if sample.claims:
            claims = sample.claims
        else:
            claims = [sample.question]

        evidence = "\n".join(sample.evidence[:3]) if sample.evidence else ""

        claim_labels = []
        claim_details = []
        per_claim_golds = sample.metadata.get("per_claim_labels", [])

        for claim in claims:
            if evidence:
                prompt = DIRECT_PROMPT_WITH_EVIDENCE.format(
                    claim=claim, evidence=evidence[:2000]
                )
            else:
                prompt = DIRECT_PROMPT_NO_EVIDENCE.format(claim=claim)

            result = self.llm.generate(prompt)
            label, confidence, reasoning = self._parse(result.text)

            claim_labels.append(label)
            claim_details.append({
                "claim": claim,
                "label": label,
                "confidence": confidence,
                "reasoning": reasoning,
            })

        predicted_label = self._aggregate_labels(claim_labels)
        claim_gold_labels = per_claim_golds if per_claim_golds else [sample.gold_label] * len(claims)

        return BaselineResult(
            sample_id=sample.id,
            predicted_label=predicted_label,
            gold_label=sample.gold_label,
            confidence=sum(d["confidence"] for d in claim_details) / len(claim_details) if claim_details else 0.5,
            claims=claim_details,
            claim_labels=claim_labels,
            claim_gold_labels=claim_gold_labels,
            metadata={"method": "direct_prompting"},
        )

    def _parse(self, text: str) -> tuple[str, float, str]:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > 0:
            try:
                data = json.loads(text[start:end])
                label = data.get("label", "N").upper().strip()
                if label not in ("S", "C", "N"):
                    label = "N"
                return label, float(data.get("confidence", 0.5)), data.get("reasoning", "")
            except (json.JSONDecodeError, ValueError):
                pass
        return "N", 0.4, "Parse error"

    def _aggregate_labels(self, labels: list[str]) -> str:
        if not labels:
            return "N"
        if "C" in labels:
            return "C"
        if all(l == "S" for l in labels):
            return "S"
        return "N"
