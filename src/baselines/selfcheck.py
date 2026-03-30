"""SelfCheckGPT baseline (Manakul et al., EMNLP 2023).

Key idea: Sample multiple responses from the same LLM, then check consistency.
If a claim appears in most samples → likely factual.
If a claim is inconsistent across samples → likely hallucination.

No external knowledge needed — purely self-consistency based.
"""

from __future__ import annotations

import json

from src.benchmarks.base import BenchmarkSample
from src.llm.base import BaseLLM

from .base import BaseBaseline, BaselineResult

CONSISTENCY_PROMPT = """\
Given the following claim and {n_samples} sampled responses to the same question, \
determine if the claim is consistent across the responses.

Question: {question}

Claim: {claim}

Sampled responses:
{samples}

Is this claim consistently stated across the sampled responses?
Reply with a JSON object:
{{
    "consistent": true/false,
    "occurrences": <number of responses that support this claim>,
    "contradictions": <number of responses that contradict this claim>,
    "label": "S" if consistently supported, "C" if contradicted, "N" if unclear,
    "confidence": <float 0-1>
}}
"""


class SelfCheckGPTBaseline(BaseBaseline):
    """SelfCheckGPT: detect hallucinations via sampling consistency.

    For each question:
    1. Generate N stochastic responses (high temperature)
    2. For each claim, check if it appears consistently across samples
    3. Inconsistent claims → likely hallucinations
    """

    name = "selfcheck_gpt"
    description = "Sampling consistency-based hallucination detection (no external knowledge)"

    def __init__(self, llm: BaseLLM, n_samples: int = 5, sample_temperature: float = 1.0):
        self.llm = llm
        self.n_samples = n_samples
        self.sample_temperature = sample_temperature

    def verify_sample(self, sample: BenchmarkSample) -> BaselineResult:
        question = sample.question

        # Step 1: Generate N stochastic samples
        original_temp = self.llm.temperature
        self.llm.temperature = self.sample_temperature

        sampled_responses = []
        for _ in range(self.n_samples):
            resp = self.llm.generate(question)
            sampled_responses.append(resp.text)

        self.llm.temperature = original_temp

        # Step 2: Determine claims to verify
        if sample.claims:
            claims = sample.claims
        else:
            claims = [question]  # For FEVER-style, the question IS the claim

        # Step 3: Check each claim for consistency
        claim_labels = []
        claim_details = []
        per_claim_golds = sample.metadata.get("per_claim_labels", [])

        for i, claim in enumerate(claims):
            samples_text = "\n".join(
                f"Response {j+1}: {resp[:500]}" for j, resp in enumerate(sampled_responses)
            )

            prompt = CONSISTENCY_PROMPT.format(
                question=question,
                claim=claim,
                samples=samples_text,
                n_samples=self.n_samples,
            )

            result = self.llm.generate(prompt)
            label, confidence = self._parse_consistency(result.text)
            claim_labels.append(label)
            claim_details.append({
                "claim": claim,
                "label": label,
                "confidence": confidence,
            })

        # Step 4: Aggregate to sample-level label
        predicted_label = self._aggregate_labels(claim_labels)

        # Gold labels for claim-level eval
        claim_gold_labels = per_claim_golds if per_claim_golds else [sample.gold_label] * len(claims)

        return BaselineResult(
            sample_id=sample.id,
            predicted_label=predicted_label,
            gold_label=sample.gold_label,
            confidence=sum(d["confidence"] for d in claim_details) / len(claim_details) if claim_details else 0.5,
            claims=claim_details,
            claim_labels=claim_labels,
            claim_gold_labels=claim_gold_labels,
            metadata={"n_samples": self.n_samples, "method": "selfcheck_gpt"},
        )

    def _parse_consistency(self, text: str) -> tuple[str, float]:
        """Parse consistency check result."""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end > 0:
            try:
                data = json.loads(text[start:end])
                label = data.get("label", "N").upper().strip()
                if label not in ("S", "C", "N"):
                    label = "N"
                confidence = float(data.get("confidence", 0.5))
                return label, confidence
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback heuristic
        lower = text.lower()
        if "contradict" in lower or "inconsistent" in lower:
            return "C", 0.6
        elif "consistent" in lower or "supported" in lower:
            return "S", 0.6
        return "N", 0.4

    def _aggregate_labels(self, labels: list[str]) -> str:
        """Aggregate claim-level labels to sample-level."""
        if not labels:
            return "N"
        # If any claim is contradicted, the sample is contradicted
        if "C" in labels:
            return "C"
        if all(l == "S" for l in labels):
            return "S"
        return "N"
