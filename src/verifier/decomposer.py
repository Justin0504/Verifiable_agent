"""Claim Decomposition: break a response into atomic, verifiable claims."""

from __future__ import annotations

import json
import uuid

from src.data.schema import AtomicClaim
from src.llm.base import BaseLLM

DECOMPOSE_SYSTEM = """\
You are a claim decomposition engine. Given a question and a model's response, \
extract every distinct factual claim made in the response.

Rules:
1. Each claim must be atomic — it asserts exactly one fact
2. Each claim must be self-contained — understandable without the original context
3. Opinions, hedges ("I think"), and meta-statements ("As an AI") are NOT claims
4. Preserve the original meaning — do not add or remove information
5. Number claims sequentially

Output format: Return a JSON array of strings, each being one atomic claim.
Example: ["Albert Einstein was born in 1879.", "He was born in Ulm, Germany."]
"""

DECOMPOSE_PROMPT = """\
Question: {question}

Response: {response}

Extract all atomic factual claims from the response above.
"""


class Decomposer:
    """Decompose a model response into atomic claims.

    Supports self-evolution: accumulated few-shot examples from past
    decomposition failures (missed claims, over-split claims) improve
    decomposition quality across epochs.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.few_shot_examples: list[dict] = []

    def decompose(self, question: str, response_text: str) -> list[AtomicClaim]:
        """Break a response into atomic claims."""
        prompt = DECOMPOSE_PROMPT.format(question=question, response=response_text)
        system = self._build_system_prompt()
        result = self.llm.generate(prompt, system=system)
        return self._parse_claims(result.text)

    def update_few_shots(self, examples: list[dict]) -> None:
        """Add corrected decomposition examples from past failures.

        Each example should have: response, missed_claims or over_split, reasoning.
        """
        self.few_shot_examples.extend(examples)
        if len(self.few_shot_examples) > 10:
            self.few_shot_examples = self.few_shot_examples[-10:]

    def _build_system_prompt(self) -> str:
        """Build system prompt, injecting few-shot corrections if available."""
        if not self.few_shot_examples:
            return DECOMPOSE_SYSTEM

        examples_text = "\n\nLearn from these past decomposition errors:\n"
        for i, ex in enumerate(self.few_shot_examples, 1):
            examples_text += (
                f"\nCorrection {i}:\n"
                f"  Response snippet: {ex.get('response', '')[:200]}\n"
                f"  Issue: {ex.get('issue', '')}\n"
                f"  Lesson: {ex.get('reasoning', '')}\n"
            )

        return DECOMPOSE_SYSTEM + examples_text

    def _parse_claims(self, text: str) -> list[AtomicClaim]:
        """Parse LLM output into AtomicClaim objects."""
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1

        claims: list[str] = []
        if start != -1 and end > 0:
            try:
                claims = json.loads(text[start:end])
            except json.JSONDecodeError:
                claims = self._fallback_parse(text)
        else:
            claims = self._fallback_parse(text)

        return [
            AtomicClaim(id=str(uuid.uuid4()), text=c)
            for c in claims
            if isinstance(c, str) and len(c.strip()) > 5
        ]

    def _fallback_parse(self, text: str) -> list[str]:
        """Fallback: split by newlines, strip numbering."""
        claims = []
        for line in text.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if len(line) > 5:
                claims.append(line)
        return claims
