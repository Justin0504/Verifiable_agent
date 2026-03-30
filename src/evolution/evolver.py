"""Evolver: update Proposer strategy based on informative failures."""

from __future__ import annotations

import json

from src.llm.base import BaseLLM

from .failure_extractor import InformativeFailure

EVOLUTION_SYSTEM = """\
You are a meta-learning agent that analyzes failure patterns from a hallucination \
detection pipeline. Your job is to identify novel attack patterns that should be \
added to the Proposer's probe generation strategy.

Given a set of informative failures, extract generalizable patterns that describe \
new types of probes the Proposer should generate in the next epoch.

Output format: Return a JSON array of objects, each with:
- "pattern": a concise description of the failure pattern
- "failure_type": "false_positive" | "false_negative" | "boundary"
- "suggestion": a concrete probe generation strategy to exploit this pattern
- "priority": "high" | "medium" | "low"
"""

EVOLUTION_PROMPT = """\
Epoch {epoch} failure analysis.

Total failures: {total}
By type: {by_type}
By risk category: {by_risk}

Sample failures:
{samples}

Analyze these failures and suggest novel probe patterns for the next epoch.
"""


class Evolver:
    """Update Proposer memory/strategy based on discovered failures."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def evolve(
        self,
        failures: list[InformativeFailure],
        epoch: int,
        max_samples: int = 20,
    ) -> list[dict]:
        """Analyze failures and produce new probe strategies.

        Returns a list of strategy dicts to be added to Proposer memory.
        """
        if not failures:
            return []

        # Summarize
        by_type: dict[str, int] = {}
        by_risk: dict[str, int] = {}
        for f in failures:
            by_type[f.failure_type] = by_type.get(f.failure_type, 0) + 1
            by_risk[f.risk_type] = by_risk.get(f.risk_type, 0) + 1

        # Sample representative failures
        samples = failures[:max_samples]
        sample_text = "\n".join(
            f"- [{f.failure_type}] risk={f.risk_type} | {f.pattern}"
            for f in samples
        )

        prompt = EVOLUTION_PROMPT.format(
            epoch=epoch,
            total=len(failures),
            by_type=json.dumps(by_type),
            by_risk=json.dumps(by_risk),
            samples=sample_text,
        )

        result = self.llm.generate(prompt, system=EVOLUTION_SYSTEM)
        return self._parse_strategies(result.text)

    def _parse_strategies(self, text: str) -> list[dict]:
        """Parse evolved strategies from LLM output."""
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            strategies = json.loads(text[start:end])
            return [s for s in strategies if isinstance(s, dict) and "pattern" in s]
        except json.JSONDecodeError:
            return []
