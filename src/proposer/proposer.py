"""Proposer: generates safety probes to stress-test LLM reliability."""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

from src.data.schema import Probe, RiskType
from src.llm.base import BaseLLM

from .templates import DOMAINS, PROPOSER_SYSTEM, RISK_TEMPLATES


class Proposer:
    """Stage 1 — generates verifiable safety probes."""

    def __init__(
        self,
        llm: BaseLLM,
        seed: int = 42,
        memory: list[dict] | None = None,
    ):
        self.llm = llm
        self.rng = random.Random(seed)
        self.memory: list[dict] = memory or []

    def generate_probes(
        self,
        risk_type: RiskType,
        n: int = 10,
        difficulty: str = "hard",
        domain: str | None = None,
    ) -> list[Probe]:
        """Generate n probes for a specific risk type."""
        if domain is None:
            domain = self.rng.choice(DOMAINS)

        template_key = risk_type.value
        template = RISK_TEMPLATES[template_key]

        memory_context = self._format_memory() if self.memory else ""
        prompt = template.format(
            n=n,
            domain=domain,
            difficulty=difficulty,
            memory_context=memory_context,
        )

        response = self.llm.generate(prompt, system=PROPOSER_SYSTEM)
        probes = self._parse_probes(response.text, risk_type)
        return probes

    def generate_all(
        self,
        n_per_type: int = 10,
        risk_weights: dict[str, float] | None = None,
    ) -> list[Probe]:
        """Generate probes across all risk types."""
        all_probes: list[Probe] = []
        for risk_type in RiskType:
            weight = 1.0
            if risk_weights:
                weight = risk_weights.get(risk_type.value, 1.0)
            count = max(1, int(n_per_type * weight))
            probes = self.generate_probes(risk_type, n=count)
            all_probes.extend(probes)
        self.rng.shuffle(all_probes)
        return all_probes

    def update_memory(self, failures: list[dict]) -> None:
        """Incorporate informative failures into memory for next epoch."""
        self.memory.extend(failures)

    def _format_memory(self) -> str:
        if not self.memory:
            return ""
        lines = ["Previously identified failure patterns to target:"]
        for f in self.memory[-20:]:  # Keep last 20 failures
            lines.append(f"- {f.get('pattern', '')} (type: {f.get('failure_type', 'unknown')})")
        return "\n".join(lines)

    def _parse_probes(self, text: str, risk_type: RiskType) -> list[Probe]:
        """Parse LLM output into Probe objects."""
        # Extract JSON array from response
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return self._fallback_parse(text, risk_type)

        try:
            items: list[dict[str, Any]] = json.loads(text[start:end])
        except json.JSONDecodeError:
            return self._fallback_parse(text, risk_type)

        probes = []
        for item in items:
            if "question" not in item:
                continue
            probes.append(
                Probe(
                    id=str(uuid.uuid4()),
                    question=item["question"],
                    risk_type=risk_type,
                    ground_truth=item.get("ground_truth"),
                    metadata={"reasoning": item.get("reasoning", "")},
                )
            )
        return probes

    def _fallback_parse(self, text: str, risk_type: RiskType) -> list[Probe]:
        """Fallback: treat each non-empty line as a question."""
        probes = []
        for line in text.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if len(line) > 10 and line.endswith("?"):
                probes.append(
                    Probe(
                        id=str(uuid.uuid4()),
                        question=line,
                        risk_type=risk_type,
                    )
                )
        return probes
