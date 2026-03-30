"""FActScore baseline (Min et al., EMNLP 2023).

Key idea: Decompose text into atomic facts, then verify each fact
against a knowledge source (Wikipedia). Score = fraction of supported facts.

This is a simplified re-implementation that uses our LLM interface
rather than the original InstructGPT + retrieval pipeline.
"""

from __future__ import annotations

import json

from src.benchmarks.base import BenchmarkSample
from src.llm.base import BaseLLM

from .base import BaseBaseline, BaselineResult

DECOMPOSE_PROMPT = """\
Break the following text into independent atomic facts. \
Each fact should be a single, self-contained statement.

Text: {text}

Return a JSON array of strings. Example: ["Fact 1.", "Fact 2."]
"""

VERIFY_FACT_PROMPT = """\
Given the following atomic fact and knowledge source, determine if the fact \
is supported by the knowledge.

Fact: {fact}

Knowledge:
{knowledge}

Reply with a JSON object:
{{
    "label": "S" if supported by the knowledge, "C" if contradicted, "N" if not enough info,
    "confidence": <float 0-1>,
    "reasoning": "<brief explanation>"
}}
"""


class FActScoreBaseline(BaseBaseline):
    """FActScore: atomic fact decomposition + knowledge-based verification.

    Pipeline:
    1. Decompose response into atomic facts (LLM-based)
    2. For each fact, retrieve relevant knowledge
    3. Verify each fact against knowledge → S/C/N
    4. Score = #supported / #total facts
    """

    name = "factscore"
    description = "Atomic fact decomposition + knowledge verification"

    def __init__(self, llm: BaseLLM, knowledge_base=None):
        """
        Args:
            llm: LLM for decomposition and verification.
            knowledge_base: Optional KnowledgeBase for evidence retrieval.
                If None, uses only evidence provided in the sample.
        """
        self.llm = llm
        self.kb = knowledge_base

    def verify_sample(self, sample: BenchmarkSample) -> BaselineResult:
        # Step 1: Get atomic facts
        if sample.claims:
            # Already decomposed (e.g., FActScore benchmark)
            facts = sample.claims
        elif sample.reference_answer:
            facts = self._decompose(sample.reference_answer)
        else:
            facts = [sample.question]

        # Step 2: Get knowledge/evidence
        knowledge = self._get_knowledge(sample)

        # Step 3: Verify each fact
        claim_labels = []
        claim_details = []
        per_claim_golds = sample.metadata.get("per_claim_labels", [])

        for i, fact in enumerate(facts):
            label, confidence, reasoning = self._verify_fact(fact, knowledge)
            claim_labels.append(label)
            claim_details.append({
                "fact": fact,
                "label": label,
                "confidence": confidence,
                "reasoning": reasoning,
            })

        # Step 4: Compute FActScore = #S / #total
        n_supported = claim_labels.count("S")
        factscore = n_supported / len(claim_labels) if claim_labels else 0.0

        # Map to sample-level label
        predicted_label = self._aggregate_labels(claim_labels)

        claim_gold_labels = per_claim_golds if per_claim_golds else [sample.gold_label] * len(facts)

        return BaselineResult(
            sample_id=sample.id,
            predicted_label=predicted_label,
            gold_label=sample.gold_label,
            confidence=factscore,
            claims=claim_details,
            claim_labels=claim_labels,
            claim_gold_labels=claim_gold_labels,
            metadata={
                "factscore": factscore,
                "n_facts": len(facts),
                "n_supported": n_supported,
                "method": "factscore",
            },
        )

    def _decompose(self, text: str) -> list[str]:
        """Decompose text into atomic facts using LLM."""
        prompt = DECOMPOSE_PROMPT.format(text=text[:2000])
        result = self.llm.generate(prompt)
        text_out = result.text.strip()

        start = text_out.find("[")
        end = text_out.rfind("]") + 1
        if start != -1 and end > 0:
            try:
                facts = json.loads(text_out[start:end])
                return [f for f in facts if isinstance(f, str) and len(f.strip()) > 5]
            except json.JSONDecodeError:
                pass

        # Fallback: split by sentences
        return [s.strip() for s in text.split(".") if len(s.strip()) > 10]

    def _get_knowledge(self, sample: BenchmarkSample) -> str:
        """Retrieve knowledge for verification."""
        parts = []

        # Use sample evidence if available
        if sample.evidence:
            parts.extend(sample.evidence)

        # Use knowledge base if available
        if self.kb and sample.question:
            try:
                docs = self.kb.search(sample.question, top_k=3)
                for doc in docs:
                    if hasattr(doc, "content"):
                        parts.append(doc.content)
                    elif isinstance(doc, dict):
                        parts.append(doc.get("content", str(doc)))
            except Exception:
                pass

        if not parts:
            parts.append("No external knowledge available.")

        return "\n\n".join(parts[:5])

    def _verify_fact(self, fact: str, knowledge: str) -> tuple[str, float, str]:
        """Verify one atomic fact against knowledge."""
        prompt = VERIFY_FACT_PROMPT.format(fact=fact, knowledge=knowledge[:3000])
        result = self.llm.generate(prompt)
        return self._parse_verification(result.text)

    def _parse_verification(self, text: str) -> tuple[str, float, str]:
        """Parse verification result."""
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
                reasoning = data.get("reasoning", "")
                return label, confidence, reasoning
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
