"""SAFE baseline (Wei et al., 2024, Google DeepMind).

Search-Augmented Factual Evaluation:
1. Decompose into atomic facts
2. For each fact, generate search queries
3. Retrieve evidence via search
4. Rate each fact as supported/not supported

We re-implement the core logic using our LLM + tool interfaces.
Without a live search API, we fall back to knowledge base retrieval.
"""

from __future__ import annotations

import json

from src.benchmarks.base import BenchmarkSample
from src.llm.base import BaseLLM

from .base import BaseBaseline, BaselineResult

GENERATE_QUERIES_PROMPT = """\
Given the following claim, generate 2-3 search queries that would help \
verify whether this claim is true or false. Focus on the key factual \
assertions that can be checked.

Claim: {claim}

Return a JSON array of search query strings.
"""

RATE_WITH_EVIDENCE_PROMPT = """\
You are a fact-checking assistant. Given a claim and search results, \
determine if the claim is supported, contradicted, or unverifiable.

Claim: {claim}

Search Results:
{evidence}

Rate the claim:
- "S" (Supported): Search results confirm the claim
- "C" (Contradicted): Search results contradict the claim
- "N" (Irrelevant): Search results don't provide enough info

Reply with JSON:
{{
    "label": "S"/"C"/"N",
    "confidence": <float 0-1>,
    "key_evidence": "<most relevant evidence snippet>",
    "reasoning": "<brief explanation>"
}}
"""


class SAFEBaseline(BaseBaseline):
    """SAFE: Search-Augmented Factual Evaluation.

    Pipeline per claim:
    1. Generate search queries from the claim
    2. Execute search (or KB retrieval as fallback)
    3. Rate the claim against retrieved evidence
    4. Aggregate ratings across all claims
    """

    name = "safe"
    description = "Search-augmented factual evaluation (Google DeepMind)"

    def __init__(self, llm: BaseLLM, knowledge_base=None, search_tool=None):
        """
        Args:
            llm: LLM for query generation and rating.
            knowledge_base: KnowledgeBase for evidence retrieval (fallback).
            search_tool: Optional web search tool. If None, uses KB only.
        """
        self.llm = llm
        self.kb = knowledge_base
        self.search_tool = search_tool

    def verify_sample(self, sample: BenchmarkSample) -> BaselineResult:
        # Get claims
        if sample.claims:
            claims = sample.claims
        else:
            claims = [sample.question]

        claim_labels = []
        claim_details = []
        per_claim_golds = sample.metadata.get("per_claim_labels", [])

        for i, claim in enumerate(claims):
            # Step 1: Generate search queries
            queries = self._generate_queries(claim)

            # Step 2: Retrieve evidence
            evidence = self._search(queries, sample)

            # Step 3: Rate claim against evidence
            label, confidence, reasoning = self._rate_claim(claim, evidence)

            claim_labels.append(label)
            claim_details.append({
                "claim": claim,
                "queries": queries,
                "label": label,
                "confidence": confidence,
                "reasoning": reasoning,
            })

        # Aggregate
        predicted_label = self._aggregate_labels(claim_labels)
        n_supported = claim_labels.count("S")
        safe_score = n_supported / len(claim_labels) if claim_labels else 0.0

        claim_gold_labels = per_claim_golds if per_claim_golds else [sample.gold_label] * len(claims)

        return BaselineResult(
            sample_id=sample.id,
            predicted_label=predicted_label,
            gold_label=sample.gold_label,
            confidence=safe_score,
            claims=claim_details,
            claim_labels=claim_labels,
            claim_gold_labels=claim_gold_labels,
            metadata={
                "safe_score": safe_score,
                "n_claims": len(claims),
                "n_supported": n_supported,
                "method": "safe",
            },
        )

    def _generate_queries(self, claim: str) -> list[str]:
        """Generate search queries for a claim."""
        prompt = GENERATE_QUERIES_PROMPT.format(claim=claim)
        result = self.llm.generate(prompt)
        text = result.text.strip()

        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > 0:
            try:
                queries = json.loads(text[start:end])
                return [q for q in queries if isinstance(q, str)][:3]
            except json.JSONDecodeError:
                pass

        return [claim]  # Fallback: use claim as query

    def _search(self, queries: list[str], sample: BenchmarkSample) -> str:
        """Execute search queries and return combined evidence."""
        evidence_parts = []

        # Include sample-provided evidence first
        if sample.evidence:
            evidence_parts.extend(sample.evidence[:3])

        # Try search tool
        if self.search_tool:
            for query in queries[:2]:
                try:
                    result = self.search_tool.execute(query)
                    if result:
                        evidence_parts.append(str(result)[:500])
                except Exception:
                    pass

        # Fall back to knowledge base
        if self.kb:
            for query in queries[:2]:
                try:
                    docs = self.kb.search(query, top_k=2)
                    for doc in docs:
                        content = doc.content if hasattr(doc, "content") else str(doc)
                        evidence_parts.append(content[:500])
                except Exception:
                    pass

        if not evidence_parts:
            evidence_parts.append("No search results available.")

        return "\n\n".join(evidence_parts[:5])

    def _rate_claim(self, claim: str, evidence: str) -> tuple[str, float, str]:
        """Rate a claim against search evidence."""
        prompt = RATE_WITH_EVIDENCE_PROMPT.format(
            claim=claim, evidence=evidence[:3000]
        )
        result = self.llm.generate(prompt)
        return self._parse_rating(result.text)

    def _parse_rating(self, text: str) -> tuple[str, float, str]:
        """Parse rating result."""
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
