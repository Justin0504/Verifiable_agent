"""Retrieve + NLI baseline.

Traditional pipeline: BM25/TF-IDF retrieval + Natural Language Inference.
Represents the pre-LLM approach to fact verification.

Without a dedicated NLI model (DeBERTa-large-mnli), we use LLM-based NLI
as a proxy, which is still a valid comparison point — it tests whether
structured retrieval + classification beats our holistic pipeline.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from src.benchmarks.base import BenchmarkSample
from src.llm.base import BaseLLM

from .base import BaseBaseline, BaselineResult

NLI_PROMPT = """\
Natural Language Inference task.

Premise (evidence):
{premise}

Hypothesis (claim):
{hypothesis}

Classify the relationship:
- "entailment" → the premise supports the hypothesis (label: S)
- "contradiction" → the premise contradicts the hypothesis (label: C)
- "neutral" → the premise neither supports nor contradicts (label: N)

Reply with JSON:
{{
    "relationship": "entailment"/"contradiction"/"neutral",
    "label": "S"/"C"/"N",
    "confidence": <float 0-1>
}}
"""


class RetrieveNLIBaseline(BaseBaseline):
    """Retrieve + NLI: traditional fact verification pipeline.

    Pipeline:
    1. Retrieve top-k evidence passages (TF-IDF or from sample)
    2. For each claim, run NLI against retrieved evidence
    3. Aggregate NLI predictions
    """

    name = "retrieve_nli"
    description = "Traditional retrieve-then-NLI fact verification pipeline"

    def __init__(self, llm: BaseLLM, knowledge_base=None, top_k: int = 3):
        """
        Args:
            llm: LLM used as NLI classifier.
            knowledge_base: Optional KnowledgeBase for retrieval.
            top_k: Number of evidence passages to retrieve.
        """
        self.llm = llm
        self.kb = knowledge_base
        self.top_k = top_k

    def verify_sample(self, sample: BenchmarkSample) -> BaselineResult:
        if sample.claims:
            claims = sample.claims
        else:
            claims = [sample.question]

        # Retrieve evidence
        evidence_passages = self._retrieve(sample)

        claim_labels = []
        claim_details = []
        per_claim_golds = sample.metadata.get("per_claim_labels", [])

        for claim in claims:
            # Run NLI against each evidence passage, take best match
            best_label = "N"
            best_confidence = 0.0
            best_reasoning = ""

            for passage in evidence_passages:
                label, confidence = self._nli_classify(claim, passage)
                if confidence > best_confidence:
                    best_label = label
                    best_confidence = confidence
                    best_reasoning = f"Matched against: {passage[:100]}..."

            # If no evidence matched well, try TF-IDF similarity as fallback
            if best_confidence < 0.4 and evidence_passages:
                best_passage = self._best_tfidf_match(claim, evidence_passages)
                if best_passage:
                    label, confidence = self._nli_classify(claim, best_passage)
                    if confidence > best_confidence:
                        best_label = label
                        best_confidence = confidence

            claim_labels.append(best_label)
            claim_details.append({
                "claim": claim,
                "label": best_label,
                "confidence": best_confidence,
                "reasoning": best_reasoning,
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
            metadata={
                "n_evidence": len(evidence_passages),
                "method": "retrieve_nli",
            },
        )

    def _retrieve(self, sample: BenchmarkSample) -> list[str]:
        """Retrieve evidence passages."""
        passages = []

        # Use sample evidence
        if sample.evidence:
            passages.extend(sample.evidence)

        # Use knowledge base
        if self.kb:
            query = sample.question
            try:
                docs = self.kb.search(query, top_k=self.top_k)
                for doc in docs:
                    content = doc.content if hasattr(doc, "content") else str(doc)
                    passages.append(content)
            except Exception:
                pass

        return passages[:self.top_k * 2] if passages else ["No evidence available."]

    def _nli_classify(self, hypothesis: str, premise: str) -> tuple[str, float]:
        """Run NLI classification on a claim-evidence pair."""
        prompt = NLI_PROMPT.format(premise=premise[:2000], hypothesis=hypothesis)
        result = self.llm.generate(prompt)
        return self._parse_nli(result.text)

    def _parse_nli(self, text: str) -> tuple[str, float]:
        """Parse NLI result."""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end > 0:
            try:
                data = json.loads(text[start:end])
                label = data.get("label", "N").upper().strip()
                if label not in ("S", "C", "N"):
                    rel = data.get("relationship", "").lower()
                    if "entail" in rel:
                        label = "S"
                    elif "contradict" in rel:
                        label = "C"
                    else:
                        label = "N"
                confidence = float(data.get("confidence", 0.5))
                return label, confidence
            except (json.JSONDecodeError, ValueError):
                pass

        return "N", 0.3

    def _best_tfidf_match(self, query: str, passages: list[str]) -> str | None:
        """Simple TF-IDF-like scoring to find best matching passage."""
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return None

        best_score = 0.0
        best_passage = None

        for passage in passages:
            passage_tokens = set(self._tokenize(passage))
            if not passage_tokens:
                continue
            overlap = len(query_tokens & passage_tokens)
            score = overlap / (math.sqrt(len(query_tokens)) * math.sqrt(len(passage_tokens)))
            if score > best_score:
                best_score = score
                best_passage = passage

        return best_passage

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 2]

    def _aggregate_labels(self, labels: list[str]) -> str:
        if not labels:
            return "N"
        if "C" in labels:
            return "C"
        if all(l == "S" for l in labels):
            return "S"
        return "N"
