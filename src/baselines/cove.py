"""Chain-of-Verification (CoVe) baseline (Dhuliawala et al., Meta 2023).

Key idea: LLM self-verifies by generating verification questions about
its own claims, answering them independently, then checking consistency.

Pipeline:
1. Generate initial response
2. Plan verification questions for key claims
3. Answer each verification question independently
4. Check for inconsistencies between original claims and verification answers
"""

from __future__ import annotations

import json

from src.benchmarks.base import BenchmarkSample
from src.llm.base import BaseLLM

from .base import BaseBaseline, BaselineResult

PLAN_VERIFICATION_PROMPT = """\
Given the following claim, generate 2-3 specific verification questions \
that, if answered correctly, would confirm or refute the claim. \
Each question should target a different factual aspect of the claim.

Claim: {claim}

Return a JSON array of verification questions.
Example: ["What year was X born?", "Where did X work?"]
"""

VERIFY_ANSWER_PROMPT = """\
Answer the following factual question. Be concise and specific. \
If you are unsure, say "I'm not sure".

Question: {question}
"""

CHECK_CONSISTENCY_PROMPT = """\
Given an original claim and answers to verification questions about it, \
determine if the claim is consistent with the verification answers.

Original claim: {claim}

Verification Q&A:
{qa_pairs}

Is the original claim consistent with the verification answers?
Reply with JSON:
{{
    "label": "S" if claim is consistent/verified, "C" if contradicted, "N" if inconclusive,
    "confidence": <float 0-1>,
    "inconsistencies": [<list any contradictions found>],
    "reasoning": "<brief explanation>"
}}
"""


class CoVeBaseline(BaseBaseline):
    """Chain-of-Verification: LLM self-verification through question decomposition.

    For each claim:
    1. Generate verification questions
    2. Answer each question independently (no access to original claim)
    3. Compare answers with original claim
    4. Flag inconsistencies as potential hallucinations
    """

    name = "cove"
    description = "Chain-of-Verification: LLM self-verification via question decomposition"

    def __init__(self, llm: BaseLLM, n_questions: int = 3):
        self.llm = llm
        self.n_questions = n_questions

    def verify_sample(self, sample: BenchmarkSample) -> BaselineResult:
        if sample.claims:
            claims = sample.claims
        else:
            claims = [sample.question]

        claim_labels = []
        claim_details = []
        per_claim_golds = sample.metadata.get("per_claim_labels", [])

        for claim in claims:
            # Step 1: Generate verification questions
            questions = self._plan_verification(claim)

            # Step 2: Answer each question independently
            qa_pairs = []
            for q in questions[:self.n_questions]:
                answer = self._answer_question(q)
                qa_pairs.append({"question": q, "answer": answer})

            # Step 3: Check consistency
            label, confidence, inconsistencies = self._check_consistency(
                claim, qa_pairs
            )

            claim_labels.append(label)
            claim_details.append({
                "claim": claim,
                "verification_qa": qa_pairs,
                "label": label,
                "confidence": confidence,
                "inconsistencies": inconsistencies,
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
            metadata={"n_questions_per_claim": self.n_questions, "method": "cove"},
        )

    def _plan_verification(self, claim: str) -> list[str]:
        """Generate verification questions for a claim."""
        prompt = PLAN_VERIFICATION_PROMPT.format(claim=claim)
        result = self.llm.generate(prompt)
        text = result.text.strip()

        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > 0:
            try:
                questions = json.loads(text[start:end])
                return [q for q in questions if isinstance(q, str)][:self.n_questions]
            except json.JSONDecodeError:
                pass

        return [f"Is it true that {claim}?"]

    def _answer_question(self, question: str) -> str:
        """Answer a verification question independently."""
        prompt = VERIFY_ANSWER_PROMPT.format(question=question)
        result = self.llm.generate(prompt)
        return result.text.strip()[:500]

    def _check_consistency(
        self, claim: str, qa_pairs: list[dict]
    ) -> tuple[str, float, list[str]]:
        """Check if claim is consistent with verification answers."""
        qa_text = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs
        )

        prompt = CHECK_CONSISTENCY_PROMPT.format(claim=claim, qa_pairs=qa_text)
        result = self.llm.generate(prompt)
        return self._parse_consistency(result.text)

    def _parse_consistency(self, text: str) -> tuple[str, float, list[str]]:
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
                inconsistencies = data.get("inconsistencies", [])
                if not isinstance(inconsistencies, list):
                    inconsistencies = []
                return label, confidence, inconsistencies
            except (json.JSONDecodeError, ValueError):
                pass

        return "N", 0.4, []

    def _aggregate_labels(self, labels: list[str]) -> str:
        if not labels:
            return "N"
        if "C" in labels:
            return "C"
        if all(l == "S" for l in labels):
            return "S"
        return "N"
