"""Predicate-based Hybrid Verification (Phase 1 optimization).

Instead of a single LLM call to judge S/C/N, this module:
1. Decomposes a claim into atomic predicates (who/what/when/where/how many)
2. Routes each predicate to the best verification method:
   - Numeric/date → regex + comparison (deterministic, no LLM)
   - Entity name   → fuzzy string match against evidence (deterministic)
   - Semantic       → NLI-framed LLM call (one predicate vs one evidence sentence)
3. Aggregates predicate verdicts into claim-level S/C/N

Inspired by ShieldAgent (2503.22738) predicate decomposition + tool-based verification.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from src.data.schema import AtomicClaim, ClaimLabel
from src.llm.base import BaseLLM


# ================================================================
# Prompts
# ================================================================

PREDICATE_EXTRACT_PROMPT = """\
Extract atomic predicates from this claim. Each predicate should be ONE verifiable fact.

Claim: {claim}

For each predicate, identify its type:
- "numeric": involves a number, date, year, age, quantity, percentage
- "entity": involves a named entity (person, place, organization) and an attribute
- "semantic": involves a relationship, causation, or qualitative assertion

Output JSON array:
[{{"text": "predicate text", "type": "numeric|entity|semantic", "key_value": "the specific value to verify"}}]

Example:
Claim: "Albert Einstein was born in 1879 in Ulm, Germany"
Output: [
  {{"text": "Albert Einstein was born in 1879", "type": "numeric", "key_value": "1879"}},
  {{"text": "Albert Einstein was born in Ulm", "type": "entity", "key_value": "Ulm"}},
  {{"text": "Ulm is in Germany", "type": "entity", "key_value": "Germany"}}
]

Be concise. 1-4 predicates max. If claim is already atomic, return it as a single predicate.
"""

NLI_VERIFY_PROMPT = """\
Natural Language Inference task.

Premise (evidence):
{premise}

Hypothesis (claim):
{hypothesis}

Classify:
- "entailment": premise supports the hypothesis → label "S"
- "contradiction": premise contradicts the hypothesis → label "C"
- "neutral": premise does not address the hypothesis → label "N"

Reply with JSON: {{"label": "S"/"C"/"N", "confidence": <float 0-1>, "reasoning": "<brief>"}}
"""

FALLBACK_VERIFY_PROMPT = """\
Given the following claim and evidence, determine if the claim is:
- "S" (Supported): The evidence supports this claim
- "C" (Contradicted): The evidence contradicts this claim
- "N" (Not enough info): The evidence is insufficient to verify

Claim: {claim}

Evidence:
{evidence}

Reply with JSON: {{"label": "S"/"C"/"N", "confidence": <float 0-1>, "reasoning": "<brief>"}}
"""


# ================================================================
# Predicate types
# ================================================================

class Predicate:
    """An atomic verifiable predicate extracted from a claim."""

    __slots__ = ("text", "pred_type", "key_value", "verdict", "confidence", "method")

    def __init__(self, text: str, pred_type: str, key_value: str = ""):
        self.text = text
        self.pred_type = pred_type  # "numeric", "entity", "semantic"
        self.key_value = key_value
        self.verdict: str | None = None  # "S", "C", "N"
        self.confidence: float = 0.5
        self.method: str = ""  # "regex", "fuzzy", "nli"


# ================================================================
# Core matcher
# ================================================================

class PredicateMatcher:
    """Hybrid verification via predicate decomposition.

    Verification priority:
    1. Numeric predicates → regex extraction + comparison (deterministic)
    2. Entity predicates  → fuzzy string match (deterministic)
    3. Semantic predicates → NLI-framed LLM call

    Self-evolution: evolved_prompt replaces NLI system prompt at epoch > 0.
    """

    def __init__(self, llm: BaseLLM, fuzzy_threshold: float = 0.75):
        self.llm = llm
        self.fuzzy_threshold = fuzzy_threshold
        self.evolved_prompt: str = ""
        self.epoch: int = 0
        # Stats
        self.stats = {"numeric": 0, "entity": 0, "semantic": 0, "fallback": 0}

    def match(self, claim: AtomicClaim, evidence: str) -> AtomicClaim:
        """Verify a claim against evidence using hybrid strategy.

        Strategy:
        1. Extract predicates and try deterministic verification (regex, fuzzy)
        2. If ANY deterministic verdict found with high confidence → use it as signal
        3. Always do a claim-level NLI call for the holistic judgment
        4. Combine: deterministic verdicts can override NLI when they conflict
        """
        claim_text = claim.text.strip()
        evidence_text = evidence.strip()

        if not evidence_text or not claim_text:
            claim.label = ClaimLabel.NOT_MENTIONED
            claim.confidence = 0.3
            return claim

        # Step 1: Extract predicates and try deterministic verification
        predicates = self._extract_predicates(claim_text)
        evidence_sentences = self._split_evidence(evidence_text)

        deterministic_verdicts = []
        for pred in predicates:
            best_ev = self._find_best_evidence(pred, evidence_sentences)
            if pred.pred_type == "numeric":
                self._verify_numeric(pred, best_ev)
            elif pred.pred_type == "entity":
                self._verify_entity(pred, best_ev)
            if pred.verdict is not None:
                deterministic_verdicts.append(pred)

        # Step 2: Claim-level NLI (always, for holistic judgment)
        nli_pred = Predicate(text=claim_text, pred_type="semantic")
        self._verify_nli(nli_pred, evidence_text)

        # Step 3: Combine deterministic + NLI
        label, confidence = self._combine_verdicts(deterministic_verdicts, nli_pred)

        claim.label = label
        claim.confidence = confidence
        claim.metadata = claim.metadata or {}
        claim.metadata["predicates"] = [
            {"text": p.text, "type": p.pred_type, "verdict": p.verdict,
             "confidence": p.confidence, "method": p.method}
            for p in deterministic_verdicts
        ]
        claim.metadata["nli_verdict"] = nli_pred.verdict
        claim.metadata["nli_confidence"] = nli_pred.confidence
        return claim

    # ================================================================
    # Step 1: Predicate extraction
    # ================================================================

    def _extract_predicates(self, claim: str) -> list[Predicate]:
        """Extract atomic predicates from a claim via LLM."""
        # Fast path: very short claims are already atomic
        if len(claim.split()) <= 6:
            pred_type = self._guess_type(claim)
            return [Predicate(text=claim, pred_type=pred_type,
                              key_value=self._extract_key_value(claim, pred_type))]

        prompt = PREDICATE_EXTRACT_PROMPT.format(claim=claim)
        try:
            result = self.llm.generate(prompt)
            text = result.text.strip()
            start, end = text.find("["), text.rfind("]") + 1
            if start != -1 and end > 0:
                items = json.loads(text[start:end])
                predicates = []
                for item in items[:4]:  # Max 4 predicates
                    if isinstance(item, dict) and item.get("text"):
                        ptype = item.get("type", "semantic")
                        if ptype not in ("numeric", "entity", "semantic"):
                            ptype = "semantic"
                        predicates.append(Predicate(
                            text=item["text"],
                            pred_type=ptype,
                            key_value=str(item.get("key_value", "")),
                        ))
                if predicates:
                    return predicates
        except Exception:
            pass

        # Fallback: single predicate
        pred_type = self._guess_type(claim)
        return [Predicate(text=claim, pred_type=pred_type,
                          key_value=self._extract_key_value(claim, pred_type))]

    def _guess_type(self, text: str) -> str:
        """Heuristic type detection without LLM."""
        numbers = re.findall(r'\b\d{2,}\b', text)
        if len(numbers) >= 1:
            return "numeric"
        # Check for capitalized entity names
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if entities:
            return "entity"
        return "semantic"

    def _extract_key_value(self, text: str, pred_type: str) -> str:
        """Extract the key value to verify."""
        if pred_type == "numeric":
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            return numbers[-1] if numbers else ""
        return ""

    # ================================================================
    # Step 2: Evidence targeting
    # ================================================================

    def _split_evidence(self, evidence: str) -> list[str]:
        """Split evidence into sentences."""
        # Split on period, exclamation, question mark followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', evidence)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _find_best_evidence(self, pred: Predicate, sentences: list[str]) -> str:
        """Find the most relevant evidence sentence for a predicate."""
        if not sentences:
            return ""

        pred_words = set(re.findall(r'\w+', pred.text.lower()))

        best_score = 0.0
        best_sent = sentences[0]

        for sent in sentences:
            sent_words = set(re.findall(r'\w+', sent.lower()))
            if not sent_words:
                continue
            overlap = len(pred_words & sent_words)
            score = overlap / max(len(pred_words), 1)
            if score > best_score:
                best_score = score
                best_sent = sent

        # If best match is very weak, return all evidence
        if best_score < 0.15:
            return " ".join(sentences)

        return best_sent

    # ================================================================
    # Step 3a: Numeric verification (deterministic)
    # ================================================================

    def _verify_numeric(self, pred: Predicate, evidence: str) -> None:
        """Verify numeric predicates by extracting and comparing numbers."""
        if not evidence:
            return

        claim_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', pred.text)
        evidence_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', evidence)

        if not claim_numbers or not evidence_numbers:
            return

        # Check if the key value appears in evidence
        key = pred.key_value
        if key and key in evidence:
            pred.verdict = "S"
            pred.confidence = 0.95
            pred.method = "regex_match"
            self.stats["numeric"] += 1
            return

        # Check if key value contradicts evidence numbers
        if key:
            try:
                claim_num = float(key)
            except ValueError:
                return  # Not a pure number (e.g. date string), skip numeric check
            # Look for the same context in evidence with different number
            for ev_num_str in evidence_numbers:
                ev_num = float(ev_num_str)
                if ev_num != claim_num:
                    # Check if they're in similar context
                    claim_context = self._number_context(pred.text, key)
                    ev_context = self._number_context(evidence, ev_num_str)
                    if claim_context and ev_context:
                        sim = SequenceMatcher(None, claim_context.lower(),
                                              ev_context.lower()).ratio()
                        if sim > 0.5:
                            pred.verdict = "C"
                            pred.confidence = 0.90
                            pred.method = "regex_contradict"
                            self.stats["numeric"] += 1
                            return

    def _number_context(self, text: str, number: str) -> str:
        """Get surrounding context of a number in text."""
        idx = text.find(number)
        if idx == -1:
            return ""
        start = max(0, idx - 30)
        end = min(len(text), idx + len(number) + 30)
        return text[start:end]

    # ================================================================
    # Step 3b: Entity verification (deterministic)
    # ================================================================

    def _verify_entity(self, pred: Predicate, evidence: str) -> None:
        """Verify entity predicates via fuzzy string matching."""
        if not evidence:
            return

        # Extract entity names from both claim and evidence
        claim_entities = self._extract_entities(pred.text)
        evidence_entities = self._extract_entities(evidence)

        if not claim_entities or not evidence_entities:
            return

        # Check if claim entities are present in evidence
        matched = 0
        total = len(claim_entities)

        for c_ent in claim_entities:
            best_ratio = 0.0
            for e_ent in evidence_entities:
                ratio = SequenceMatcher(None, c_ent.lower(), e_ent.lower()).ratio()
                best_ratio = max(best_ratio, ratio)

            if best_ratio >= self.fuzzy_threshold:
                matched += 1
            elif best_ratio < 0.4:
                # Entity in claim not found at all in evidence
                # Check if evidence mentions a DIFFERENT entity in same role
                if self._entities_conflict(c_ent, evidence):
                    pred.verdict = "C"
                    pred.confidence = 0.85
                    pred.method = "fuzzy_contradict"
                    self.stats["entity"] += 1
                    return

        if matched == total and total > 0:
            pred.verdict = "S"
            pred.confidence = 0.90
            pred.method = "fuzzy_match"
            self.stats["entity"] += 1

    def _extract_entities(self, text: str) -> list[str]:
        """Extract capitalized entity names."""
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Filter out common sentence starters
        stopwords = {"The", "This", "That", "These", "Those", "It", "He", "She",
                     "They", "We", "You", "His", "Her", "Its", "My", "Our", "Your",
                     "Some", "Many", "Most", "All", "No", "Not", "But", "And", "Or",
                     "If", "When", "Where", "What", "How", "Why", "Who", "Which",
                     "After", "Before", "During", "Since", "Until", "While",
                     "However", "Therefore", "Furthermore", "Moreover", "Although",
                     "Because", "According", "Based"}
        return [e for e in entities if e not in stopwords and len(e) > 1]

    def _entities_conflict(self, claim_entity: str, evidence: str) -> bool:
        """Check if evidence contains a different entity in similar context."""
        # Simple heuristic: if claim says "born in X" and evidence says "born in Y"
        patterns = [
            (r'born\s+in\s+(\w+)', "birthplace"),
            (r'capital\s+of\s+(\w+)', "capital"),
            (r'president\s+of\s+(\w+)', "leadership"),
            (r'located\s+in\s+(\w+)', "location"),
            (r'founded\s+(?:in|by)\s+(\w+)', "founding"),
        ]
        for pattern, _ in patterns:
            claim_match = re.search(pattern, claim_entity, re.IGNORECASE)
            ev_match = re.search(pattern, evidence, re.IGNORECASE)
            if claim_match and ev_match:
                if claim_match.group(1).lower() != ev_match.group(1).lower():
                    return True
        return False

    # ================================================================
    # Step 3c: Semantic verification (NLI-framed LLM)
    # ================================================================

    def _verify_nli(self, pred: Predicate, evidence: str) -> None:
        """Verify via NLI-framed LLM call."""
        if not evidence:
            pred.verdict = "N"
            pred.confidence = 0.4
            pred.method = "no_evidence"
            return

        system = self.evolved_prompt if self.evolved_prompt else None
        prompt = NLI_VERIFY_PROMPT.format(premise=evidence[:1500], hypothesis=pred.text)

        try:
            result = self.llm.generate(prompt, system=system)
            label, confidence = self._parse_nli(result.text)
            pred.verdict = label
            pred.confidence = confidence
            pred.method = "nli"
            self.stats["semantic"] += 1
        except Exception:
            pred.verdict = "N"
            pred.confidence = 0.4
            pred.method = "nli_error"

    def _parse_nli(self, text: str) -> tuple[str, float]:
        """Parse NLI result JSON."""
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
                return label, max(0.0, min(1.0, confidence))
            except (json.JSONDecodeError, ValueError):
                pass
        return "N", 0.4

    # ================================================================
    # Step 4: Aggregation
    # ================================================================

    def _combine_verdicts(self, det_preds: list[Predicate], nli_pred: Predicate) -> tuple[ClaimLabel, float]:
        """Combine deterministic predicate verdicts with claim-level NLI.

        Priority:
        1. Deterministic contradiction (high confidence) → C (overrides NLI)
        2. Deterministic support + NLI support → S (boosted confidence)
        3. Deterministic support + NLI contradiction → trust NLI (it sees full context)
        4. No deterministic signal → trust NLI entirely
        """
        label_map = {"S": ClaimLabel.SUPPORTED, "C": ClaimLabel.CONTRADICTED,
                     "N": ClaimLabel.NOT_MENTIONED}

        nli_label = nli_pred.verdict or "N"
        nli_conf = nli_pred.confidence

        if not det_preds:
            # No deterministic signal, trust NLI
            return label_map.get(nli_label, ClaimLabel.NOT_MENTIONED), nli_conf

        # Check for deterministic contradictions
        det_contradictions = [p for p in det_preds if p.verdict == "C" and p.confidence >= 0.80]
        det_supports = [p for p in det_preds if p.verdict == "S" and p.confidence >= 0.80]

        if det_contradictions:
            # Deterministic contradiction overrides NLI
            best_conf = max(p.confidence for p in det_contradictions)
            return ClaimLabel.CONTRADICTED, max(best_conf, nli_conf if nli_label == "C" else best_conf)

        if det_supports and nli_label == "S":
            # Both agree on support → boost confidence
            boosted = min(1.0, nli_conf + 0.1)
            return ClaimLabel.SUPPORTED, boosted

        if det_supports and nli_label == "C":
            # Conflict: deterministic says S but NLI says C
            # Trust NLI — it sees the full claim-evidence relationship
            return ClaimLabel.CONTRADICTED, nli_conf * 0.9

        if det_supports and nli_label == "N":
            # Deterministic found matching values but NLI says not enough info
            # Slight lean toward S if deterministic is strong
            avg_det_conf = sum(p.confidence for p in det_supports) / len(det_supports)
            if avg_det_conf >= 0.90:
                return ClaimLabel.SUPPORTED, avg_det_conf * 0.85
            return ClaimLabel.NOT_MENTIONED, nli_conf

        # Default: trust NLI
        return label_map.get(nli_label, ClaimLabel.NOT_MENTIONED), nli_conf

    def _aggregate(self, predicates: list[Predicate]) -> tuple[ClaimLabel, float]:
        """Aggregate predicate verdicts into claim-level label.

        Rules:
        1. Any high-confidence contradiction → C
        2. All supported → S
        3. Mix or unknown → weighted vote
        """
        if not predicates:
            return ClaimLabel.NOT_MENTIONED, 0.5

        verdicts = [p.verdict or "N" for p in predicates]
        confidences = [p.confidence for p in predicates]

        # Rule 1: High-confidence contradiction dominates
        for v, c in zip(verdicts, confidences):
            if v == "C" and c >= 0.75:
                return ClaimLabel.CONTRADICTED, c

        # Rule 2: All supported
        if all(v == "S" for v in verdicts):
            avg_conf = sum(confidences) / len(confidences)
            return ClaimLabel.SUPPORTED, avg_conf

        # Rule 3: Weighted vote
        label_weight = {"S": 0.0, "C": 0.0, "N": 0.0}
        for v, c in zip(verdicts, confidences):
            label_weight[v] += c

        # Tie-breaking: C > N > S (conservative for hallucination detection)
        priority = {"C": 2, "N": 1, "S": 0}
        best = max(label_weight, key=lambda l: (label_weight[l], priority[l]))

        label_map = {"S": ClaimLabel.SUPPORTED, "C": ClaimLabel.CONTRADICTED,
                     "N": ClaimLabel.NOT_MENTIONED}
        avg_conf = sum(confidences) / len(confidences)
        return label_map[best], avg_conf

    # ================================================================
    # Fallback: single LLM call (for very simple claims)
    # ================================================================

    def _fallback_verify(self, claim: AtomicClaim, evidence: str) -> AtomicClaim:
        """Fallback to single LLM call when predicate extraction fails."""
        self.stats["fallback"] += 1
        system = self.evolved_prompt if self.evolved_prompt else None
        prompt = FALLBACK_VERIFY_PROMPT.format(claim=claim.text, evidence=evidence)

        try:
            result = self.llm.generate(prompt, system=system)
            label, confidence = self._parse_nli(result.text)
            label_map = {"S": ClaimLabel.SUPPORTED, "C": ClaimLabel.CONTRADICTED,
                         "N": ClaimLabel.NOT_MENTIONED}
            claim.label = label_map.get(label, ClaimLabel.NOT_MENTIONED)
            claim.confidence = confidence
        except Exception:
            claim.label = ClaimLabel.NOT_MENTIONED
            claim.confidence = 0.3

        return claim

    def get_stats(self) -> dict:
        """Return verification method usage stats."""
        total = sum(self.stats.values())
        return {**self.stats, "total": total,
                "deterministic_ratio": (self.stats["numeric"] + self.stats["entity"]) / max(total, 1)}
