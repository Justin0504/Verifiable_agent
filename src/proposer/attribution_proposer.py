"""Attribution Proposer: generates adversarial (claim, source) pairs.

Targets 4 ambiguity types that make attribution verification hard:
1. Paraphrase — semantically equivalent but lexically different
2. Partial support — some claims supported, others not
3. Temporal/conditional — time-bounded or conditional statements
4. Aggregation — multiple sources needed, single source insufficient

Uses importance-sampling (FACT-AUDIT inspired) to focus on categories
where the verifier currently struggles most.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.llm.base import BaseLLM


@dataclass
class AttributionProbe:
    """A (claim, source) pair for attribution verification."""

    id: str
    claim: str
    source: str
    gold_label: str  # "Attributable" or "Not Attributable"
    ambiguity_type: str  # paraphrase | partial | temporal | aggregation
    difficulty: float = 0.5  # 0=easy, 1=hard
    metadata: dict = field(default_factory=dict)


# Ambiguity type definitions
AMBIGUITY_TYPES = {
    "paraphrase": {
        "name": "Paraphrase / Reformulation",
        "description": "Claim is semantically equivalent to the source but uses different wording, structure, or abstraction level.",
        "challenge": "Models often fail when surface-level lexical overlap is low but meaning is preserved.",
        "examples": [
            "Source says 'The CEO resigned effective March 1' → Claim: 'The company's top executive stepped down at the start of March'",
        ],
    },
    "partial": {
        "name": "Partial Support",
        "description": "Source supports some aspects of the claim but not all. The unsupported part may be a subtle addition, exaggeration, or specification.",
        "challenge": "Models often miss small unsupported additions (e.g., adding 'all' or a specific number not in source).",
        "examples": [
            "Source says '60% of participants improved' → Claim: 'Most participants significantly improved' (adds 'significantly')",
        ],
    },
    "temporal": {
        "name": "Temporal / Conditional",
        "description": "Claim involves time-bounded statements, conditional qualifiers, or evolving facts where the source captures only one point in time.",
        "challenge": "Models fail to check whether temporal/conditional qualifiers in the claim match the source's scope.",
        "examples": [
            "Source says 'As of 2023, the policy was under review' → Claim: 'The policy is under review' (strips temporal qualifier)",
        ],
    },
    "aggregation": {
        "name": "Aggregation / Over-generalization",
        "description": "Claim generalizes from a specific case in the source, or aggregates multiple specific statements into a broader claim.",
        "challenge": "Models incorrectly attribute generalizations that go beyond what a single source supports.",
        "examples": [
            "Source discusses one study → Claim: 'Research consistently shows...' (single study ≠ consistent research)",
        ],
    },
}


PROPOSER_SYSTEM = """\
You are an adversarial probe generator for fact attribution verification. \
Your job is to create challenging (claim, source_document) pairs where the \
attribution relationship is subtle and hard to determine.

For each pair, you generate:
1. A realistic source document (2-4 sentences, resembling a news article, \
scientific abstract, or report excerpt)
2. A claim that either IS or IS NOT attributable to the source
3. The gold label: "Attributable" or "Not Attributable"

Requirements:
- Source documents must be realistic and self-contained
- Claims must be natural (not obviously adversarial)
- The boundary between Attributable and Not Attributable should be SUBTLE
- Balance: generate roughly equal Attributable and Not Attributable examples
- Each example must have a defensible ground truth

Output: JSON array of objects with "source", "claim", "label", "reasoning".
"""


ADAPTIVE_PROMPT = """\
The current verifier has these known weaknesses:
{weakness_summary}

Its accuracy on this ambiguity type is approximately {type_accuracy:.0%}.

Generate {n} challenging (claim, source) pairs of type: {ambiguity_type}
Description: {ambiguity_desc}
Challenge: {ambiguity_challenge}

Domain: {domain}

{memory_context}

Focus on generating examples at the BOUNDARY of the verifier's ability — \
not trivially easy, not impossibly hard.

Output: JSON array of objects with "source", "claim", "label", "reasoning".
"""


CONTRASTIVE_PROMPT = """\
Given the following (claim, source) pair where the claim IS attributable to the source, \
generate {n} minimally modified versions of the claim that are NOT attributable.

Each modification should change as LITTLE as possible — ideally one word or phrase — \
so the claim looks almost identical but is no longer supported by the source.

Modification types to use:
1. **Numerical tampering**: change a number, percentage, date, or quantity
2. **Negation flip**: add or remove a negation ("confirmed" → "could not confirm")
3. **Scope inflation**: "some" → "all", "one study" → "research consistently shows"
4. **Temporal shift**: remove or alter a temporal qualifier ("as of 2023" → present tense)
5. **Entity swap**: replace a name, location, or organization with a similar one

Original claim: {claim}
Source: {source}

Output: JSON array of objects with "claim" (modified), "modification_type", "what_changed".
"""


QUALITY_FILTER_PROMPT = """\
Evaluate these candidate attribution probes for quality. A good probe must:
1. Have a DEFENSIBLE ground truth (reasonable annotators would agree)
2. Be CHALLENGING — a naive verifier would likely get it wrong
3. Be REALISTIC — resembles real-world attribution scenarios
4. Be SELF-CONTAINED — source contains enough context to judge

For each probe, output: {{"id": N, "keep": true/false, "reason": "..."}}
Return a JSON array.

Probes:
{probes}
"""


DOMAINS = [
    "science and medicine",
    "politics and policy",
    "technology and AI",
    "economics and finance",
    "legal and regulatory",
    "environment and climate",
    "history and social science",
    "sports and entertainment",
]


class AttributionProposer:
    """Generates adversarial (claim, source) pairs for attribution verification.

    Uses importance-sampling to focus probe generation on ambiguity types
    where the verifier currently struggles most (FACT-AUDIT inspired).
    """

    def __init__(
        self,
        llm: BaseLLM,
        seed: int = 42,
        memory: list[dict] | None = None,
        type_accuracies: dict[str, float] | None = None,
    ):
        self.llm = llm
        self.rng = random.Random(seed)
        self.memory: list[dict] = memory or []
        # Per-type accuracy: used for importance sampling
        self.type_accuracies: dict[str, float] = type_accuracies or {
            t: 0.5 for t in AMBIGUITY_TYPES
        }

    def generate_probes(
        self,
        ambiguity_type: str,
        n: int = 10,
        domain: str | None = None,
        adaptive: bool = False,
    ) -> list[AttributionProbe]:
        """Generate n probes for a specific ambiguity type."""
        if domain is None:
            domain = self.rng.choice(DOMAINS)

        type_info = AMBIGUITY_TYPES[ambiguity_type]
        memory_context = self._format_memory() if self.memory else ""

        if adaptive and self.memory:
            prompt = ADAPTIVE_PROMPT.format(
                weakness_summary=self._format_weaknesses(ambiguity_type),
                type_accuracy=self.type_accuracies.get(ambiguity_type, 0.5),
                n=n,
                ambiguity_type=type_info["name"],
                ambiguity_desc=type_info["description"],
                ambiguity_challenge=type_info["challenge"],
                domain=domain,
                memory_context=memory_context,
            )
        else:
            prompt = (
                f"Generate {n} challenging (claim, source) pairs.\n\n"
                f"Ambiguity type: {type_info['name']}\n"
                f"Description: {type_info['description']}\n"
                f"Challenge: {type_info['challenge']}\n"
                f"Example: {type_info['examples'][0]}\n\n"
                f"Domain: {domain}\n\n"
                f"{memory_context}\n\n"
                f"Output: JSON array of objects with 'source', 'claim', 'label', 'reasoning'."
            )

        response = self.llm.generate(prompt, system=PROPOSER_SYSTEM)
        return self._parse_probes(response.text, ambiguity_type)

    def generate_all(
        self,
        n_total: int = 40,
        adaptive: bool = False,
        filter_quality: bool = False,
    ) -> list[AttributionProbe]:
        """Generate probes across all ambiguity types.

        Uses importance-sampling: allocates more probes to types where
        the verifier accuracy is closest to 50% (maximum learning signal).
        """
        # Compute importance weights: maximize at boundary (50% accuracy)
        weights = {}
        for atype in AMBIGUITY_TYPES:
            acc = self.type_accuracies.get(atype, 0.5)
            # Boundary-optimal weight: highest at 50%, lower at extremes
            weights[atype] = 1.0 - abs(acc - 0.5) * 2.0
            # Floor: always generate at least some of each type
            weights[atype] = max(weights[atype], 0.2)

        total_weight = sum(weights.values())

        all_probes: list[AttributionProbe] = []
        for atype, w in weights.items():
            count = max(2, int(n_total * w / total_weight))
            gen_count = int(count * 1.3) if filter_quality else count
            probes = self.generate_probes(atype, n=gen_count, adaptive=adaptive)
            all_probes.extend(probes)

        if filter_quality and all_probes:
            all_probes = self._filter_quality(all_probes)

        self.rng.shuffle(all_probes)
        return all_probes

    def generate_contrastive_pairs(
        self,
        probes: list[AttributionProbe],
        n_per_probe: int = 2,
    ) -> list[AttributionProbe]:
        """Generate minimally modified negative counterparts for Attributable probes.

        For each Attributable probe, create n variants where one small change
        makes the claim Not Attributable. This teaches the verifier to attend
        to critical details (numbers, negations, scope, temporality).
        """
        positives = [p for p in probes if p.gold_label == "Attributable"]
        if not positives or not self.llm:
            return []

        contrastive = []
        for probe in positives:
            prompt = CONTRASTIVE_PROMPT.format(
                n=n_per_probe,
                claim=probe.claim,
                source=probe.source[:1500],
            )
            try:
                response = self.llm.generate(prompt, system=PROPOSER_SYSTEM)
                text = response.text.strip()
                start = text.find("[")
                end = text.rfind("]") + 1
                if start == -1 or end == 0:
                    continue
                items = json.loads(text[start:end])
                for item in items:
                    if "claim" not in item:
                        continue
                    contrastive.append(
                        AttributionProbe(
                            id=str(uuid.uuid4()),
                            claim=item["claim"],
                            source=probe.source,  # same source
                            gold_label="Not Attributable",
                            ambiguity_type="contrastive",
                            metadata={
                                "modification_type": item.get("modification_type", ""),
                                "what_changed": item.get("what_changed", ""),
                                "original_probe_id": probe.id,
                            },
                        )
                    )
            except Exception:
                continue

        return contrastive

    def _filter_quality(self, probes: list[AttributionProbe]) -> list[AttributionProbe]:
        """Filter probes by quality using LLM judge."""
        if len(probes) <= 3:
            return probes

        probes_text = "\n".join(
            f"{i+1}. Claim: {p.claim[:150]}\n"
            f"   Source: {p.source[:150]}\n"
            f"   Label: {p.gold_label}"
            for i, p in enumerate(probes)
        )

        prompt = QUALITY_FILTER_PROMPT.format(probes=probes_text)
        try:
            result = self.llm.generate(prompt)
            text = result.text.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end > 0:
                verdicts = json.loads(text[start:end])
                keep_ids = {v["id"] for v in verdicts if v.get("keep", True)}
                filtered = [p for i, p in enumerate(probes) if (i + 1) in keep_ids]
                if filtered:
                    return filtered
        except Exception:
            pass
        return probes

    def update_memory(self, failures: list[dict]) -> None:
        """Add failure patterns for next-round adaptive generation."""
        self.memory.extend(failures)

    def update_type_accuracies(self, accuracies: dict[str, float]) -> None:
        """Update per-type accuracy for importance sampling."""
        self.type_accuracies.update(accuracies)

    def _format_memory(self) -> str:
        if not self.memory:
            return ""
        lines = ["Previously observed failure patterns:"]
        for f in self.memory[-20:]:
            lines.append(
                f"- [{f.get('ambiguity_type', '?')}] {f.get('pattern', f.get('claim', '')[:100])}"
            )
        return "\n".join(lines)

    def _format_weaknesses(self, ambiguity_type: str) -> str:
        relevant = [
            m for m in self.memory
            if m.get("ambiguity_type") == ambiguity_type
        ]
        if not relevant:
            relevant = self.memory[-10:]
        return "\n".join(
            f"- {m.get('pattern', m.get('claim', '')[:100])}"
            for m in relevant[:10]
        )

    def _parse_probes(
        self, text: str, ambiguity_type: str
    ) -> list[AttributionProbe]:
        """Parse LLM output into AttributionProbe objects."""
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        try:
            items: list[dict[str, Any]] = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []

        probes = []
        for item in items:
            if "claim" not in item or "source" not in item:
                continue

            label = item.get("label", "Not Attributable")
            if label.lower() in ("attributable", "supported", "yes", "true", "entailment"):
                label = "Attributable"
            else:
                label = "Not Attributable"

            probes.append(
                AttributionProbe(
                    id=str(uuid.uuid4()),
                    claim=item["claim"],
                    source=item["source"],
                    gold_label=label,
                    ambiguity_type=ambiguity_type,
                    metadata={"reasoning": item.get("reasoning", "")},
                )
            )
        return probes
