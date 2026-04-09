"""ReasoningBank: structured rule distillation from verification trajectories.

Inspired by:
- ReasoningBank (2509.25140): distill rules from success/failure trajectories
- A-MEM (2502.12110): Zettelkasten-style memory with autonomous linking

Each verification outcome (success or failure) is distilled into a structured
Rule with title, description, content, and links to related rules. The bank
grows over epochs and is injected into the Verifier's system prompt.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from src.llm.base import BaseLLM


@dataclass
class Rule:
    """A structured verification rule distilled from experience."""

    id: str
    title: str  # One-line strategy name
    description: str  # One-sentence summary
    content: str  # Detailed reasoning steps / insights
    source_type: str  # "success" | "failure" | "calibration"
    tags: list[str] = field(default_factory=list)
    linked_rule_ids: list[str] = field(default_factory=list)
    usage_count: int = 0
    success_count: int = 0
    epoch_created: int = 0
    created_at: str = ""

    @property
    def effectiveness(self) -> float:
        if self.usage_count == 0:
            return 0.5  # Prior: neutral
        return self.success_count / self.usage_count


DISTILL_SYSTEM = """\
You are a meta-learning agent that distills reusable verification rules from \
experience. Given a verification trajectory (claim, evidence, prediction, outcome), \
extract a structured rule that captures the generalizable lesson.

For SUCCESSFUL trajectories: extract the strategy that led to correct verification.
For FAILED trajectories: extract the pitfall, what went wrong, and a counterfactual \
lesson (what should have been done instead).

Output a JSON object with:
- "title": concise rule name (under 10 words)
- "description": one-sentence summary of the rule
- "content": detailed reasoning pattern or decision heuristic (2-4 sentences)
- "tags": list of keywords for retrieval (claim types, error patterns, domains)
"""

DISTILL_PROMPT = """\
Trajectory:
- Claim: {claim}
- Evidence snippet: {evidence}
- Predicted label: {predicted}
- Outcome: {outcome}
- Confidence: {confidence}
- Context: {context}

Distill this into a reusable verification rule.
"""

LINK_SYSTEM = """\
You are analyzing a memory bank of verification rules. Given a NEW rule and a set \
of EXISTING rules, identify meaningful connections. Two rules are linked if they:
1. Address the same type of claim or error pattern
2. Offer complementary or contrasting strategies
3. One generalizes or refines the other

Also, suggest if any existing rule should have its description or tags updated \
in light of the new rule.

Output JSON:
- "links": list of existing rule IDs that connect to the new rule
- "updates": list of {{"rule_id": ..., "new_tags": [...], "refined_description": ...}} \
  for existing rules that should be updated (empty list if none)
"""

LINK_PROMPT = """\
NEW RULE:
  Title: {new_title}
  Description: {new_desc}
  Tags: {new_tags}

EXISTING RULES:
{existing_rules}

Identify connections and suggest updates.
"""


class ReasoningBank:
    """Persistent bank of structured verification rules with autonomous linking."""

    MAX_RULES = 100  # Keep bounded to avoid prompt bloat
    MAX_INJECT = 15  # Max rules injected into system prompt per query

    def __init__(self, path: str = "memory/reasoning_bank.json", llm: BaseLLM | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.rules: list[Rule] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self.rules = [Rule(**r) for r in data]

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump([asdict(r) for r in self.rules], f, indent=2)

    def distill_from_trajectory(
        self,
        claim: str,
        evidence: str,
        predicted_label: str,
        gold_label: str | None,
        confidence: float,
        context: str = "",
        epoch: int = 0,
    ) -> Rule | None:
        """Distill a structured rule from a single verification trajectory.

        If gold_label is available, uses it for success/failure classification.
        Otherwise, uses confidence-based heuristics (high-confidence = success).
        """
        if not self.llm:
            return None

        # Classify outcome
        if gold_label:
            is_success = predicted_label == gold_label
            outcome = f"CORRECT (predicted={predicted_label}, gold={gold_label})" if is_success \
                else f"WRONG (predicted={predicted_label}, gold={gold_label})"
        else:
            is_success = confidence >= 0.8
            outcome = f"HIGH CONFIDENCE ({confidence:.2f})" if is_success \
                else f"LOW CONFIDENCE ({confidence:.2f}) — likely unreliable"

        prompt = DISTILL_PROMPT.format(
            claim=claim[:200],
            evidence=evidence[:300],
            predicted=predicted_label,
            outcome=outcome,
            confidence=confidence,
            context=context[:200],
        )

        result = self.llm.generate(prompt, system=DISTILL_SYSTEM)
        rule = self._parse_rule(result.text, is_success, epoch)

        if rule:
            self._add_rule(rule)
        return rule

    def distill_batch(
        self,
        trajectories: list[dict],
        epoch: int = 0,
    ) -> list[Rule]:
        """Distill rules from a batch of trajectories. Deduplicates internally."""
        rules = []
        for t in trajectories:
            rule = self.distill_from_trajectory(
                claim=t["claim"],
                evidence=t.get("evidence", ""),
                predicted_label=t["predicted_label"],
                gold_label=t.get("gold_label"),
                confidence=t.get("confidence", 0.5),
                context=t.get("context", ""),
                epoch=epoch,
            )
            if rule:
                rules.append(rule)
        return rules

    def retrieve_relevant(self, claim: str, top_k: int | None = None) -> list[Rule]:
        """Retrieve the most relevant rules for a given claim.

        Uses word-level overlap between claim and rule tags/title/description,
        plus effectiveness weighting. Filters out proven low-effectiveness rules.
        """
        k = top_k or self.MAX_INJECT
        if not self.rules:
            return []

        claim_words = set(self._tokenize(claim))
        if not claim_words:
            return []

        scored: list[tuple[float, Rule]] = []
        for rule in self.rules:
            # Skip rules with proven low effectiveness (used 3+ times, <30% success)
            if rule.usage_count >= 3 and rule.effectiveness < 0.3:
                continue

            # Build rule word set from tags + title + description
            rule_words = set()
            for tag in rule.tags:
                rule_words.update(self._tokenize(tag))
            rule_words.update(self._tokenize(rule.title))
            rule_words.update(self._tokenize(rule.description))

            # Word-level overlap score
            overlap = claim_words & rule_words
            if not overlap:
                continue

            # Jaccard-like score normalized by claim length
            word_score = len(overlap) / max(len(claim_words), 1)

            # Effectiveness bonus (rules that have worked before are preferred)
            eff_bonus = rule.effectiveness * 0.3 if rule.usage_count > 0 else 0.15
            # Recency bonus (newer rules slightly preferred)
            recency = 0.05 * min(rule.epoch_created, 10)
            score = word_score + eff_bonus + recency
            scored.append((score, rule))

        scored.sort(key=lambda x: -x[0])
        return [rule for _, rule in scored[:k]]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization, lowercase, skip stopwords and short words."""
        import re
        _STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "that", "this", "these", "those", "and", "but", "or", "nor",
            "not", "so", "very", "just", "about", "also", "than", "too", "only",
            "same", "other", "each", "every", "all", "both", "few", "more", "most",
            "such", "no", "any", "some", "what", "which", "who", "whom", "how",
            "when", "where", "why", "if", "it", "its", "he", "she", "they", "them",
        }
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if len(w) > 2 and w not in _STOPWORDS]

    def format_for_prompt(self, rules: list[Rule]) -> str:
        """Format retrieved rules for injection into the Verifier system prompt."""
        if not rules:
            return ""

        lines = ["\n\nVERIFICATION RULES (learned from past experience):"]
        for i, rule in enumerate(rules, 1):
            source_tag = "+" if rule.source_type == "success" else "-"
            eff = f" [eff={rule.effectiveness:.0%}]" if rule.usage_count > 0 else ""
            lines.append(
                f"\n[{source_tag}] Rule {i}: {rule.title}{eff}\n"
                f"  {rule.description}\n"
                f"  Detail: {rule.content}"
            )
        return "\n".join(lines)

    def record_usage(self, rule_id: str, was_helpful: bool) -> None:
        """Record whether a rule was helpful in a verification decision."""
        for rule in self.rules:
            if rule.id == rule_id:
                rule.usage_count += 1
                if was_helpful:
                    rule.success_count += 1
                break

    def _add_rule(self, rule: Rule) -> None:
        """Add a new rule and trigger A-MEM style autonomous linking."""
        self.rules.append(rule)

        # Link to existing rules (if LLM available and enough rules exist)
        if self.llm and len(self.rules) > 1:
            self._auto_link(rule)

        # Evict low-performing rules (quality + capacity)
        self._evict()

        self.save()

    def _auto_link(self, new_rule: Rule) -> None:
        """A-MEM inspired: use LLM to find connections and update existing rules."""
        existing = [r for r in self.rules if r.id != new_rule.id]
        if not existing:
            return

        # Only send top candidates (by tag overlap) to avoid huge prompts
        candidates = existing[-20:]  # Most recent 20
        existing_text = "\n".join(
            f"  [{r.id[:8]}] {r.title} | Tags: {', '.join(r.tags)} | {r.description}"
            for r in candidates
        )

        prompt = LINK_PROMPT.format(
            new_title=new_rule.title,
            new_desc=new_rule.description,
            new_tags=", ".join(new_rule.tags),
            existing_rules=existing_text,
        )

        try:
            result = self.llm.generate(prompt, system=LINK_SYSTEM)
            data = self._extract_json(result.text)
            if not data:
                return

            # Add links
            linked_ids = data.get("links", [])
            for rid in linked_ids:
                rid_str = str(rid)
                for r in candidates:
                    if r.id.startswith(rid_str):
                        if new_rule.id not in r.linked_rule_ids:
                            r.linked_rule_ids.append(new_rule.id)
                        if r.id not in new_rule.linked_rule_ids:
                            new_rule.linked_rule_ids.append(r.id)

            # Apply updates to existing rules (A-MEM: new memory refines old)
            for update in data.get("updates", []):
                rid = str(update.get("rule_id", ""))
                for r in candidates:
                    if r.id.startswith(rid):
                        if update.get("new_tags"):
                            r.tags = list(set(r.tags + update["new_tags"]))
                        if update.get("refined_description"):
                            r.description = update["refined_description"]
        except Exception:
            pass  # Linking is best-effort, don't crash on failure

    def _evict(self) -> None:
        """Remove lowest-performing rules. Runs both:
        1. Quality eviction: remove rules with proven low effectiveness (any time)
        2. Capacity eviction: remove lowest-scoring rules when over MAX_RULES
        """
        # Quality eviction: remove rules used 3+ times with <20% effectiveness
        self.rules = [
            r for r in self.rules
            if not (r.usage_count >= 3 and r.effectiveness < 0.2)
        ]

        if len(self.rules) <= self.MAX_RULES:
            return
        # Capacity eviction: score = effectiveness * 0.7 + recency * 0.3
        max_epoch = max(r.epoch_created for r in self.rules) or 1
        scored = [
            (r.effectiveness * 0.7 + (r.epoch_created / max_epoch) * 0.3, r)
            for r in self.rules
        ]
        scored.sort(key=lambda x: x[0])
        # Remove bottom 20%
        n_remove = len(self.rules) - self.MAX_RULES + 10
        remove_ids = {s[1].id for s in scored[:n_remove]}
        self.rules = [r for r in self.rules if r.id not in remove_ids]

    def _parse_rule(self, text: str, is_success: bool, epoch: int) -> Rule | None:
        data = self._extract_json(text)
        if not data or "title" not in data:
            return None
        return Rule(
            id=str(uuid.uuid4()),
            title=data["title"],
            description=data.get("description", ""),
            content=data.get("content", ""),
            source_type="success" if is_success else "failure",
            tags=data.get("tags", []),
            epoch_created=epoch,
            created_at=datetime.now().isoformat(),
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    def stats(self) -> dict:
        success_rules = [r for r in self.rules if r.source_type == "success"]
        failure_rules = [r for r in self.rules if r.source_type == "failure"]
        linked = sum(1 for r in self.rules if r.linked_rule_ids)
        return {
            "total_rules": len(self.rules),
            "success_rules": len(success_rules),
            "failure_rules": len(failure_rules),
            "linked_rules": linked,
            "avg_effectiveness": (
                sum(r.effectiveness for r in self.rules) / len(self.rules)
                if self.rules else 0
            ),
        }
