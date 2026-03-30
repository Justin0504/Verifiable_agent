"""Adversarial perturbation strategies for generating harder benchmark data.

Each strategy takes an original claim + metadata and produces a harder
variant with a known gold label. Strategies are composable.

Six core strategies:
1. NumericalPerturb — change numbers slightly to create subtle contradictions
2. MultiHopGraft — combine two claims into a multi-hop reasoning chain
3. Presupposition — inject a false presupposition into the claim
4. UnanswerableWrap — transform into an unanswerable-looking question
5. Paraphrase — rephrase while preserving semantics (tests robustness)
6. EntityConfusion — swap attributes between similar entities
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AdversarialSample:
    """A single adversarial sample with provenance."""

    id: str
    claim: str
    gold_label: str  # "S", "C", "N"
    strategy: str  # which strategy generated this
    difficulty: str  # "easy", "medium", "hard"
    original_id: str = ""  # source sample id
    original_claim: str = ""
    evidence: list[str] = field(default_factory=list)
    explanation: str = ""  # why this is hard / what changed
    metadata: dict = field(default_factory=dict)


_STOP_ENTITIES = {
    "The", "This", "That", "These", "Those", "It", "He", "She", "They",
    "In", "On", "At", "By", "For", "From", "With", "About", "After",
    "Before", "During", "Between", "Since", "Until", "Into", "Through",
    "However", "Moreover", "Furthermore", "Therefore", "Although",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "According", "Research", "Studies", "Evidence", "Data",
}


def extract_entities(text: str) -> list[str]:
    """Extract proper noun entities, filtering out common false positives."""
    candidates = re.findall(r'([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+)', text)
    singles = re.findall(r'\b([A-Z][a-z]{3,})\b', text)
    all_candidates = candidates + singles
    return [e for e in all_candidates if e.split()[0] not in _STOP_ENTITIES]


class AdversarialStrategy(ABC):
    """Base class for adversarial perturbation strategies."""

    name: str = ""
    description: str = ""
    output_label: str = ""  # expected gold label of generated sample

    @abstractmethod
    def generate(
        self,
        claim: str,
        label: str,
        evidence: list[str],
        metadata: dict,
        rng: random.Random,
    ) -> list[AdversarialSample]:
        """Generate adversarial variants of a claim.

        Args:
            claim: Original claim text.
            label: Original gold label ("S", "C", "N").
            evidence: Supporting evidence passages.
            metadata: Additional context (topic, category, etc.).
            rng: Seeded random generator for reproducibility.

        Returns:
            List of AdversarialSample objects (may produce 0 if not applicable).
        """
        ...


class NumericalPerturbStrategy(AdversarialStrategy):
    """Perturb numbers in claims to create subtle contradictions.

    Changes years, quantities, percentages, and measurements by small amounts
    that are wrong but plausible. The gold label flips from S → C.
    """

    name = "numerical_perturb"
    description = "Subtly alter numbers to create contradictions"
    output_label = "C"

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        numbers = list(re.finditer(r'\b(\d{1,4}(?:,\d{3})*(?:\.\d+)?)\b', claim))
        if not numbers:
            return []

        results = []
        for match in numbers[:2]:  # Perturb up to 2 numbers per claim
            original_num = match.group(1)
            num_val = float(original_num.replace(",", ""))

            # Choose perturbation based on magnitude
            if num_val > 1900 and num_val < 2100:
                # Likely a year — shift by 1-3
                delta = rng.choice([-3, -2, -1, 1, 2, 3])
                new_val = int(num_val + delta)
                new_str = str(new_val)
                explanation = f"Changed year {original_num} → {new_str} (off by {abs(delta)})"
            elif num_val > 100:
                # Large number — percentage shift
                pct = rng.uniform(0.05, 0.20)
                direction = rng.choice([-1, 1])
                new_val = num_val * (1 + direction * pct)
                if num_val == int(num_val):
                    new_str = f"{int(new_val):,}" if "," in original_num else str(int(new_val))
                else:
                    new_str = f"{new_val:.1f}"
                explanation = f"Changed {original_num} → {new_str} ({direction * pct:+.0%})"
            elif num_val < 100:
                # Small number — shift by 1-5
                delta = rng.choice([-5, -3, -2, -1, 1, 2, 3, 5])
                new_val = max(0, num_val + delta)
                if num_val == int(num_val):
                    new_str = str(int(new_val))
                else:
                    new_str = f"{new_val:.1f}"
                explanation = f"Changed {original_num} → {new_str} (off by {abs(delta)})"
            else:
                continue

            perturbed = claim[:match.start(1)] + new_str + claim[match.end(1):]

            results.append(AdversarialSample(
                id="",  # assigned by generator
                claim=perturbed,
                gold_label="C",
                strategy=self.name,
                difficulty="medium",
                original_claim=claim,
                evidence=evidence,
                explanation=explanation,
            ))

        return results


class MultiHopGraftStrategy(AdversarialStrategy):
    """Combine two claims into a compound claim requiring multi-hop reasoning.

    Takes two related claims and creates a new claim that requires
    understanding both to verify. Tests compositional reasoning.
    """

    name = "multi_hop_graft"
    description = "Combine claims into multi-hop reasoning chains"
    output_label = "S"  # or "C" depending on construction

    TEMPLATES = [
        "The {entity_a} of {subject_a}, which {fact_b}, {claim_tail_a}.",
        "{subject_a}, who {fact_b}, {claim_tail_a}.",
        "In the same year that {fact_b}, {claim_a_rephrased}.",
        "Unlike {contrast_entity}, {claim_a_rephrased}.",
    ]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        # This strategy needs a partner claim — store for batch processing
        # For single-claim mode, create self-referential multi-hop
        if label != "S":
            return []

        # Extract year if present
        year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', claim)
        if not year_match:
            return []

        year = year_match.group(1)

        # Create a multi-hop by adding a temporal qualifier
        templates = [
            f"{rng.randint(1, 10)} years after {int(year) - rng.randint(1, 10)}, {claim.lower()}",
            f"In the decade that began with {int(year) - int(year) % 10}, {claim.lower()}",
            f"During the same century as the founding of {rng.choice(['MIT', 'Stanford', 'Oxford', 'Harvard'])}, {claim.lower()}",
        ]

        chosen = rng.choice(templates)

        return [AdversarialSample(
            id="",
            claim=chosen,
            gold_label="S",
            strategy=self.name,
            difficulty="hard",
            original_claim=claim,
            evidence=evidence,
            explanation=f"Wrapped claim in multi-hop temporal reasoning around year {year}",
        )]


class PresuppositionStrategy(AdversarialStrategy):
    """Inject false presuppositions into claims.

    Adds a false premise that makes the claim look plausible but is actually
    built on incorrect assumptions. Tests whether verifiers catch the
    hidden false premise.
    """

    name = "presupposition"
    description = "Inject false presuppositions into claims"
    output_label = "C"

    FALSE_PRESUPPOSITIONS = [
        ("was born in {wrong_place}", "false birthplace"),
        ("who was known for {wrong_achievement}", "false achievement"),
        ("which occurred in {wrong_year}", "false year"),
        ("the {wrong_nationality} {entity}", "false nationality"),
        ("after winning the {wrong_award}", "false award"),
        ("during the {wrong_era} era", "false time period"),
    ]

    WRONG_PLACES = ["London", "New York", "Tokyo", "Berlin", "Moscow", "Beijing", "Sydney", "Rome"]
    WRONG_ACHIEVEMENTS = [
        "inventing the telescope", "discovering America", "writing the first novel",
        "building the first computer", "founding the United Nations", "discovering oxygen",
    ]
    WRONG_NATIONALITIES = ["French", "German", "Japanese", "Russian", "Italian", "Brazilian", "Canadian"]
    WRONG_AWARDS = [
        "Nobel Peace Prize", "Fields Medal", "Pulitzer Prize",
        "Academy Award", "Grammy Award", "Turing Award",
    ]
    WRONG_ERAS = ["Renaissance", "Victorian", "Medieval", "Baroque", "Enlightenment", "Industrial"]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        entities = extract_entities(claim)
        if not entities:
            return []

        entity = rng.choice(entities)
        template_idx = rng.randint(0, len(self.FALSE_PRESUPPOSITIONS) - 1)
        template, desc = self.FALSE_PRESUPPOSITIONS[template_idx]

        # Fill template
        filled = template.format(
            wrong_place=rng.choice(self.WRONG_PLACES),
            wrong_achievement=rng.choice(self.WRONG_ACHIEVEMENTS),
            wrong_year=str(rng.randint(1800, 2020)),
            wrong_nationality=rng.choice(self.WRONG_NATIONALITIES),
            entity=entity,
            wrong_award=rng.choice(self.WRONG_AWARDS),
            wrong_era=rng.choice(self.WRONG_ERAS),
        )

        # Inject presupposition
        injection_patterns = [
            f"{entity}, {filled}, {claim[claim.find(entity) + len(entity):].strip().lstrip(',')}",
            f"Since {entity} {filled}, {claim.lower()}",
            f"Given that {entity} {filled}, it follows that {claim.lower()}",
        ]

        chosen = rng.choice(injection_patterns)

        return [AdversarialSample(
            id="",
            claim=chosen,
            gold_label="C",
            strategy=self.name,
            difficulty="hard",
            original_claim=claim,
            evidence=evidence,
            explanation=f"Injected false presupposition: {desc} for entity '{entity}'",
        )]


class UnanswerableWrapStrategy(AdversarialStrategy):
    """Transform claims into questions that appear answerable but lack evidence.

    Wraps a factual claim in a question format that asks about something
    unprovable or unrecorded, making the expected answer "not enough info".
    """

    name = "unanswerable_wrap"
    description = "Wrap claims in unanswerable question format"
    output_label = "N"

    WRAP_TEMPLATES = [
        "What was {entity}'s personal opinion about {topic}?",
        "How did {entity} feel when {event}?",
        "What would have happened if {entity} had not {action}?",
        "Did {entity} regret {action}?",
        "What was {entity} thinking about on the day of {event}?",
        "How many times did {entity} attempt {action} before succeeding?",
        "What inspired {entity} to pursue {topic}?",
        "Who was {entity}'s closest friend during {event}?",
    ]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        entities = extract_entities(claim)
        if not entities:
            return []

        entity = entities[0]

        # Extract verb phrases as potential actions
        verbs = re.findall(r'\b(?:discovered|invented|wrote|built|founded|won|created|developed|published)\s+\w+(?:\s+\w+)?', claim.lower())
        action = verbs[0] if verbs else "this achievement"

        # Extract topics
        topics = re.findall(r'(?:the\s+)?(?:theory of|study of|field of|concept of)\s+\w+', claim.lower())
        topic = topics[0] if topics else "this work"

        # Extract events
        year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', claim)
        event = f"the events of {year_match.group(1)}" if year_match else action

        template = rng.choice(self.WRAP_TEMPLATES)
        question = template.format(
            entity=entity,
            topic=topic,
            event=event,
            action=action,
        )

        return [AdversarialSample(
            id="",
            claim=question,
            gold_label="N",
            strategy=self.name,
            difficulty="hard",
            original_claim=claim,
            evidence=evidence,
            explanation=f"Transformed factual claim about {entity} into unanswerable question about subjective/unrecorded information",
        )]


class ParaphraseStrategy(AdversarialStrategy):
    """Rephrase claims while preserving semantics.

    Tests verifier robustness — a correct verifier should give the same
    label regardless of surface form. Uses rule-based transformations.
    """

    name = "paraphrase"
    description = "Rephrase claims to test verifier robustness"
    output_label = ""  # same as original

    TRANSFORMS = [
        # Active ↔ Passive patterns
        (r'(\w+)\s+(?:discovered|invented)\s+(.*)', r'\2 was discovered/invented by \1'),
        (r'(\w+)\s+was awarded\s+(.*)', r'\2 was given to \1'),
        (r'(\w+)\s+won\s+(.*)', r'\2 was won by \1'),
        # Numeric rephrasing
        (r'approximately (\d+)', r'around \1'),
        (r'about (\d+)', r'roughly \1'),
        (r'(\d+) years', r'a span of \1 years'),
        # Temporal rephrasing
        (r'in (\d{4})', r'during the year \1'),
        (r'on (\w+ \d+, \d{4})', r'as of \1'),
    ]

    HEDGES = [
        "It is known that {}",
        "According to historical records, {}",
        "{}",  # no change
        "Research indicates that {}",
        "It has been established that {}",
    ]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label not in ("S", "C"):
            return []

        modified = claim

        # Apply a random rule-based transform
        for pattern, replacement in self.TRANSFORMS:
            if re.search(pattern, modified, re.IGNORECASE):
                modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
                break

        # Apply a hedge
        hedge = rng.choice(self.HEDGES)
        if modified[0].isupper():
            modified_lower = modified[0].lower() + modified[1:]
        else:
            modified_lower = modified
        modified = hedge.format(modified_lower)

        # Ensure first letter is capitalized
        modified = modified[0].upper() + modified[1:]

        # Ensure ends with period
        if not modified.endswith("."):
            modified += "."

        if modified == claim:
            return []

        return [AdversarialSample(
            id="",
            claim=modified,
            gold_label=label,  # same as original
            strategy=self.name,
            difficulty="easy",
            original_claim=claim,
            evidence=evidence,
            explanation="Surface-form paraphrase; gold label should be unchanged",
        )]


class EntityConfusionStrategy(AdversarialStrategy):
    """Swap attributes between similar entities to create subtle errors.

    Takes a claim about entity A and replaces it with a similar entity B's
    attributes, creating a plausible-looking but incorrect claim.
    """

    name = "entity_confusion"
    description = "Swap attributes between similar entities"
    output_label = "C"

    # Pairs of commonly confused entities with their distinguishing attributes
    CONFUSION_PAIRS = [
        {
            "entities": ["Albert Einstein", "Niels Bohr"],
            "swaps": {"1879": "1885", "Ulm": "Copenhagen", "photoelectric effect": "atomic model"},
        },
        {
            "entities": ["Marie Curie", "Rosalind Franklin"],
            "swaps": {"radium": "DNA structure", "1903": "1952", "Nobel Prize": "X-ray crystallography"},
        },
        {
            "entities": ["Neil Armstrong", "Buzz Aldrin"],
            "swaps": {"first person": "second person", "first man": "second man"},
        },
        {
            "entities": ["Edison", "Tesla"],
            "swaps": {"light bulb": "alternating current", "direct current": "alternating current", "Menlo Park": "New York"},
        },
        {
            "entities": ["Newton", "Leibniz"],
            "swaps": {"1665": "1675", "fluxions": "infinitesimals", "English": "German"},
        },
        {
            "entities": ["Darwin", "Wallace"],
            "swaps": {"On the Origin of Species": "the Malay Archipelago", "1859": "1858"},
        },
        {
            "entities": ["Wright brothers", "Santos-Dumont"],
            "swaps": {"1903": "1906", "Kitty Hawk": "Paris", "Flyer": "14-bis"},
        },
        {
            "entities": ["Amazon River", "Nile River"],
            "swaps": {"6,400": "6,650", "South America": "Africa", "discharge volume": "length"},
        },
        {
            "entities": ["Mount Everest", "K2"],
            "swaps": {"8,848": "8,611", "Nepal": "Pakistan", "highest": "second highest"},
        },
        {
            "entities": ["Pacific Ocean", "Atlantic Ocean"],
            "swaps": {"largest": "second largest", "Pacific": "Atlantic"},
        },
    ]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        claim_lower = claim.lower()
        results = []

        for pair in self.CONFUSION_PAIRS:
            # Check if any entity from this pair is mentioned
            matched_entity = None
            for entity in pair["entities"]:
                if entity.lower() in claim_lower:
                    matched_entity = entity
                    break

            if not matched_entity:
                continue

            # Apply swaps
            modified = claim
            applied_swaps = []
            for old, new in pair["swaps"].items():
                if old.lower() in modified.lower():
                    # Case-preserving replacement
                    pattern = re.compile(re.escape(old), re.IGNORECASE)
                    modified = pattern.sub(new, modified, count=1)
                    applied_swaps.append(f"{old} → {new}")

            if applied_swaps and modified != claim:
                other_entity = [e for e in pair["entities"] if e != matched_entity]
                results.append(AdversarialSample(
                    id="",
                    claim=modified,
                    gold_label="C",
                    strategy=self.name,
                    difficulty="hard",
                    original_claim=claim,
                    evidence=evidence,
                    explanation=f"Swapped attributes of {matched_entity} with {other_entity[0] if other_entity else 'similar entity'}: {'; '.join(applied_swaps)}",
                ))
                break  # One confusion per claim is enough

        return results


# Registry of all strategies
ALL_STRATEGIES: list[type[AdversarialStrategy]] = [
    NumericalPerturbStrategy,
    MultiHopGraftStrategy,
    PresuppositionStrategy,
    UnanswerableWrapStrategy,
    ParaphraseStrategy,
    EntityConfusionStrategy,
]
