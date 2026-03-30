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


# ── Shared entity extraction ──────────────────────────────────────────

_NON_ENTITIES = {
    # Determiners / pronouns / prepositions
    "The", "This", "That", "These", "Those", "It", "He", "She", "They",
    "In", "On", "At", "By", "For", "From", "With", "About", "After",
    "Before", "During", "Between", "Since", "Until", "Into", "Through",
    # Conjunctions / adverbs
    "However", "Moreover", "Furthermore", "Therefore", "Although", "Also",
    "Because", "While", "Where", "When", "Which", "What", "Who", "How",
    # Months / days
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    # Common nouns that get capitalized at sentence start
    "According", "Research", "Studies", "Evidence", "Data", "Scientists",
    # Non-entity proper-looking words (units, elements, generic nouns)
    "Celsius", "Fahrenheit", "Kelvin", "Water", "Gold", "Iron", "Carbon",
    "Ocean", "River", "Lake", "Mountain", "Desert", "Island", "Reef",
    "Tower", "Bridge", "Wall", "Gate", "Palace", "Castle", "Temple",
    "North", "South", "East", "West", "Earth", "World", "Universe",
    "Yes", "No", "True", "False", "Both", "Many", "Most", "Some",
    "Nobel", "Prize", "Award", "University", "Institute", "College",
    # Country/continent names that are too generic for person-like strategies
    "Japan", "China", "India", "France", "Germany", "Italy", "Spain",
    "Russia", "Brazil", "Mexico", "Canada", "Australia", "Korea",
    "Africa", "Asia", "Europe", "America", "Antarctica",
}


def extract_entities(text: str) -> list[str]:
    """Extract proper noun entities (persons, organizations, places).

    Prefers multi-word names, filters out common false positives.
    """
    # Multi-word capitalized sequences (strongest signal)
    multi = re.findall(r'([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+)', text)
    # Single proper nouns (>3 chars, preceded by space — not sentence start)
    singles = re.findall(r'(?<=\s)([A-Z][a-z]{3,})', text)
    inner_singles: list[str] = []

    all_candidates = multi + singles + inner_singles
    # Filter: remove words where ANY part is in the stop list
    filtered = []
    seen = set()
    for e in all_candidates:
        words = e.split()
        if all(w not in _NON_ENTITIES for w in words) and e not in seen:
            filtered.append(e)
            seen.add(e)
    return filtered


def _is_person_entity(entity: str, claim: str) -> bool:
    """Heuristic: does this entity refer to a person?"""
    person_signals = [
        r'\b' + re.escape(entity) + r"'s\b",
        r'\b' + re.escape(entity) + r'\s+(?:was|is|had|has|won|wrote|discovered|invented|created|founded|born|died)',
        r'(?:by|of)\s+' + re.escape(entity),
    ]
    return any(re.search(p, claim, re.IGNORECASE) for p in person_signals)


# ── Base class ─────────────────────────────────────────────────────────

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
        ...


# ── Strategy 1: Numerical Perturbation ────────────────────────────────

class NumericalPerturbStrategy(AdversarialStrategy):
    """Perturb numbers in claims to create subtle contradictions.

    Handles: years, large numbers, small numbers, ordinals, percentages,
    and spelled-out numbers.
    """

    name = "numerical_perturb"
    description = "Subtly alter numbers to create contradictions"
    output_label = "C"

    # Spelled-out numbers to digit mapping
    SPELLED = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000,
    }
    DIGIT_TO_SPELLED = {v: k for k, v in SPELLED.items() if v <= 20}

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        results = []

        # Pattern 1: Digit-based numbers
        digit_numbers = list(re.finditer(r'\b(\d{1,10}(?:,\d{3})*(?:\.\d+)?)\b', claim))
        for match in digit_numbers[:2]:
            perturbed_claim = self._perturb_digit(claim, match, rng)
            if perturbed_claim:
                results.append(perturbed_claim)

        # Pattern 2: Spelled-out numbers ("two Nobel Prizes")
        for word, val in self.SPELLED.items():
            pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
            m = pattern.search(claim)
            if m and val <= 20 and val > 0:
                delta = rng.choice([-2, -1, 1, 2])
                new_val = max(1, val + delta)
                if new_val in self.DIGIT_TO_SPELLED:
                    new_word = self.DIGIT_TO_SPELLED[new_val]
                    # Preserve original case
                    if m.group(0)[0].isupper():
                        new_word = new_word.capitalize()
                    perturbed = claim[:m.start()] + new_word + claim[m.end():]
                    results.append(AdversarialSample(
                        id="", claim=perturbed, gold_label="C",
                        strategy=self.name, difficulty="hard",
                        original_claim=claim, evidence=evidence,
                        explanation=f"Changed spelled number '{m.group(0)}' → '{new_word}'",
                    ))
                break  # One spelled number per claim

        # Pattern 3: Ordinals ("1st", "2nd", "16th")
        ordinals = list(re.finditer(r'\b(\d+)(?:st|nd|rd|th)\b', claim))
        for m in ordinals[:1]:
            val = int(m.group(1))
            delta = rng.choice([-2, -1, 1, 2])
            new_val = max(1, val + delta)
            suffix = "th"
            if new_val % 10 == 1 and new_val != 11:
                suffix = "st"
            elif new_val % 10 == 2 and new_val != 12:
                suffix = "nd"
            elif new_val % 10 == 3 and new_val != 13:
                suffix = "rd"
            new_str = f"{new_val}{suffix}"
            perturbed = claim[:m.start()] + new_str + claim[m.end():]
            results.append(AdversarialSample(
                id="", claim=perturbed, gold_label="C",
                strategy=self.name, difficulty="medium",
                original_claim=claim, evidence=evidence,
                explanation=f"Changed ordinal '{m.group(0)}' → '{new_str}'",
            ))

        return results

    def _perturb_digit(self, claim, match, rng) -> AdversarialSample | None:
        original_num = match.group(1)
        num_val = float(original_num.replace(",", ""))

        if num_val > 1900 and num_val < 2100:
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            new_val = int(num_val + delta)
            new_str = str(new_val)
            explanation = f"Changed year {original_num} → {new_str} (off by {abs(delta)})"
        elif num_val >= 100:
            pct = rng.uniform(0.05, 0.20)
            direction = rng.choice([-1, 1])
            new_val = num_val * (1 + direction * pct)
            if num_val == int(num_val):
                new_str = f"{int(new_val):,}" if "," in original_num else str(int(new_val))
            else:
                new_str = f"{new_val:.1f}"
            explanation = f"Changed {original_num} → {new_str} ({direction * pct:+.0%})"
        elif num_val > 0:
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
            new_val = max(1, num_val + delta)
            new_str = str(int(new_val)) if num_val == int(num_val) else f"{new_val:.1f}"
            explanation = f"Changed {original_num} → {new_str} (off by {abs(delta)})"
        else:
            return None

        perturbed = claim[:match.start(1)] + new_str + claim[match.end(1):]
        return AdversarialSample(
            id="", claim=perturbed, gold_label="C",
            strategy=self.name, difficulty="medium",
            original_claim=claim, evidence=[],
            explanation=explanation,
        )


# ── Strategy 2: Multi-Hop Graft ───────────────────────────────────────

class MultiHopGraftStrategy(AdversarialStrategy):
    """Combine claims into multi-hop reasoning chains.

    Works with or without years — can use entities, locations, or
    causal relationships as the linking dimension.
    """

    name = "multi_hop_graft"
    description = "Combine claims into multi-hop reasoning chains"
    output_label = "S"

    # Known entity → related fact pairs for grafting
    ENTITY_FACTS = {
        "Einstein": "who developed the theory of relativity",
        "Newton": "who formulated the laws of motion",
        "Darwin": "who proposed the theory of evolution",
        "Shakespeare": "who wrote plays at the Globe Theatre",
        "Curie": "who researched radioactivity",
        "Tesla": "who pioneered alternating current",
        "Edison": "who held over 1,000 patents",
        "Galileo": "who improved the telescope",
        "Napoleon": "who crowned himself Emperor of France",
        "Columbus": "who sailed across the Atlantic in 1492",
        "Fleming": "who worked at St Mary's Hospital",
        "Turing": "who broke the Enigma code",
    }

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        results = []

        # Method 1: Temporal multi-hop (if year present)
        year_match = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', claim)
        if year_match:
            year = int(year_match.group(1))
            offset = rng.randint(1, 8)
            # Ensure the math works out correctly (verifiably true)
            base_year = year - offset
            templates = [
                f"{offset} years after {base_year}, {claim[0].lower() + claim[1:]}",
                f"In the {year // 10 * 10}s, {claim[0].lower() + claim[1:]}",
            ]
            chosen = rng.choice(templates)
            results.append(AdversarialSample(
                id="", claim=chosen, gold_label="S",
                strategy=self.name, difficulty="hard",
                original_claim=claim, evidence=evidence,
                explanation=f"Added temporal multi-hop: {offset} years after {base_year} = {year}",
            ))

        # Method 2: Entity-linked multi-hop (no year needed)
        entities = extract_entities(claim)
        for entity in entities:
            for name_key, fact in self.ENTITY_FACTS.items():
                if name_key.lower() in entity.lower():
                    connector = rng.choice([
                        f"{entity}, {fact},",
                        f"The scientist {fact}, {entity},",
                    ])
                    # Rebuild: "{entity}, {related fact}, {rest of claim}"
                    rest = claim[claim.find(entity) + len(entity):].strip().lstrip(",").strip()
                    if rest:
                        grafted = f"{connector} {rest}"
                    else:
                        grafted = f"{connector} is well documented."
                    results.append(AdversarialSample(
                        id="", claim=grafted, gold_label="S",
                        strategy=self.name, difficulty="hard",
                        original_claim=claim, evidence=evidence,
                        explanation=f"Grafted related fact about {entity} to create multi-hop verification",
                    ))
                    break

        # Method 3: Negation-based multi-hop (create a "not X but Y" structure)
        if not results:
            wrong_claims = {
                "Paris": "not in London but in Paris",
                "Tokyo": "not in Beijing but in Tokyo",
                "1928": "not in the 1930s but in 1928",
                "first": "not the second but the first",
            }
            for keyword, replacement in wrong_claims.items():
                if keyword in claim:
                    # Insert "not X but Y" structure
                    grafted = claim.replace(keyword, replacement, 1)
                    results.append(AdversarialSample(
                        id="", claim=grafted, gold_label="S",
                        strategy=self.name, difficulty="medium",
                        original_claim=claim, evidence=evidence,
                        explanation=f"Added contrastive multi-hop: '{replacement}'",
                    ))
                    break

        return results[:1]  # Return at most 1


# ── Strategy 3: Presupposition ─────────────────────────────────────────

class PresuppositionStrategy(AdversarialStrategy):
    """Inject false presuppositions — only for person/organization entities."""

    name = "presupposition"
    description = "Inject false presuppositions into claims"
    output_label = "C"

    PERSON_TEMPLATES = [
        ("born in {wrong_place}", "false birthplace"),
        ("known for {wrong_achievement}", "false achievement"),
        ("the {wrong_nationality} scientist", "false nationality"),
    ]

    ORG_TEMPLATES = [
        ("founded in {wrong_year}", "false founding year"),
        ("headquartered in {wrong_place}", "false location"),
        ("established during the {wrong_era} era", "false time period"),
    ]

    WRONG_PLACES = ["London", "New York", "Tokyo", "Berlin", "Moscow", "Beijing", "Sydney",
                    "Rome", "Cairo", "Rio de Janeiro", "Toronto", "Seoul"]
    WRONG_ACHIEVEMENTS = [
        "inventing the telescope", "discovering oxygen", "writing the first novel",
        "building the first computer", "founding the United Nations",
        "circumnavigating the globe", "splitting the atom",
    ]
    WRONG_NATIONALITIES = ["French", "German", "Japanese", "Russian", "Italian",
                          "Brazilian", "Canadian", "Australian", "Swedish", "Chinese"]
    WRONG_ERAS = ["Renaissance", "Victorian", "Medieval", "Baroque", "Enlightenment", "Industrial"]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        entities = extract_entities(claim)
        if not entities:
            return []

        # Prefer person entities
        person_entities = [e for e in entities if _is_person_entity(e, claim)]
        if person_entities:
            entity = rng.choice(person_entities)
            template, desc = rng.choice(self.PERSON_TEMPLATES)
        elif len(entities[0].split()) >= 2:
            # Multi-word entity likely an org or place
            entity = entities[0]
            template, desc = rng.choice(self.ORG_TEMPLATES)
        else:
            return []  # Can't safely inject presupposition

        filled = template.format(
            wrong_place=rng.choice(self.WRONG_PLACES),
            wrong_achievement=rng.choice(self.WRONG_ACHIEVEMENTS),
            wrong_nationality=rng.choice(self.WRONG_NATIONALITIES),
            wrong_year=str(rng.randint(1800, 2020)),
            wrong_era=rng.choice(self.WRONG_ERAS),
        )

        # Clean injection patterns — keep original claim casing intact
        pattern = rng.choice([
            f"Although {entity} was {filled}, {claim}",
            f"Despite being {filled}, {claim}",
            f"Contrary to popular belief that {entity} was {filled}, {claim}",
        ])

        # Ensure proper ending
        if not pattern.endswith("."):
            pattern += "."

        return [AdversarialSample(
            id="", claim=pattern, gold_label="C",
            strategy=self.name, difficulty="hard",
            original_claim=claim, evidence=evidence,
            explanation=f"Injected false presupposition ({desc}) for '{entity}'",
        )]

    @staticmethod
    def _extract_predicate(entity: str, claim: str) -> str:
        """Extract the predicate (what comes after the entity in the claim)."""
        idx = claim.find(entity)
        if idx >= 0:
            rest = claim[idx + len(entity):].strip().lstrip(",").strip()
            if rest:
                return rest
        return claim[0].lower() + claim[1:]


# ── Strategy 4: Unanswerable Wrap ──────────────────────────────────────

class UnanswerableWrapStrategy(AdversarialStrategy):
    """Transform claims into questions about unrecorded/subjective information.

    Only wraps around person entities to avoid nonsensical questions.
    """

    name = "unanswerable_wrap"
    description = "Wrap claims in unanswerable question format"
    output_label = "N"

    PERSON_TEMPLATES = [
        "What was {entity}'s personal opinion about {topic}?",
        "How did {entity} feel about {topic} later in life?",
        "What would have happened if {entity} had not {action}?",
        "Did {entity} ever express regret about {action}?",
        "What was {entity}'s motivation for {action}?",
        "Who influenced {entity} most in their early career?",
        "What was {entity}'s daily routine while working on {topic}?",
        "How did {entity}'s family react to {action}?",
    ]

    GENERIC_TEMPLATES = [
        "What would the world look like today if {event} had never happened?",
        "How did contemporaries privately feel about {event}?",
        "What were the undocumented consequences of {event}?",
    ]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        entities = extract_entities(claim)
        person_entities = [e for e in entities if _is_person_entity(e, claim)]

        # Extract meaningful action/topic from the claim
        verbs = re.findall(
            r'\b(?:discovered|invented|wrote|built|founded|won|created|developed|'
            r'published|proposed|proved|demonstrated|established|introduced|designed)\s+'
            r'(?:the\s+)?[\w\s]{3,30}',
            claim.lower(),
        )
        action = verbs[0] if verbs else None

        # Extract topic from the claim (after "about", "of", or key nouns)
        topic_match = re.search(r'(?:theory of|study of|discovery of|invention of)\s+[\w\s]+', claim.lower())
        topic = topic_match.group(0) if topic_match else None

        if person_entities:
            entity = rng.choice(person_entities)
            template = rng.choice(self.PERSON_TEMPLATES)
            # Clean up action for template: "won two nobel prizes" → "winning two Nobel Prizes"
            action_clean = action or "their most notable achievement"
            if action_clean.startswith(("discovered", "invented", "wrote", "built",
                                        "founded", "won", "created", "developed")):
                # Convert past tense to gerund for natural phrasing
                verb = action_clean.split()[0]
                gerund_map = {
                    "discovered": "discovering", "invented": "inventing",
                    "wrote": "writing", "built": "building", "founded": "founding",
                    "won": "winning", "created": "creating", "developed": "developing",
                    "published": "publishing", "proposed": "proposing",
                }
                if verb in gerund_map:
                    action_clean = gerund_map[verb] + action_clean[len(verb):]

            question = template.format(
                entity=entity,
                topic=topic or action_clean or "their most famous work",
                action=action_clean,
            )
        elif action:
            # Use generic event-based template
            template = rng.choice(self.GENERIC_TEMPLATES)
            question = template.format(event=action)
        else:
            return []  # Can't generate a meaningful unanswerable question

        return [AdversarialSample(
            id="", claim=question, gold_label="N",
            strategy=self.name, difficulty="hard",
            original_claim=claim, evidence=evidence,
            explanation=f"Converted factual claim into unanswerable question about subjective/undocumented aspects",
        )]


# ── Strategy 5: Paraphrase ────────────────────────────────────────────

class ParaphraseStrategy(AdversarialStrategy):
    """Rephrase claims with safe, grammatically correct transformations."""

    name = "paraphrase"
    description = "Rephrase claims to test verifier robustness"
    output_label = ""  # same as original

    HEDGES = [
        "It is widely accepted that {claim}",
        "Historical records confirm that {claim}",
        "It has been established that {claim}",
        "Scholars generally agree that {claim}",
        "{claim}",  # No change (filtered out later)
    ]

    TEMPORAL_SWAPS = [
        (r'\bin (\d{4})\b', r'during the year \1'),
        (r'\bin (\d{4})\b', r'as of \1'),
        (r'\bapproximately (\d+)', r'roughly \1'),
        (r'\babout (\d+)', r'approximately \1'),
        (r'\baround (\d+)', r'close to \1'),
    ]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label not in ("S", "C"):
            return []

        modified = claim.rstrip(".")

        # Safe temporal swap (pick one at random)
        swaps = list(self.TEMPORAL_SWAPS)
        rng.shuffle(swaps)
        for pattern, replacement in swaps:
            if re.search(pattern, modified):
                modified = re.sub(pattern, replacement, modified, count=1)
                break

        # Apply hedge wrapper — preserve proper noun capitalization
        hedge = rng.choice(self.HEDGES)
        # Only lowercase the first char if it's NOT a proper noun (multi-word caps or known entity)
        first_word = modified.split()[0] if modified else ""
        is_proper = (
            len(modified.split()) > 1
            and modified.split()[1][0:1].isupper()  # "Marie Curie" → keep
        ) or first_word not in ("The", "A", "An", "It", "In", "On", "At")

        if is_proper:
            claim_text = modified  # Keep original case
        else:
            claim_text = modified[0].lower() + modified[1:]

        modified = hedge.format(claim=claim_text)

        # Capitalize and punctuate
        modified = modified[0].upper() + modified[1:]
        if not modified.endswith("."):
            modified += "."

        if modified.rstrip(".") == claim.rstrip("."):
            return []

        return [AdversarialSample(
            id="", claim=modified, gold_label=label,
            strategy=self.name, difficulty="easy",
            original_claim=claim, evidence=evidence,
            explanation="Surface-form paraphrase; gold label unchanged",
        )]


# ── Strategy 6: Entity Confusion ──────────────────────────────────────

class EntityConfusionStrategy(AdversarialStrategy):
    """Swap attributes between similar entities to create subtle errors."""

    name = "entity_confusion"
    description = "Swap attributes between similar entities"
    output_label = "C"

    CONFUSION_PAIRS = [
        # Scientists
        {"entities": ["Albert Einstein", "Einstein"], "swaps": {"1879": "1885", "Ulm": "Copenhagen", "photoelectric effect": "atomic model", "1921": "1922"}},
        {"entities": ["Marie Curie", "Curie"], "swaps": {"radium": "DNA structure", "1903": "1952", "polonium": "francium", "Warsaw": "Paris"}},
        {"entities": ["Neil Armstrong"], "swaps": {"first person": "second person", "first man": "second man", "first to walk": "second to walk"}},
        {"entities": ["Edison", "Thomas Edison"], "swaps": {"light bulb": "alternating current", "Menlo Park": "Wardenclyffe"}},
        {"entities": ["Newton", "Isaac Newton"], "swaps": {"1665": "1675", "1687": "1696", "Principia": "Monadology"}},
        {"entities": ["Darwin", "Charles Darwin"], "swaps": {"1859": "1858", "On the Origin of Species": "The Malay Archipelago", "Galápagos": "Borneo"}},
        {"entities": ["Alexander Fleming", "Fleming"], "swaps": {"1928": "1935", "penicillin": "sulfonamide", "St Mary": "Cambridge"}},
        {"entities": ["Galileo"], "swaps": {"1609": "1608", "Jupiter": "Saturn", "Italy": "Netherlands"}},
        {"entities": ["Watson and Crick", "Watson"], "swaps": {"1953": "1951", "double helix": "triple helix"}},
        {"entities": ["Turing", "Alan Turing"], "swaps": {"Enigma": "Lorenz", "1936": "1938", "Manchester": "Oxford"}},
        # Geography
        {"entities": ["Amazon River", "Amazon"], "swaps": {"6,400": "6,650", "South America": "Africa", "discharge volume": "length", "largest": "longest"}},
        {"entities": ["Mount Everest", "Everest"], "swaps": {"8,848": "8,611", "Nepal": "Pakistan", "highest": "second highest", "Hillary": "Messner"}},
        {"entities": ["Pacific Ocean", "Pacific"], "swaps": {"largest": "second largest", "Pacific": "Atlantic"}},
        {"entities": ["Sahara"], "swaps": {"largest hot desert": "largest desert", "9.2 million": "14 million", "Africa": "Asia"}},
        {"entities": ["Nile"], "swaps": {"6,650": "6,400", "longest": "largest", "Africa": "South America"}},
        # Historical events
        {"entities": ["Wright brothers", "Wright"], "swaps": {"1903": "1906", "Kitty Hawk": "Paris", "North Carolina": "France"}},
        {"entities": ["Apollo 11"], "swaps": {"1969": "1968", "Armstrong": "Aldrin", "Eagle": "Columbia"}},
        {"entities": ["Berlin Wall"], "swaps": {"1989": "1991", "November": "October", "1961": "1963"}},
        {"entities": ["Titanic"], "swaps": {"1912": "1913", "April 15": "April 14", "iceberg": "reef"}},
        # Tech
        {"entities": ["Tim Berners-Lee", "Berners-Lee"], "swaps": {"1989": "1991", "CERN": "MIT", "World Wide Web": "Internet"}},
        {"entities": ["Apple", "Steve Jobs"], "swaps": {"1976": "1977", "Steve Wozniak": "Bill Gates", "Apple II": "Macintosh"}},
        {"entities": ["Bitcoin"], "swaps": {"2008": "2009", "2009": "2008", "Satoshi Nakamoto": "Hal Finney"}},
    ]

    def generate(self, claim, label, evidence, metadata, rng) -> list[AdversarialSample]:
        if label != "S":
            return []

        claim_lower = claim.lower()

        for pair in self.CONFUSION_PAIRS:
            matched_entity = None
            for entity in pair["entities"]:
                if entity.lower() in claim_lower:
                    matched_entity = entity
                    break
            if not matched_entity:
                continue

            modified = claim
            applied_swaps = []
            for old, new in pair["swaps"].items():
                if old.lower() in modified.lower():
                    pattern = re.compile(re.escape(old), re.IGNORECASE)
                    modified = pattern.sub(new, modified, count=1)
                    applied_swaps.append(f"{old} → {new}")

            if applied_swaps and modified != claim:
                return [AdversarialSample(
                    id="", claim=modified, gold_label="C",
                    strategy=self.name, difficulty="hard",
                    original_claim=claim, evidence=evidence,
                    explanation=f"Swapped attributes of {matched_entity}: {'; '.join(applied_swaps)}",
                )]

        return []


# Registry of all strategies
ALL_STRATEGIES: list[type[AdversarialStrategy]] = [
    NumericalPerturbStrategy,
    MultiHopGraftStrategy,
    PresuppositionStrategy,
    UnanswerableWrapStrategy,
    ParaphraseStrategy,
    EntityConfusionStrategy,
]
