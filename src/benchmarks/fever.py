"""FEVER benchmark loader (full version).

FEVER (Thorne et al., 2018) is the largest fact verification benchmark
with 185K claims. Each claim is labeled SUPPORTS, REFUTES, or NOT ENOUGH INFO
and paired with Wikipedia evidence sentences.

HuggingFace: fever/fever
"""

from __future__ import annotations

from .base import BenchmarkLoader, BenchmarkSample

_LABEL_MAP = {
    0: ("SUPPORTS", "S"),
    1: ("REFUTES", "C"),
    2: ("NOT ENOUGH INFO", "N"),
    "SUPPORTS": ("SUPPORTS", "S"),
    "REFUTES": ("REFUTES", "C"),
    "NOT ENOUGH INFO": ("NOT ENOUGH INFO", "N"),
}


class FEVERLoader(BenchmarkLoader):
    """Load FEVER benchmark for fact verification.

    Each sample has:
    - A claim derived from Wikipedia
    - Label: SUPPORTS / REFUTES / NOT ENOUGH INFO
    - Evidence: Wikipedia sentences (for SUPPORTS and REFUTES)
    """

    name = "fever"
    description = "185K fact verification claims with Wikipedia evidence"

    def load(self, split: str = "validation", limit: int | None = None) -> list[BenchmarkSample]:
        datasets = self._try_import_datasets()

        # Use copenlu/fever_gold_evidence which includes evidence text
        ds = datasets.load_dataset("copenlu/fever_gold_evidence", split=split)

        samples = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break

            claim = row.get("claim", "")
            raw_label = row.get("label", "NOT ENOUGH INFO")

            label_info = _LABEL_MAP.get(raw_label, ("NOT ENOUGH INFO", "N"))
            native_label, our_label = label_info

            # Extract evidence from structured format
            evidence_texts = []
            evidence_list = row.get("evidence", [])
            if isinstance(evidence_list, list):
                for ev in evidence_list:
                    if isinstance(ev, list) and len(ev) >= 3:
                        # Format: [wiki_title, sent_id, sentence_text]
                        wiki_title = ev[0].replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")")
                        sent_text = ev[2] if len(ev) > 2 else ""
                        if sent_text:
                            evidence_texts.append(f"[{wiki_title}] {sent_text}")
                    elif isinstance(ev, str):
                        evidence_texts.append(ev)

            sample = BenchmarkSample(
                id=f"fever_{i:06d}",
                question=claim,
                reference_answer="",
                gold_label=our_label,
                evidence=evidence_texts,
                metadata={
                    "risk_type": "missing_evidence",
                    "native_label": native_label,
                    "original_id": row.get("original_id", ""),
                },
            )
            samples.append(sample)

        return samples

    def load_manual_sample(self, limit: int = 50) -> list[BenchmarkSample]:
        """Fallback: curated FEVER-style claims (extends load_fever.py manual set)."""
        claims = [
            ("The Eiffel Tower is located in Paris, France.", "S"),
            ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", "S"),
            ("Shakespeare wrote Romeo and Juliet.", "S"),
            ("The Pacific Ocean is the largest ocean on Earth.", "S"),
            ("Marie Curie won two Nobel Prizes.", "S"),
            ("The human body has 206 bones in adulthood.", "S"),
            ("Tokyo is the capital of Japan.", "S"),
            ("Venus is the hottest planet in our solar system.", "S"),
            ("DNA has a double helix structure.", "S"),
            ("The Berlin Wall fell in November 1989.", "S"),
            ("Abraham Lincoln was the 16th President of the United States.", "S"),
            ("Gold has the chemical symbol Au.", "S"),
            ("The Titanic sank on April 15, 1912.", "S"),
            ("The Earth orbits the Sun in approximately 365.25 days.", "S"),
            ("Mount Kilimanjaro is the highest mountain in Africa.", "S"),
            ("Usain Bolt set the 100m world record of 9.58 seconds in 2009.", "S"),
            ("The speed of sound in air is approximately 343 meters per second.", "S"),
            ("Nelson Mandela was released from prison in 1990.", "S"),
            ("The Wright brothers first flew on December 17, 1903.", "S"),
            ("Penicillin was discovered by Alexander Fleming in 1928.", "S"),
            ("The Amazon River is the largest river by discharge volume.", "S"),
            ("Mount Everest is 8,848.86 meters tall.", "S"),
            ("The Great Barrier Reef is the world's largest coral reef system.", "S"),
            ("India is the most populous country as of 2024.", "S"),
            ("Bitcoin was created by Satoshi Nakamoto.", "S"),
            # REFUTES
            ("The Great Wall of China is visible from the Moon.", "C"),
            ("Albert Einstein failed mathematics in school.", "C"),
            ("Humans only use 10% of their brains.", "C"),
            ("Napoleon was extremely short at about 5 feet tall.", "C"),
            ("Goldfish have a 3-second memory.", "C"),
            ("Lightning never strikes the same place twice.", "C"),
            ("Bats are blind.", "C"),
            ("Vikings wore horned helmets.", "C"),
            ("Thomas Edison invented the light bulb alone.", "C"),
            ("The Moon has no gravity.", "C"),
            ("Diamonds are formed from compressed coal.", "C"),
            ("Hair and fingernails grow after death.", "C"),
            ("Sugar causes hyperactivity in children.", "C"),
            ("George Washington had wooden teeth.", "C"),
            ("Sushi means raw fish.", "C"),
            # NOT ENOUGH INFO
            ("Aristotle's favorite food was olives.", "N"),
            ("Cleopatra spoke exactly nine languages.", "N"),
            ("Shakespeare visited Italy before writing Romeo and Juliet.", "N"),
            ("Genghis Khan's horse was white.", "N"),
            ("Isaac Newton's favorite color was blue.", "N"),
            ("Mozart composed his first symphony while sleepwalking.", "N"),
            ("Leonardo da Vinci was vegetarian for ethical reasons.", "N"),
            ("Julius Caesar's last meal included fish.", "N"),
            ("Galileo had a pet cat.", "N"),
            ("Confucius could play the guqin.", "N"),
        ]

        samples = []
        for i, (claim, label) in enumerate(claims[:limit]):
            samples.append(BenchmarkSample(
                id=f"fever_{i:06d}",
                question=claim,
                reference_answer="",
                gold_label=label,
                metadata={"risk_type": "missing_evidence", "native_label": label},
            ))
        return samples
