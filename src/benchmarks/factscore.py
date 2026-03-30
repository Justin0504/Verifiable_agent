"""FActScore benchmark loader.

FActScore (Min et al., 2023) evaluates factual precision of LLM-generated
biographies by decomposing them into atomic facts and checking each against
Wikipedia. This aligns directly with our claim decomposition pipeline.

HuggingFace: factscore/factscore (or original repo data)
"""

from __future__ import annotations

from .base import BenchmarkLoader, BenchmarkSample


class FActScoreLoader(BenchmarkLoader):
    """Load FActScore benchmark for atomic fact verification evaluation.

    Each sample has:
    - A person entity (topic)
    - LLM-generated biography text
    - Atomic facts decomposed from the biography
    - Per-fact labels (supported / not supported by Wikipedia)
    """

    name = "factscore"
    description = "Atomic fact verification for LLM-generated biographies"

    def load(self, split: str = "validation", limit: int | None = None) -> list[BenchmarkSample]:
        datasets = self._try_import_datasets()

        try:
            ds = datasets.load_dataset("factscore/factscore", split=split)
        except Exception:
            # Fallback: try the labeled data
            ds = datasets.load_dataset("factscore/factscore-data", split=split)

        samples = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break

            topic = row.get("topic", row.get("entity", ""))
            generation = row.get("generation", row.get("output", ""))
            atomic_facts = row.get("atomic_facts", [])
            labels = row.get("labels", [])  # per-fact S/NS labels

            # Convert per-fact labels to our format
            claims = []
            gold_labels = []
            if isinstance(atomic_facts, list):
                for j, fact in enumerate(atomic_facts):
                    if isinstance(fact, str):
                        claims.append(fact)
                        if j < len(labels):
                            # FActScore: True = supported, False = not supported
                            lbl = "S" if labels[j] else "C"
                            gold_labels.append(lbl)
                    elif isinstance(fact, dict):
                        claims.append(fact.get("text", str(fact)))
                        if fact.get("label") is not None:
                            gold_labels.append("S" if fact["label"] else "C")

            sample = BenchmarkSample(
                id=f"fs_{i:04d}",
                question=f"Tell me about {topic}",
                reference_answer=generation,
                gold_label="mixed",  # per-claim labels, not single label
                claims=claims,
                metadata={
                    "topic": topic,
                    "risk_type": "missing_evidence",
                    "per_claim_labels": gold_labels,
                    "num_supported": gold_labels.count("S"),
                    "num_contradicted": gold_labels.count("C"),
                },
            )
            samples.append(sample)

        return samples

    def load_manual_sample(self, limit: int = 20) -> list[BenchmarkSample]:
        """Fallback: curated FActScore-style examples."""
        items = [
            {
                "topic": "Albert Einstein",
                "claims": [
                    "Albert Einstein was born on March 14, 1879.",
                    "He was born in Ulm, Germany.",
                    "Einstein developed the theory of general relativity.",
                    "He won the Nobel Prize in Physics in 1921.",
                    "The Nobel Prize was for his explanation of the photoelectric effect.",
                    "Einstein became a US citizen in 1940.",
                    "He worked at the Institute for Advanced Study in Princeton.",
                    "Einstein died on April 18, 1955.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "S"],
            },
            {
                "topic": "Marie Curie",
                "claims": [
                    "Marie Curie was born in Warsaw, Poland.",
                    "She was the first woman to win a Nobel Prize.",
                    "Curie discovered radium and polonium.",
                    "She won Nobel Prizes in both Physics and Chemistry.",
                    "Her first Nobel Prize was in Physics in 1903.",
                    "Her second Nobel Prize was in Chemistry in 1911.",
                    "She died on July 4, 1934.",
                    "Curie was the first person to win Nobel Prizes in two sciences.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "S"],
            },
            {
                "topic": "Leonardo da Vinci",
                "claims": [
                    "Leonardo da Vinci was born on April 15, 1452.",
                    "He was born in Vinci, Republic of Florence.",
                    "Da Vinci painted the Mona Lisa.",
                    "He painted The Last Supper.",
                    "Da Vinci designed flying machines centuries before the airplane.",
                    "He wrote his notes in mirror script.",
                    "Da Vinci was primarily left-handed.",
                    "He died on May 2, 1519 in Amboise, France.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "S"],
            },
            {
                "topic": "Nikola Tesla",
                "claims": [
                    "Nikola Tesla was born in Serbia.",
                    "Tesla invented the alternating current electrical system.",
                    "He worked for Thomas Edison before starting his own company.",
                    "Tesla held over 300 patents.",
                    "He proposed a wireless energy transmission system.",
                    "Tesla was born on July 10, 1856.",
                    "He died on January 7, 1943 in New York City.",
                    "Tesla invented the radio before Marconi.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "C"],
            },
            {
                "topic": "Cleopatra",
                "claims": [
                    "Cleopatra was the last active ruler of the Ptolemaic Kingdom of Egypt.",
                    "She was of Greek descent, not Egyptian.",
                    "Cleopatra had relationships with Julius Caesar and Mark Antony.",
                    "She died in 30 BC.",
                    "Cleopatra died from a snake bite.",
                    "She could speak nine languages.",
                    "Cleopatra was known for her extraordinary beauty.",
                    "She was 18 when she became queen.",
                ],
                "labels": ["S", "S", "S", "S", "C", "S", "C", "S"],
            },
        ]

        samples = []
        for i, item in enumerate(items[:limit]):
            per_claim_labels = item["labels"]
            samples.append(BenchmarkSample(
                id=f"fs_{i:04d}",
                question=f"Tell me about {item['topic']}",
                reference_answer="",
                gold_label="mixed",
                claims=item["claims"],
                metadata={
                    "topic": item["topic"],
                    "risk_type": "missing_evidence",
                    "per_claim_labels": per_claim_labels,
                    "num_supported": per_claim_labels.count("S"),
                    "num_contradicted": per_claim_labels.count("C"),
                },
            ))
        return samples
