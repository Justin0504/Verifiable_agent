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

    def load_manual_sample(self, limit: int = 30) -> list[BenchmarkSample]:
        """Fallback: curated FActScore-style examples with per-claim S/C labels."""
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
            {
                "topic": "Isaac Newton",
                "claims": [
                    "Isaac Newton was born on January 4, 1643.",
                    "He was born in Woolsthorpe, England.",
                    "Newton formulated the laws of motion and universal gravitation.",
                    "He published Principia Mathematica in 1687.",
                    "Newton served as president of the Royal Society.",
                    "He was a professor at Cambridge University.",
                    "Newton died on March 31, 1727.",
                    "Newton discovered calculus independently of Leibniz.",
                    "He invented the reflecting telescope.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "S", "S"],
            },
            {
                "topic": "Mahatma Gandhi",
                "claims": [
                    "Mahatma Gandhi was born on October 2, 1869.",
                    "He was born in Porbandar, India.",
                    "Gandhi studied law in London.",
                    "He led India's non-violent independence movement.",
                    "Gandhi was assassinated on January 30, 1948.",
                    "He was shot by Nathuram Godse.",
                    "Gandhi won the Nobel Peace Prize.",
                    "He is known as the Father of the Nation in India.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "C", "S"],
            },
            {
                "topic": "William Shakespeare",
                "claims": [
                    "William Shakespeare was born in Stratford-upon-Avon in 1564.",
                    "He wrote approximately 37 plays.",
                    "Shakespeare wrote Romeo and Juliet.",
                    "He was born on April 23, 1564.",
                    "Shakespeare died on April 23, 1616.",
                    "He married Anne Hathaway in 1582.",
                    "Shakespeare invented over 1,700 words in English.",
                    "He performed at the Globe Theatre.",
                    "Shakespeare had a university degree from Oxford.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "S", "C"],
            },
            {
                "topic": "Napoleon Bonaparte",
                "claims": [
                    "Napoleon was born on August 15, 1769 in Corsica.",
                    "He became Emperor of France in 1804.",
                    "Napoleon was defeated at the Battle of Waterloo in 1815.",
                    "He was exiled to the island of Saint Helena.",
                    "Napoleon was unusually short at about 5 feet 2 inches.",
                    "He sold the Louisiana Territory to the United States in 1803.",
                    "Napoleon died on May 5, 1821.",
                    "He introduced the Napoleonic Code.",
                ],
                "labels": ["S", "S", "S", "S", "C", "S", "S", "S"],
            },
            {
                "topic": "Charles Darwin",
                "claims": [
                    "Charles Darwin was born on February 12, 1809.",
                    "He traveled on HMS Beagle from 1831 to 1836.",
                    "Darwin visited the Galápagos Islands during the voyage.",
                    "He published On the Origin of Species in 1859.",
                    "Darwin proposed the theory of natural selection.",
                    "He studied theology at Cambridge before turning to science.",
                    "Darwin died on April 19, 1882.",
                    "He was buried in Westminster Abbey.",
                    "Darwin immediately published his theory upon returning from the Beagle voyage.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "S", "C"],
            },
            {
                "topic": "Frida Kahlo",
                "claims": [
                    "Frida Kahlo was born on July 6, 1907 in Mexico City.",
                    "She was involved in a serious bus accident at age 18.",
                    "Kahlo married fellow artist Diego Rivera.",
                    "She was known for her self-portraits.",
                    "Kahlo had a troubled relationship with Rivera and they divorced once.",
                    "She was a member of the Mexican Communist Party.",
                    "Kahlo died on July 13, 1954.",
                    "Her painting 'The Two Fridas' is her most expensive work sold at auction.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "C"],
            },
            {
                "topic": "Martin Luther King Jr.",
                "claims": [
                    "Martin Luther King Jr. was born on January 15, 1929.",
                    "He delivered the 'I Have a Dream' speech in 1963.",
                    "King was a Baptist minister.",
                    "He won the Nobel Peace Prize in 1964.",
                    "King was assassinated on April 4, 1968 in Memphis.",
                    "He led the Montgomery bus boycott in 1955.",
                    "King was the first African American to win the Nobel Peace Prize.",
                    "He received a PhD from Boston University.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "C", "S"],
            },
            {
                "topic": "Ada Lovelace",
                "claims": [
                    "Ada Lovelace was born on December 10, 1815.",
                    "She was the daughter of the poet Lord Byron.",
                    "Lovelace worked with Charles Babbage on the Analytical Engine.",
                    "She is considered the first computer programmer.",
                    "Lovelace wrote the first algorithm intended for machine processing.",
                    "She died on November 27, 1852.",
                    "Lovelace was a professor of mathematics at the University of London.",
                    "The programming language Ada was named after her.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "C", "S"],
            },
            {
                "topic": "Genghis Khan",
                "claims": [
                    "Genghis Khan was born around 1162.",
                    "His birth name was Temüjin.",
                    "He founded the Mongol Empire.",
                    "The Mongol Empire became the largest contiguous land empire in history.",
                    "Genghis Khan died in 1227.",
                    "He conquered China, Persia, and parts of Eastern Europe.",
                    "Genghis Khan could read and write in multiple languages.",
                    "He established the Yassa, a code of law for the Mongol Empire.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "C", "S"],
            },
            {
                "topic": "Alexander the Great",
                "claims": [
                    "Alexander the Great was born in 356 BC in Pella, Macedonia.",
                    "He was tutored by Aristotle.",
                    "Alexander became king at age 20 after his father Philip II was assassinated.",
                    "He conquered the Persian Empire.",
                    "Alexander died in Babylon in 323 BC at age 32.",
                    "He never lost a battle in his military career.",
                    "Alexander founded the city of Alexandria in Egypt.",
                    "He conquered all of India.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "C"],
            },
            {
                "topic": "Galileo Galilei",
                "claims": [
                    "Galileo was born on February 15, 1564 in Pisa, Italy.",
                    "He improved the telescope for astronomical observation.",
                    "Galileo discovered four moons of Jupiter.",
                    "He was tried by the Roman Inquisition for supporting heliocentrism.",
                    "Galileo was forced to recant his views in 1633.",
                    "He spent his final years under house arrest.",
                    "Galileo dropped objects from the Leaning Tower of Pisa to test gravity.",
                    "He died on January 8, 1642.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "C", "S"],
            },
            {
                "topic": "Alan Turing",
                "claims": [
                    "Alan Turing was born on June 23, 1912.",
                    "He is considered the father of theoretical computer science.",
                    "Turing worked at Bletchley Park during World War II.",
                    "He helped crack the German Enigma code.",
                    "Turing proposed the Turing test for machine intelligence.",
                    "He was prosecuted for homosexuality in 1952.",
                    "Turing died on June 7, 1954.",
                    "He received a royal pardon in 2013.",
                    "Turing built the first electronic computer.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "S", "C"],
            },
            {
                "topic": "Nelson Mandela",
                "claims": [
                    "Nelson Mandela was born on July 18, 1918.",
                    "He was imprisoned for 27 years.",
                    "Mandela was held on Robben Island for most of his imprisonment.",
                    "He became the first Black president of South Africa in 1994.",
                    "Mandela won the Nobel Peace Prize in 1993.",
                    "He shared the prize with F.W. de Klerk.",
                    "Mandela died on December 5, 2013.",
                    "He served two terms as president.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "S", "C"],
            },
            {
                "topic": "Amelia Earhart",
                "claims": [
                    "Amelia Earhart was born on July 24, 1897.",
                    "She was the first woman to fly solo across the Atlantic Ocean.",
                    "Earhart completed this flight in 1932.",
                    "She disappeared over the Pacific Ocean in 1937.",
                    "Earhart was attempting to circumnavigate the globe.",
                    "Her navigator was Fred Noonan.",
                    "Earhart's plane was found on a remote Pacific island in 1940.",
                    "She set many aviation speed and distance records.",
                ],
                "labels": ["S", "S", "S", "S", "S", "S", "C", "S"],
            },
            {
                "topic": "Stephen Hawking",
                "claims": [
                    "Stephen Hawking was born on January 8, 1942.",
                    "He was diagnosed with ALS at age 21.",
                    "Hawking proposed that black holes emit radiation (Hawking radiation).",
                    "He held the Lucasian Professorship at Cambridge.",
                    "Hawking wrote A Brief History of Time.",
                    "He won the Nobel Prize in Physics for his work on black holes.",
                    "Hawking died on March 14, 2018.",
                    "He communicated using a speech-generating device.",
                ],
                "labels": ["S", "S", "S", "S", "S", "C", "S", "S"],
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
