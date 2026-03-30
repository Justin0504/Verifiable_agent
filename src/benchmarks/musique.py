"""MuSiQue benchmark loader.

MuSiQue (Trivedi et al., 2022) is a multi-hop question answering dataset
with both answerable and unanswerable questions. Each answerable question
requires 2-4 hops of reasoning.

HuggingFace: drt/musique
"""

from __future__ import annotations

from .base import BenchmarkLoader, BenchmarkSample


class MuSiQueLoader(BenchmarkLoader):
    """Load MuSiQue benchmark for multi-hop and unanswerable question evaluation.

    Each sample has:
    - A multi-hop question (2-4 hops)
    - Answer (or "unanswerable" flag)
    - Supporting paragraphs with decomposed sub-questions
    """

    name = "musique"
    description = "Multi-hop QA with answerable and unanswerable questions"

    def load(self, split: str = "validation", limit: int | None = None) -> list[BenchmarkSample]:
        datasets = self._try_import_datasets()

        ds = datasets.load_dataset("drt/musique", split=split)

        samples = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break

            question = row.get("question", "")
            answer = row.get("answer", "")
            answerable = row.get("answerable", True)

            # Extract supporting paragraphs
            paragraphs = row.get("paragraphs", [])
            evidence = []
            if isinstance(paragraphs, list):
                for p in paragraphs:
                    if isinstance(p, dict):
                        if p.get("is_supporting", False):
                            title = p.get("title", "")
                            text = p.get("paragraph_text", p.get("text", ""))
                            evidence.append(f"[{title}] {text}" if title else text)
                    elif isinstance(p, str):
                        evidence.append(p)

            # Extract decomposed sub-questions
            decomposition = row.get("question_decomposition", [])
            sub_questions = []
            if isinstance(decomposition, list):
                for d in decomposition:
                    if isinstance(d, dict):
                        sub_questions.append(d.get("question", str(d)))
                    elif isinstance(d, str):
                        sub_questions.append(d)

            if answerable:
                gold_label = "S"
                risk_type = "multi_hop"
            else:
                gold_label = "N"
                risk_type = "unanswerable"

            sample = BenchmarkSample(
                id=f"mq_{i:05d}",
                question=question,
                reference_answer=answer if answerable else "",
                gold_label=gold_label,
                claims=sub_questions,
                evidence=evidence,
                metadata={
                    "answerable": answerable,
                    "risk_type": risk_type,
                    "num_hops": len(sub_questions),
                },
            )
            samples.append(sample)

        return samples

    def load_manual_sample(self, limit: int = 20) -> list[BenchmarkSample]:
        """Fallback: curated multi-hop and unanswerable examples."""
        items = [
            # Answerable multi-hop questions
            {
                "q": "What country is the birthplace of the inventor of the telephone?",
                "a": "Scotland",
                "answerable": True,
                "hops": ["Who invented the telephone?", "What country was Alexander Graham Bell born in?"],
                "evidence": [
                    "Alexander Graham Bell is credited with inventing the first practical telephone.",
                    "Alexander Graham Bell was born on March 3, 1847 in Edinburgh, Scotland.",
                ],
            },
            {
                "q": "What is the capital of the country where the Eiffel Tower is located?",
                "a": "Paris",
                "answerable": True,
                "hops": ["What country is the Eiffel Tower in?", "What is the capital of France?"],
                "evidence": [
                    "The Eiffel Tower is located in Paris, France.",
                    "Paris is the capital and largest city of France.",
                ],
            },
            {
                "q": "Who directed the movie that won Best Picture the year Titanic was released?",
                "a": "James Cameron",
                "answerable": True,
                "hops": ["What year was Titanic released?", "What movie won Best Picture in 1998?", "Who directed Titanic?"],
                "evidence": [
                    "Titanic was released in 1997.",
                    "Titanic won Best Picture at the 70th Academy Awards in 1998.",
                    "Titanic was directed by James Cameron.",
                ],
            },
            {
                "q": "What language is spoken in the country where the composer of The Four Seasons was born?",
                "a": "Italian",
                "answerable": True,
                "hops": ["Who composed The Four Seasons?", "What country was Vivaldi born in?", "What language is spoken in Italy?"],
                "evidence": [
                    "The Four Seasons was composed by Antonio Vivaldi.",
                    "Antonio Vivaldi was born in Venice, Republic of Venice (modern-day Italy).",
                    "The official language of Italy is Italian.",
                ],
            },
            {
                "q": "What is the population of the city where the first modern Olympics were held?",
                "a": "Approximately 3.1 million (Athens metropolitan area)",
                "answerable": True,
                "hops": ["Where were the first modern Olympics held?", "What is the population of Athens?"],
                "evidence": [
                    "The first modern Olympic Games were held in Athens, Greece in 1896.",
                    "Athens has a population of approximately 660,000 in the city proper and 3.1 million in the metropolitan area.",
                ],
            },
            # Unanswerable questions
            {
                "q": "What is the favorite color of the person who invented calculus?",
                "a": "",
                "answerable": False,
                "hops": ["Who invented calculus?", "What is Newton's favorite color?"],
                "evidence": ["There is no reliable historical record of Isaac Newton's favorite color."],
            },
            {
                "q": "How many dreams did Shakespeare have while writing Hamlet?",
                "a": "",
                "answerable": False,
                "hops": ["When did Shakespeare write Hamlet?", "How many dreams did he have?"],
                "evidence": ["There is no historical record of Shakespeare's dreams."],
            },
            {
                "q": "What song was playing when the Berlin Wall fell?",
                "a": "",
                "answerable": False,
                "hops": ["When did the Berlin Wall fall?", "What specific song was playing?"],
                "evidence": ["While there were celebrations, there is no single definitive song associated with the exact moment."],
            },
            {
                "q": "What was Cleopatra's phone number?",
                "a": "",
                "answerable": False,
                "hops": [],
                "evidence": ["Telephones did not exist in ancient Egypt."],
            },
            {
                "q": "What did Genghis Khan think about democracy?",
                "a": "",
                "answerable": False,
                "hops": [],
                "evidence": ["There are no reliable records of Genghis Khan's views on democracy as a concept."],
            },
        ]

        samples = []
        for i, item in enumerate(items[:limit]):
            if item["answerable"]:
                gold_label = "S"
                risk_type = "multi_hop"
            else:
                gold_label = "N"
                risk_type = "unanswerable"

            samples.append(BenchmarkSample(
                id=f"mq_{i:05d}",
                question=item["q"],
                reference_answer=item["a"],
                gold_label=gold_label,
                claims=item["hops"],
                evidence=item["evidence"],
                metadata={
                    "answerable": item["answerable"],
                    "risk_type": risk_type,
                    "num_hops": len(item["hops"]),
                },
            ))
        return samples
