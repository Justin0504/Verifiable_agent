"""HaluEval benchmark loader.

HaluEval (Li et al., 2023) is a large-scale hallucination evaluation
benchmark with 35K samples across three tasks: QA, dialogue, and
summarization. Each sample has a hallucinated and non-hallucinated response.

HuggingFace: pminervini/HaluEval
"""

from __future__ import annotations

from .base import BenchmarkLoader, BenchmarkSample


class HaluEvalLoader(BenchmarkLoader):
    """Load HaluEval benchmark for hallucination detection evaluation.

    Each sample has:
    - A question/context
    - A correct (non-hallucinated) response
    - A hallucinated response
    - Task type (qa / dialogue / summarization)
    """

    name = "halueval"
    description = "35K hallucination evaluation samples across QA, dialogue, and summarization"

    def __init__(self, task: str = "qa"):
        """
        Args:
            task: "qa", "dialogue", or "summarization".
        """
        self.task = task

    def load(self, split: str = "data", limit: int | None = None) -> list[BenchmarkSample]:
        datasets = self._try_import_datasets()

        config_map = {
            "qa": "qa_samples",
            "dialogue": "dialogue_samples",
            "summarization": "summarization_samples",
        }
        config = config_map.get(self.task, "qa_samples")

        ds = datasets.load_dataset("pminervini/HaluEval", config, split=split)

        samples = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break

            if self.task == "qa":
                question = row.get("question", "")
                knowledge = row.get("knowledge", "")
                correct = row.get("right_answer", row.get("answer", ""))
                hallucinated = row.get("hallucinated_answer", "")
            elif self.task == "dialogue":
                question = row.get("dialogue_history", "")
                knowledge = row.get("knowledge", "")
                correct = row.get("right_response", row.get("response", ""))
                hallucinated = row.get("hallucinated_response", "")
            else:  # summarization
                question = row.get("document", "")
                knowledge = ""
                correct = row.get("right_summary", row.get("summary", ""))
                hallucinated = row.get("hallucinated_summary", "")

            # Create two samples: one supported (correct), one contradicted (hallucinated)
            if correct:
                samples.append(BenchmarkSample(
                    id=f"halu_{self.task}_{i:05d}_correct",
                    question=question,
                    reference_answer=correct,
                    gold_label="S",
                    evidence=[knowledge] if knowledge else [],
                    metadata={
                        "task": self.task,
                        "risk_type": "missing_evidence",
                        "is_hallucinated": False,
                    },
                ))
            if hallucinated:
                samples.append(BenchmarkSample(
                    id=f"halu_{self.task}_{i:05d}_halluc",
                    question=question,
                    reference_answer=hallucinated,
                    gold_label="C",
                    evidence=[knowledge] if knowledge else [],
                    metadata={
                        "task": self.task,
                        "risk_type": "missing_evidence",
                        "is_hallucinated": True,
                    },
                ))

        return samples

    def load_manual_sample(self, limit: int = 20) -> list[BenchmarkSample]:
        """Fallback: curated HaluEval-style examples."""
        items = [
            {
                "q": "When was the Eiffel Tower built?",
                "knowledge": "The Eiffel Tower was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair in Paris.",
                "correct": "The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair.",
                "hallucinated": "The Eiffel Tower was built in 1900 for the Paris Olympic Games.",
            },
            {
                "q": "Who wrote the play Hamlet?",
                "knowledge": "Hamlet is a tragedy written by William Shakespeare, believed to have been composed between 1599 and 1601.",
                "correct": "Hamlet was written by William Shakespeare around 1600.",
                "hallucinated": "Hamlet was written by Christopher Marlowe in 1595.",
            },
            {
                "q": "What is the boiling point of water?",
                "knowledge": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
                "correct": "Water boils at 100°C (212°F) at standard atmospheric pressure.",
                "hallucinated": "Water boils at 90°C at standard atmospheric pressure.",
            },
            {
                "q": "What is the capital of Australia?",
                "knowledge": "Canberra is the capital city of Australia, chosen as a compromise between Sydney and Melbourne.",
                "correct": "The capital of Australia is Canberra.",
                "hallucinated": "The capital of Australia is Sydney.",
            },
            {
                "q": "Who painted the Sistine Chapel ceiling?",
                "knowledge": "The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512.",
                "correct": "Michelangelo painted the Sistine Chapel ceiling between 1508 and 1512.",
                "hallucinated": "Leonardo da Vinci painted the Sistine Chapel ceiling in 1505.",
            },
            {
                "q": "What year did World War I begin?",
                "knowledge": "World War I began in 1914 following the assassination of Archduke Franz Ferdinand.",
                "correct": "World War I began in 1914.",
                "hallucinated": "World War I began in 1916 after the sinking of the Lusitania.",
            },
            {
                "q": "How many planets are in our solar system?",
                "knowledge": "There are 8 planets in our solar system since Pluto was reclassified as a dwarf planet in 2006.",
                "correct": "There are 8 planets in our solar system.",
                "hallucinated": "There are 9 planets in our solar system, including Pluto.",
            },
            {
                "q": "What is the largest organ in the human body?",
                "knowledge": "The skin is the largest organ of the human body, covering about 20 square feet in adults.",
                "correct": "The skin is the largest organ in the human body.",
                "hallucinated": "The liver is the largest organ in the human body.",
            },
            {
                "q": "Who discovered gravity?",
                "knowledge": "Isaac Newton formulated the law of universal gravitation, published in his Principia Mathematica in 1687.",
                "correct": "Isaac Newton is credited with formulating the law of universal gravitation in 1687.",
                "hallucinated": "Galileo Galilei discovered gravity in 1610 by dropping objects from the Leaning Tower of Pisa.",
            },
            {
                "q": "What is the speed of light?",
                "knowledge": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                "correct": "The speed of light is approximately 299,792,458 meters per second in a vacuum.",
                "hallucinated": "The speed of light is approximately 150,000 kilometers per second.",
            },
        ]

        samples = []
        for i, item in enumerate(items[:limit]):
            samples.append(BenchmarkSample(
                id=f"halu_qa_{i:05d}_correct",
                question=item["q"],
                reference_answer=item["correct"],
                gold_label="S",
                evidence=[item["knowledge"]],
                metadata={"task": "qa", "risk_type": "missing_evidence", "is_hallucinated": False},
            ))
            samples.append(BenchmarkSample(
                id=f"halu_qa_{i:05d}_halluc",
                question=item["q"],
                reference_answer=item["hallucinated"],
                gold_label="C",
                evidence=[item["knowledge"]],
                metadata={"task": "qa", "risk_type": "missing_evidence", "is_hallucinated": True},
            ))
        return samples
