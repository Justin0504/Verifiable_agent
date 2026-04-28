"""Generate halueval.jsonl for eval_seva.py from HuggingFace HaluEval dataset.

Downloads HaluEval QA samples, converts to SEVA eval format (claim/source/gold_label),
stratified-samples 200 samples (100 Attributable + 100 Not Attributable).

Usage:
    pip install datasets
    python scripts/prepare_halueval.py
    # Output: data/attribution/halueval.jsonl (200 samples)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    random.seed(42)
    output_path = Path(__file__).resolve().parent.parent / "data" / "attribution" / "halueval.jsonl"

    print("Loading HaluEval QA from HuggingFace...")
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    print(f"Loaded {len(ds)} raw samples")

    attributable = []  # non-hallucinated answers (supported by knowledge)
    not_attributable = []  # hallucinated answers (not supported)

    for row in ds:
        question = row.get("question", "")
        knowledge = row.get("knowledge", "")
        answer = row.get("answer", "")
        is_hallucinated = row.get("hallucination", "").strip().lower() == "yes"

        if not knowledge or not question or not answer:
            continue

        claim = f"{question} {answer}"
        sample = {
            "claim": claim,
            "source": knowledge,
            "gold_label": "Not Attributable" if is_hallucinated else "Attributable",
            "benchmark": "halueval",
        }

        if is_hallucinated:
            not_attributable.append(sample)
        else:
            attributable.append(sample)

    print(f"Attributable candidates: {len(attributable)}")
    print(f"Not Attributable candidates: {len(not_attributable)}")

    # Stratified sample: 100 each
    random.shuffle(attributable)
    random.shuffle(not_attributable)
    samples = attributable[:100] + not_attributable[:100]
    random.shuffle(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {output_path}")

    # Verify
    attr_count = sum(1 for s in samples if s["gold_label"] == "Attributable")
    print(f"  Attributable: {attr_count}, Not Attributable: {len(samples) - attr_count}")


if __name__ == "__main__":
    main()
