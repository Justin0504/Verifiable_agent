"""Load and convert FEVER dataset claims into knowledge base documents.

FEVER (Fact Extraction and VERification) is a benchmark dataset for
fact verification. This script downloads a sample and converts it into
our JSONL KB format.

Usage:
    python scripts/load_fever.py [--output knowledge_base/documents/fever_facts.jsonl] [--limit 200]

Requires: datasets (pip install datasets)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_from_huggingface(limit: int = 200) -> list[dict]:
    """Load FEVER claims from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package required. Install with: pip install datasets")
        sys.exit(1)

    print("Downloading FEVER dataset from HuggingFace...")
    dataset = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)

    # Filter for SUPPORTS and REFUTES labels (skip NOT ENOUGH INFO for KB)
    supported = [ex for ex in dataset if ex["label"] == 0][:limit // 2]  # SUPPORTS
    refuted = [ex for ex in dataset if ex["label"] == 1][:limit // 2]   # REFUTES

    records = []
    for i, ex in enumerate(supported + refuted):
        label_name = "SUPPORTS" if ex["label"] == 0 else "REFUTES"
        claim = ex.get("claim", "")
        evidence_info = ex.get("evidence_sentence", "") or ex.get("evidence_wiki_url", "")

        records.append({
            "id": f"fever_{i+1:04d}",
            "title": f"FEVER Claim: {claim[:80]}",
            "content": claim,
            "source": "FEVER Dataset",
            "metadata": {
                "label": label_name,
                "evidence": evidence_info,
                "original_id": ex.get("id", i),
            },
        })

    return records


def load_manual_fever_sample() -> list[dict]:
    """Fallback: manually curated FEVER-style claims with known verdicts.

    Used when the HuggingFace datasets library is not available.
    Each entry has a factual claim and ground truth for calibration.
    """
    claims = [
        # SUPPORTS — true claims
        {"claim": "The Eiffel Tower is located in Paris, France.", "label": "SUPPORTS"},
        {"claim": "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "label": "SUPPORTS"},
        {"claim": "The Great Wall of China is visible from low Earth orbit under certain conditions.", "label": "SUPPORTS"},
        {"claim": "Shakespeare wrote Romeo and Juliet.", "label": "SUPPORTS"},
        {"claim": "The Pacific Ocean is the largest ocean on Earth.", "label": "SUPPORTS"},
        {"claim": "Usain Bolt set the 100m world record of 9.58 seconds in 2009.", "label": "SUPPORTS"},
        {"claim": "Marie Curie won two Nobel Prizes.", "label": "SUPPORTS"},
        {"claim": "The human body has 206 bones in adulthood.", "label": "SUPPORTS"},
        {"claim": "Tokyo is the capital of Japan.", "label": "SUPPORTS"},
        {"claim": "The Amazon Rainforest produces approximately 20% of the world's oxygen.", "label": "SUPPORTS"},
        {"claim": "Venus is the hottest planet in our solar system.", "label": "SUPPORTS"},
        {"claim": "The Mona Lisa was painted by Leonardo da Vinci.", "label": "SUPPORTS"},
        {"claim": "DNA has a double helix structure.", "label": "SUPPORTS"},
        {"claim": "The Berlin Wall fell in November 1989.", "label": "SUPPORTS"},
        {"claim": "Photosynthesis converts carbon dioxide and water into glucose and oxygen.", "label": "SUPPORTS"},
        {"claim": "The speed of sound in air at sea level is approximately 343 meters per second.", "label": "SUPPORTS"},
        {"claim": "Abraham Lincoln was the 16th President of the United States.", "label": "SUPPORTS"},
        {"claim": "Gold has the chemical symbol Au.", "label": "SUPPORTS"},
        {"claim": "The Titanic sank on April 15, 1912.", "label": "SUPPORTS"},
        {"claim": "The Earth orbits the Sun in approximately 365.25 days.", "label": "SUPPORTS"},
        {"claim": "Nelson Mandela was released from prison in 1990.", "label": "SUPPORTS"},
        {"claim": "The human brain contains approximately 86 billion neurons.", "label": "SUPPORTS"},
        {"claim": "Mount Kilimanjaro is the highest mountain in Africa.", "label": "SUPPORTS"},
        {"claim": "The Wright brothers made the first sustained powered flight on December 17, 1903.", "label": "SUPPORTS"},
        {"claim": "Penicillin was discovered by Alexander Fleming in 1928.", "label": "SUPPORTS"},

        # REFUTES — false claims for contradiction detection
        {"claim": "The Great Wall of China is visible from the Moon with the naked eye.", "label": "REFUTES"},
        {"claim": "Albert Einstein failed mathematics in school.", "label": "REFUTES"},
        {"claim": "Humans only use 10% of their brains.", "label": "REFUTES"},
        {"claim": "Napoleon Bonaparte was extremely short at about 5 feet tall.", "label": "REFUTES"},
        {"claim": "The tongue has specific taste zones for different flavors.", "label": "REFUTES"},
        {"claim": "Goldfish have a 3-second memory.", "label": "REFUTES"},
        {"claim": "Lightning never strikes the same place twice.", "label": "REFUTES"},
        {"claim": "The dark side of the Moon is always dark.", "label": "REFUTES"},
        {"claim": "Thomas Edison invented the light bulb.", "label": "REFUTES"},
        {"claim": "Christopher Columbus discovered America.", "label": "REFUTES"},
        {"claim": "Bats are blind.", "label": "REFUTES"},
        {"claim": "Bulls are enraged by the color red.", "label": "REFUTES"},
        {"claim": "Sushi means raw fish.", "label": "REFUTES"},
        {"claim": "Vikings wore horned helmets.", "label": "REFUTES"},
        {"claim": "The Sahara is the largest desert in the world.", "label": "REFUTES"},
        {"claim": "Mount Everest is the tallest mountain on Earth measured from base to peak.", "label": "REFUTES"},
        {"claim": "George Washington had wooden teeth.", "label": "REFUTES"},
        {"claim": "Chameleons change color to blend in with their surroundings.", "label": "REFUTES"},
        {"claim": "The Moon has no gravity.", "label": "REFUTES"},
        {"claim": "Diamonds are formed from compressed coal.", "label": "REFUTES"},
        {"claim": "Swimming right after eating causes cramps and drowning.", "label": "REFUTES"},
        {"claim": "Hair and fingernails continue to grow after death.", "label": "REFUTES"},
        {"claim": "Sugar causes hyperactivity in children.", "label": "REFUTES"},
        {"claim": "Cracking your knuckles causes arthritis.", "label": "REFUTES"},
        {"claim": "The Great Fire of London in 1666 killed thousands of people.", "label": "REFUTES"},
    ]

    records = []
    for i, item in enumerate(claims):
        records.append({
            "id": f"fever_{i+1:04d}",
            "title": f"FEVER: {item['claim'][:80]}",
            "content": item["claim"],
            "source": "FEVER-style Manual Dataset",
            "metadata": {"label": item["label"]},
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Load FEVER dataset into knowledge base")
    parser.add_argument("--output", default="knowledge_base/documents/fever_facts.jsonl")
    parser.add_argument("--limit", type=int, default=200, help="Max claims to load")
    parser.add_argument("--manual", action="store_true", help="Use manual curated sample (no download)")
    args = parser.parse_args()

    if args.manual:
        records = load_manual_fever_sample()
    else:
        try:
            records = load_from_huggingface(limit=args.limit)
        except Exception as e:
            print(f"HuggingFace download failed ({e}), falling back to manual sample...")
            records = load_manual_fever_sample()

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Stats
    labels = {}
    for r in records:
        label = r.get("metadata", {}).get("label", "UNKNOWN")
        labels[label] = labels.get(label, 0) + 1

    print(f"Saved {len(records)} claims to {output_path}")
    print(f"Label distribution: {labels}")


if __name__ == "__main__":
    main()
