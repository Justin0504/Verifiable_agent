"""Generate SFT training data using GPT-4o as teacher.

Creates verification trajectories with reasoning traces for Qwen2.5-3B fine-tuning.
Data sources: FEVER (gold evidence) + TruthfulQA
Format: JSONL with chat messages (compatible with LLaMA-Factory / veRL)

Usage:
    python scripts/generate_sft_data.py --total 2000 --output data/sft_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are a fact verification assistant. Given a claim and evidence, classify the claim as:
- S (Supported): The evidence supports the claim
- C (Contradicted): The evidence contradicts the claim
- N (Not Enough Info): The evidence is insufficient to verify the claim

Respond with ONLY a JSON object: {"label": "S/C/N", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""

USER_TEMPLATE = """Claim: {claim}

Evidence: {evidence}

Classify this claim as S (Supported), C (Contradicted), or N (Not Enough Info).
Respond with JSON only."""

TEACHER_SYSTEM = """You are an expert fact verification teacher. Given a claim, evidence, and the CORRECT gold label, generate a high-quality verification response.

Your response must be a JSON object with:
- "label": the gold label (S, C, or N)
- "confidence": a realistic confidence score (0.7-0.99 for clear cases, 0.5-0.7 for ambiguous)
- "reasoning": a clear, step-by-step explanation (2-3 sentences) of WHY this label is correct.
  Focus on specific evidence alignment or gaps. Be concise but thorough.

IMPORTANT:
- Match the gold label exactly
- Make the reasoning educational - it should teach a student model HOW to verify claims
- For S: explain what in the evidence supports the claim
- For C: explain the specific contradiction between claim and evidence
- For N: explain what information is missing or why the evidence is insufficient"""

TEACHER_USER = """Claim: {claim}
Evidence: {evidence}
Gold label: {gold_label}

Generate the verification response JSON."""


def load_fever_samples(limit: int) -> list[dict]:
    """Load balanced FEVER samples with gold evidence."""
    from datasets import load_dataset
    ds = load_dataset("copenlu/fever_gold_evidence", split="train")

    label_map = {"SUPPORTS": "S", "REFUTES": "C", "NOT ENOUGH INFO": "N"}
    by_label: dict[str, list] = {"S": [], "C": [], "N": []}

    for row in ds:
        label = label_map.get(row["label"])
        if label is None:
            continue
        ev_list = row.get("evidence", [])
        evidence = " ".join(e[2] if len(e) > 2 else str(e) for e in ev_list) if ev_list else ""
        if not evidence.strip():
            continue
        by_label[label].append({"claim": row["claim"], "evidence": evidence, "gold": label})

    per_class = limit // 3
    rng = random.Random(42)
    samples = []
    for label in ["S", "C", "N"]:
        pool = by_label[label]
        rng.shuffle(pool)
        samples.extend(pool[:per_class])

    rng.shuffle(samples)
    return samples


def load_truthfulqa_samples(limit: int) -> list[dict]:
    """Load balanced TruthfulQA samples."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    rng = random.Random(42)
    all_rows = list(ds)
    rng.shuffle(all_rows)

    per_class = limit // 3
    s_samples, c_samples, n_samples = [], [], []

    for row in all_rows:
        question = row["question"]
        correct = row.get("correct_answers", [])
        incorrect = row.get("incorrect_answers", [])

        if len(s_samples) < per_class and correct:
            s_samples.append({"claim": question, "evidence": correct[0], "gold": "S"})
        if len(c_samples) < per_class and incorrect:
            c_samples.append({"claim": question, "evidence": incorrect[0], "gold": "C"})
        if len(s_samples) >= per_class and len(c_samples) >= per_class:
            break

    for i in range(min(per_class, len(all_rows))):
        j = (i + len(all_rows) // 2) % len(all_rows)
        n_samples.append({
            "claim": all_rows[i]["question"],
            "evidence": all_rows[j]["best_answer"],
            "gold": "N",
        })
        if len(n_samples) >= per_class:
            break

    samples = s_samples + c_samples + n_samples
    rng.shuffle(samples)
    return samples


def generate_teacher_response(claim: str, evidence: str, gold_label: str, model: str = "gpt-4o") -> dict | None:
    """Use GPT-4o to generate a high-quality verification trajectory."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": TEACHER_USER.format(
                    claim=claim, evidence=evidence, gold_label=gold_label
                )},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()

        # Parse JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        data = json.loads(text[start:end])

        # Validate
        if data.get("label") != gold_label:
            data["label"] = gold_label  # Force correct label
        if "reasoning" not in data or len(data["reasoning"]) < 10:
            return None
        if "confidence" not in data:
            data["confidence"] = 0.85

        return data
    except Exception as e:
        print(f"  Error: {e}")
        return None


def format_sft_sample(claim: str, evidence: str, teacher_response: dict) -> dict:
    """Format as chat messages for SFT training."""
    user_msg = USER_TEMPLATE.format(claim=claim, evidence=evidence)
    assistant_msg = json.dumps(teacher_response, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "gold_label": teacher_response["label"],
            "source": "teacher_gpt4o",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=2000, help="Total samples to generate")
    parser.add_argument("--fever-ratio", type=float, default=0.7, help="Fraction from FEVER")
    parser.add_argument("--model", default="gpt-4o", help="Teacher model")
    parser.add_argument("--output", default="data/sft_train.jsonl")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing if resuming
    existing = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                claim = d["messages"][1]["content"].split("\n")[0].replace("Claim: ", "")
                existing.add(claim[:50])
        print(f"Resuming: {len(existing)} existing samples")

    # Load source data
    n_fever = int(args.total * args.fever_ratio)
    n_tqa = args.total - n_fever

    print(f"Loading data: {n_fever} FEVER + {n_tqa} TruthfulQA = {args.total} total")
    fever_samples = load_fever_samples(n_fever)
    tqa_samples = load_truthfulqa_samples(n_tqa)
    all_samples = fever_samples + tqa_samples
    random.Random(42).shuffle(all_samples)

    print(f"Loaded {len(all_samples)} samples")
    print(f"Distribution: {dict(Counter(s['gold'] for s in all_samples))}")

    # Generate teacher responses
    mode = "a" if args.resume else "w"
    generated = 0
    failed = 0
    t0 = time.time()

    with open(output_path, mode) as f:
        for i, sample in enumerate(all_samples):
            # Skip if already done
            if sample["claim"][:50] in existing:
                continue

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = generated / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{len(all_samples)}] generated={generated}, failed={failed}, "
                      f"rate={rate:.1f}/s, elapsed={elapsed:.0f}s")

            response = generate_teacher_response(
                sample["claim"], sample["evidence"], sample["gold"], args.model
            )

            if response is None:
                failed += 1
                continue

            sft_entry = format_sft_sample(sample["claim"], sample["evidence"], response)
            f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1

    elapsed = time.time() - t0
    print(f"\nDone! Generated {generated} samples in {elapsed:.0f}s ({failed} failed)")
    print(f"Saved to {output_path}")

    # Print distribution
    labels = Counter()
    with open(output_path) as f:
        for line in f:
            d = json.loads(line)
            labels[d["metadata"]["gold_label"]] += 1
    print(f"Final distribution: {dict(labels)}")


if __name__ == "__main__":
    main()
