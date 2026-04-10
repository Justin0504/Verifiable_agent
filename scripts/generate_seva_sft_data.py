"""Generate SEVA v2 SFT training data with structured reasoning chains.

Uses GPT-4o as teacher to annotate existing (claim, source, label) triples
with multi-granularity structured output:
  - evidence_alignment: claim_span → source_span mappings
  - reasoning_chain: step-by-step grounded verification
  - error_type + fix_suggestion (for Not Attributable)

Usage:
    python scripts/generate_seva_sft_data.py --input data/attribution/sft_train.jsonl
    python scripts/generate_seva_sft_data.py --input data/attribution/sft_train.jsonl --max-samples 1000 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.verifier.seva_format import (
    ERROR_TYPES,
    TEACHER_SYSTEM_PROMPT,
    TEACHER_USER_TEMPLATE,
    SEVA_SYSTEM_PROMPT,
    SEVA_USER_TEMPLATE,
)
from src.llm.openai_llm import OpenAILLM


# ============================================================
# Validation
# ============================================================
REQUIRED_FIELDS = {"evidence_alignment", "reasoning_chain", "label", "confidence"}
VALID_LABELS = {"Attributable", "Not Attributable"}
VALID_STATUSES = {"match", "mismatch", "not_found"}
VALID_JUDGMENTS = {"supported", "not_supported", "partially_supported"}


def validate_structured_output(obj: dict, gold_label: str) -> tuple[bool, str]:
    """Validate teacher-generated structured output for correctness."""
    # Check required fields
    missing = REQUIRED_FIELDS - set(obj.keys())
    if missing:
        return False, f"Missing fields: {missing}"

    # Check label matches gold
    if obj.get("label") != gold_label:
        return False, f"Label mismatch: {obj.get('label')} != {gold_label}"

    # Check evidence_alignment
    alignment = obj.get("evidence_alignment", [])
    if not isinstance(alignment, list) or len(alignment) == 0:
        return False, "evidence_alignment must be a non-empty list"

    for i, entry in enumerate(alignment):
        if not isinstance(entry, dict):
            return False, f"alignment[{i}] is not a dict"
        if "claim_span" not in entry or "source_span" not in entry:
            return False, f"alignment[{i}] missing claim_span or source_span"
        status = entry.get("status", "")
        if status not in VALID_STATUSES:
            return False, f"alignment[{i}] invalid status: {status}"

    # Check reasoning_chain
    chain = obj.get("reasoning_chain", [])
    if not isinstance(chain, list) or len(chain) == 0:
        return False, "reasoning_chain must be a non-empty list"

    for i, step in enumerate(chain):
        if not isinstance(step, dict):
            return False, f"chain[{i}] is not a dict"
        if "judgment" not in step or "explanation" not in step:
            return False, f"chain[{i}] missing judgment or explanation"
        if step.get("judgment") not in VALID_JUDGMENTS:
            return False, f"chain[{i}] invalid judgment: {step.get('judgment')}"

    # Check confidence
    conf = obj.get("confidence", -1)
    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
        return False, f"confidence out of range: {conf}"

    # Check error_type for Not Attributable
    if gold_label == "Not Attributable":
        error_type = obj.get("error_type", "")
        if error_type not in ERROR_TYPES:
            return False, f"Invalid error_type: {error_type}"
        if not obj.get("fix_suggestion"):
            return False, "Missing fix_suggestion for Not Attributable"

    return True, "ok"


def extract_json(text: str) -> dict | None:
    """Extract JSON object from teacher response."""
    text = text.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# ============================================================
# Teacher annotation
# ============================================================
def annotate_sample(
    llm: OpenAILLM,
    claim: str,
    source: str,
    label: str,
    max_retries: int = 2,
) -> dict | None:
    """Call teacher model to generate structured annotation for one sample."""
    prompt = TEACHER_USER_TEMPLATE.format(claim=claim, source=source, label=label)

    for attempt in range(max_retries + 1):
        try:
            response = llm.generate(prompt, system=TEACHER_SYSTEM_PROMPT)
            obj = extract_json(response.text)
            if obj is None:
                continue

            valid, reason = validate_structured_output(obj, label)
            if valid:
                return obj
            elif attempt < max_retries:
                # Retry with hint about what went wrong
                prompt_retry = (
                    f"{prompt}\n\n"
                    f"Your previous response had an issue: {reason}. "
                    f"Please fix and regenerate."
                )
                response = llm.generate(prompt_retry, system=TEACHER_SYSTEM_PROMPT)
                obj = extract_json(response.text)
                if obj:
                    valid2, _ = validate_structured_output(obj, label)
                    if valid2:
                        return obj
        except Exception as e:
            if attempt == max_retries:
                print(f"  ERROR: {e}")
    return None


def make_seva_sft_sample(
    claim: str, source: str, structured_output: dict
) -> dict:
    """Create a chat-format SFT sample with SEVA v2 structured output."""
    messages = [
        {"role": "system", "content": SEVA_SYSTEM_PROMPT},
        {"role": "user", "content": SEVA_USER_TEMPLATE.format(claim=claim, source=source)},
        {"role": "assistant", "content": json.dumps(structured_output, ensure_ascii=False)},
    ]
    return {
        "messages": messages,
        "metadata": {
            "gold_label": structured_output["label"],
            "format": "seva_v2",
            "error_type": structured_output.get("error_type", ""),
        },
    }


def make_seva_grpo_sample(
    claim: str, source: str, label: str, data_source: str = "attribution"
) -> dict:
    """Create a veRL-format GRPO sample with SEVA v2 system prompt."""
    prompt = (
        f"<|im_start|>system\n{SEVA_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{SEVA_USER_TEMPLATE.format(claim=claim, source=source)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "fact_attribution",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": label},
        },
        "extra_info": {},
    }


# ============================================================
# Main pipeline
# ============================================================
def process_batch(
    llm: OpenAILLM,
    samples: list[dict],
    workers: int = 4,
) -> list[dict]:
    """Process a batch of samples with concurrent teacher calls."""
    results = []
    failed = 0

    def process_one(sample: dict) -> dict | None:
        # Extract claim/source/label from the chat format
        messages = sample.get("messages", [])
        if len(messages) < 2:
            return None
        user_msg = messages[1]["content"]
        # Parse claim and source from user message
        claim_start = user_msg.find("Claim: ") + 7
        source_start = user_msg.find("Source: ") + 8
        question_start = user_msg.find("\n\nIs this claim")

        claim = user_msg[claim_start:user_msg.find("\n\nSource:")].strip()
        if question_start > 0:
            source = user_msg[source_start:question_start].strip()
        else:
            source = user_msg[source_start:].strip()

        label = sample.get("metadata", {}).get("gold_label", "Not Attributable")

        structured = annotate_sample(llm, claim, source, label)
        if structured is None:
            return None

        return {
            "claim": claim,
            "source": source,
            "label": label,
            "structured_output": structured,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one, s): i
            for i, s in enumerate(samples)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                print(f"  Sample {idx} error: {e}")
                failed += 1

            done = len(results) + failed
            if done % 50 == 0:
                print(f"  Progress: {done}/{len(samples)} "
                      f"(success={len(results)}, failed={failed})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate SEVA v2 SFT data")
    parser.add_argument("--input", type=str,
                        default="data/attribution/sft_train.jsonl",
                        help="Input SFT data (old format)")
    parser.add_argument("--output-dir", type=str,
                        default="data/attribution",
                        help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to annotate")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent API calls")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Teacher model")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Teacher temperature (low for consistency)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input samples
    print(f"Loading input from {input_path}...")
    samples = []
    with open(input_path) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples")

    if args.max_samples:
        random.shuffle(samples)
        samples = samples[:args.max_samples]
        print(f"  Subsampled to {len(samples)}")

    # Initialize teacher LLM
    llm = OpenAILLM(
        model=args.model,
        temperature=args.temperature,
        max_tokens=2048,
    )

    # Process
    print(f"\nAnnotating with {args.model} (workers={args.workers})...")
    t0 = time.time()
    annotated = process_batch(llm, samples, workers=args.workers)
    elapsed = time.time() - t0
    print(f"\nDone: {len(annotated)} annotated in {elapsed:.0f}s "
          f"({len(annotated)/elapsed:.1f} samples/s)")

    # Save SFT data
    sft_path = output_dir / "seva_sft_train.jsonl"
    with open(sft_path, "w") as f:
        for item in annotated:
            sft_sample = make_seva_sft_sample(
                item["claim"], item["source"], item["structured_output"]
            )
            f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
    print(f"  SFT data: {len(annotated)} → {sft_path}")

    # Save GRPO data
    import pandas as pd

    grpo_records = []
    for item in annotated:
        grpo = make_seva_grpo_sample(
            item["claim"], item["source"], item["label"]
        )
        grpo_records.append({
            "data_source": grpo["data_source"],
            "prompt": grpo["prompt"],
            "ability": grpo["ability"],
            "reward_model": json.dumps(grpo["reward_model"]),
            "extra_info": json.dumps(grpo["extra_info"]),
        })

    df = pd.DataFrame(grpo_records)
    random.shuffle(grpo_records)  # shuffle before split
    val_size = min(500, len(df) // 10)

    train_df = df.iloc[val_size:]
    val_df = df.iloc[:val_size]

    train_path = output_dir / "seva_grpo_train.parquet"
    val_path = output_dir / "seva_grpo_val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    print(f"  GRPO train: {len(train_df)} → {train_path}")
    print(f"  GRPO val:   {len(val_df)} → {val_path}")

    # Save raw annotations (for analysis)
    raw_path = output_dir / "seva_annotations_raw.jsonl"
    with open(raw_path, "w") as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Raw annotations: {len(annotated)} → {raw_path}")

    # Stats
    label_dist = {}
    error_dist = {}
    for item in annotated:
        label = item["label"]
        label_dist[label] = label_dist.get(label, 0) + 1
        et = item["structured_output"].get("error_type", "")
        if et:
            error_dist[et] = error_dist.get(et, 0) + 1

    print(f"\n{'='*50}")
    print("Label distribution:")
    for l, c in sorted(label_dist.items()):
        print(f"  {l}: {c}")
    print("Error type distribution:")
    for e, c in sorted(error_dist.items()):
        print(f"  {e}: {c}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
