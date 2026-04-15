"""Post-RL evaluation: compare zero-shot → SFT → GRPO on fact verification test set.

Usage:
    python post_rl_eval.py --model /path/to/grpo_checkpoint --test-data /path/to/test.parquet
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_json_from_response(text: str) -> dict | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    label_match = re.search(r'"label"\s*:\s*"([SCN])"', text, re.IGNORECASE)
    if label_match:
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
        reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
        return {
            "label": label_match.group(1).upper(),
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "reasoning": reason_match.group(1) if reason_match else "",
        }
    return None


def evaluate_model(model, tokenizer, test_df, device, max_new_tokens=256):
    results = []
    correct = 0
    total = 0
    format_errors = 0
    class_correct = Counter()
    class_total = Counter()

    for i, row in test_df.iterrows():
        prompt = row["prompt"]
        gold = row["reward_model"]["ground_truth"]["target"]

        # Build chat input
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed = extract_json_from_response(response)

        pred = None
        if parsed and parsed.get("label") in ("S", "C", "N"):
            pred = parsed["label"]
        else:
            format_errors += 1

        is_correct = pred == gold
        if is_correct:
            correct += 1
            class_correct[gold] += 1
        total += 1
        class_total[gold] += 1

        results.append({
            "index": i,
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "response": response[:300],
            "confidence": parsed.get("confidence") if parsed else None,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_df)}] acc={correct/total:.3f}, fmt_err={format_errors}")

    return results, correct, total, format_errors, class_correct, class_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", required=True, help="Path to test.parquet")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()

    print(f"Loading test data: {args.test_data}")
    test_df = pd.read_parquet(args.test_data)
    print(f"  {len(test_df)} samples")

    print("Running evaluation...")
    results, correct, total, fmt_err, class_correct, class_total = evaluate_model(
        model, tokenizer, test_df, args.device
    )

    # Overall accuracy
    acc = correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {correct}/{total} = {acc:.1%}")
    print(f"Format Errors: {fmt_err}/{total} = {fmt_err/total:.1%}")

    # Per-class accuracy
    print(f"\nPer-class:")
    for label in ["S", "C", "N"]:
        ct = class_total[label]
        cc = class_correct[label]
        print(f"  {label}: {cc}/{ct} = {cc/ct:.1%}" if ct > 0 else f"  {label}: N/A")

    # Save results
    output_path = args.output or f"eval_results_grpo.json"
    summary = {
        "model": args.model,
        "test_data": args.test_data,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "format_errors": fmt_err,
        "per_class": {
            label: {"correct": class_correct[label], "total": class_total[label],
                     "accuracy": class_correct[label] / class_total[label] if class_total[label] > 0 else 0}
            for label in ["S", "C", "N"]
        },
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
