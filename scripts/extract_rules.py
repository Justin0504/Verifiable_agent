"""Extract verification rules from successful reasoning chains (REFLECT phase).

Analyzes model outputs where the model got the answer RIGHT,
extracts reusable verification patterns, and builds a ReasoningBank
that can be injected into future prompts.

Usage:
    python scripts/extract_rules.py \
        --results-file results/seva_eval/clearfacts_results.json \
        --output-dir results/reasoning_bank/round0

    # Update existing bank with new results
    python scripts/extract_rules.py \
        --results-file results/seva_eval/clearfacts_results.json \
        --existing-bank results/reasoning_bank/round0/bank.json \
        --output-dir results/reasoning_bank/round1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "drzero" / "verl" / "custom_reward"))

from seva_reward import extract_json_from_response, normalize_label, VALID_ERROR_TYPES
from src.llm.openai_llm import OpenAILLM


# ============================================================
# Rule extraction prompts
# ============================================================

RULE_EXTRACTION_PROMPT = """\
You are analyzing successful fact verification reasoning chains to extract \
reusable verification rules.

Below are {n} examples where a verifier correctly identified whether a claim \
was attributable to a source. Each example includes the reasoning chain the \
verifier used.

Extract {k} general verification rules from these examples. Each rule should be:
1. GENERAL: applicable beyond these specific examples
2. ACTIONABLE: tells a verifier what to check and how
3. SPECIFIC: not vague ("check carefully" is bad; "compare exact numbers" is good)
4. GROUNDED: derived from patterns you observe across multiple examples

Output a JSON array of rules:
[
  {{
    "id": "R{start_id}",
    "name": "Short descriptive name",
    "content": "The rule in 1-2 sentences. Be precise.",
    "error_types": ["which error types this rule helps detect"],
    "source_examples": [list of example indices that demonstrate this rule]
  }}
]

Examples:
{examples}

Output ONLY the JSON array."""


def collect_successful_chains(details: list) -> list[dict]:
    """Collect correct predictions with high-quality reasoning chains."""
    successes = []

    for i, d in enumerate(details):
        if not d.get("correct"):
            continue

        response = d.get("response", "")
        parsed = extract_json_from_response(response)
        if parsed is None:
            continue

        chain = parsed.get("reasoning_chain", [])
        alignment = parsed.get("evidence_alignment", [])

        # Quality filter: must have real reasoning
        if not chain or len(chain) < 2:
            continue
        if not alignment:
            continue

        # Chain quality: steps must have explanations
        good_steps = sum(
            1 for s in chain
            if isinstance(s, dict)
            and len(s.get("explanation", "")) >= 15
            and s.get("judgment") in ("supported", "not_supported", "partially_supported")
        )
        if good_steps < 2:
            continue

        successes.append({
            "index": i,
            "claim": d.get("claim", ""),
            "gold": d["gold"],
            "pred": d["pred"],
            "confidence": d.get("confidence", 0),
            "error_type": parsed.get("error_type", ""),
            "alignment": alignment,
            "chain": chain,
            "alignment_score": d.get("alignment_score", 0),
            "chain_score": d.get("chain_score", 0),
        })

    return successes


def format_examples_for_extraction(successes: list, max_examples: int = 20) -> str:
    """Format successful chains for the rule extraction prompt."""
    # Sample diverse examples: mix of Attributable and Not Attributable
    attr = [s for s in successes if s["gold"] == "Attributable"]
    na = [s for s in successes if s["gold"] == "Not Attributable"]

    # Take highest quality from each
    attr.sort(key=lambda x: x["chain_score"], reverse=True)
    na.sort(key=lambda x: x["chain_score"], reverse=True)

    n_each = max_examples // 2
    selected = attr[:n_each] + na[:n_each]

    lines = []
    for idx, s in enumerate(selected):
        lines.append(f"--- Example {idx} (Gold: {s['gold']}) ---")
        lines.append(f"Claim: {s['claim'][:200]}")

        # Show reasoning chain
        for step in s["chain"][:4]:
            if isinstance(step, dict):
                j = step.get("judgment", "?")
                e = step.get("explanation", "")[:150]
                cp = step.get("claim_part", "")[:80]
                lines.append(f"  Step: [{j}] {cp} — {e}")

        if s.get("error_type"):
            lines.append(f"  Error type: {s['error_type']}")
        lines.append("")

    return "\n".join(lines)


def extract_rules_with_llm(llm: OpenAILLM, successes: list,
                           num_rules: int = 10,
                           existing_rules: list = None) -> list[dict]:
    """Use LLM to extract verification rules from successful chains."""
    start_id = 1
    if existing_rules:
        max_id = max(int(r["id"].replace("R", "")) for r in existing_rules
                     if r["id"].startswith("R"))
        start_id = max_id + 1

    examples_text = format_examples_for_extraction(successes, max_examples=20)

    prompt = RULE_EXTRACTION_PROMPT.format(
        n=min(len(successes), 20),
        k=num_rules,
        start_id=start_id,
        examples=examples_text,
    )

    # Add existing rules context to avoid duplicates
    if existing_rules:
        existing_text = "\n".join(
            f"  {r['id']}: {r['content']}" for r in existing_rules
        )
        prompt += (
            f"\n\nExisting rules (DO NOT duplicate these):\n{existing_text}\n"
            f"Generate NEW rules that are different from the above."
        )

    try:
        response = llm.generate(prompt, system="You extract verification rules from examples. Output only JSON.")
        text = response.text.strip()

        # Extract JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        rules = json.loads(text[start:end])
        if not isinstance(rules, list):
            return []

        # Validate each rule
        valid_rules = []
        for r in rules:
            if isinstance(r, dict) and r.get("content") and r.get("name"):
                if "id" not in r:
                    r["id"] = f"R{start_id + len(valid_rules)}"
                r["effectiveness"] = 0.5  # initial score
                r["citations"] = 0
                r["epoch_created"] = None  # filled by caller
                valid_rules.append(r)

        return valid_rules

    except (json.JSONDecodeError, Exception) as e:
        print(f"  ERROR extracting rules: {e}")
        return []


def extract_rules_heuristic(successes: list) -> list[dict]:
    """Extract rules using heuristic patterns (no LLM needed).

    Looks for common patterns in successful reasoning chains.
    """
    rules = []

    # Pattern 1: Number comparison rules
    number_chains = [
        s for s in successes
        if s["gold"] == "Not Attributable"
        and s.get("error_type") == "numerical_exaggeration"
    ]
    if len(number_chains) >= 3:
        rules.append({
            "id": "RH1",
            "name": "Exact number verification",
            "content": (
                "When the claim contains specific numbers (percentages, counts, "
                "dates, measurements), verify exact match with the source. "
                "Even small differences (e.g., '30%' vs '29%') count as "
                "not attributable if the source is precise."
            ),
            "error_types": ["numerical_exaggeration"],
            "source": "heuristic",
            "effectiveness": 0.5,
            "citations": 0,
        })

    # Pattern 2: Scope/qualifier checking
    scope_chains = [
        s for s in successes
        if s["gold"] == "Not Attributable"
        and s.get("error_type") in ("scope_inflation", "temporal_shift")
    ]
    if len(scope_chains) >= 3:
        rules.append({
            "id": "RH2",
            "name": "Qualifier and scope checking",
            "content": (
                "Check if the claim uses absolute terms ('all', 'every', 'always', "
                "'never') where the source uses qualified terms ('some', 'many', "
                "'often', 'in some cases'). Also check temporal qualifiers: "
                "'recently' vs specific dates, 'always' vs 'since 2020'."
            ),
            "error_types": ["scope_inflation", "temporal_shift"],
            "source": "heuristic",
            "effectiveness": 0.5,
            "citations": 0,
        })

    # Pattern 3: Entity verification
    entity_chains = [
        s for s in successes
        if s["gold"] == "Not Attributable"
        and s.get("error_type") == "entity_substitution"
    ]
    if len(entity_chains) >= 2:
        rules.append({
            "id": "RH3",
            "name": "Entity identity verification",
            "content": (
                "Verify that named entities (people, organizations, locations) "
                "in the claim exactly match those in the source. Watch for "
                "subtle swaps: similar names, parent/subsidiary companies, "
                "or geographic confusions (city vs country)."
            ),
            "error_types": ["entity_substitution"],
            "source": "heuristic",
            "effectiveness": 0.5,
            "citations": 0,
        })

    # Pattern 4: Negation detection
    negation_chains = [
        s for s in successes
        if s["gold"] == "Not Attributable"
        and s.get("error_type") == "negation_flip"
    ]
    if len(negation_chains) >= 2:
        rules.append({
            "id": "RH4",
            "name": "Negation and polarity check",
            "content": (
                "Check whether the claim preserves the polarity of the source. "
                "Watch for: added/removed 'not', 'no', 'never', 'without'; "
                "antonym substitution ('increase' vs 'decrease'); and "
                "double negatives that flip meaning."
            ),
            "error_types": ["negation_flip"],
            "source": "heuristic",
            "effectiveness": 0.5,
            "citations": 0,
        })

    # Pattern 5: NOT_FOUND span pattern
    not_found_chains = [
        s for s in successes
        if s["gold"] == "Not Attributable"
        and any(
            a.get("status") == "not_found"
            for a in s.get("alignment", [])
            if isinstance(a, dict)
        )
    ]
    if len(not_found_chains) >= 3:
        rules.append({
            "id": "RH5",
            "name": "Fabricated detail detection",
            "content": (
                "If any key phrase from the claim cannot be found in or "
                "paraphrased from the source (alignment status = not_found), "
                "the claim likely contains fabricated information. This is "
                "especially suspicious for specific claims (names, numbers, "
                "technical terms)."
            ),
            "error_types": ["fabrication"],
            "source": "heuristic",
            "effectiveness": 0.5,
            "citations": 0,
        })

    return rules


def update_rule_effectiveness(bank: list[dict], results_details: list) -> list[dict]:
    """Update rule effectiveness based on new evaluation results.

    Rules that the model "used" (by following similar reasoning patterns)
    in correct predictions get higher effectiveness scores.
    """
    # For now, a simplified version: track which error types the model
    # handles well, and boost rules targeting those types
    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)

    for d in results_details:
        response = d.get("response", "")
        parsed = extract_json_from_response(response)
        if parsed is None:
            continue

        etype = parsed.get("error_type", "general")
        total_by_type[etype] += 1
        if d.get("correct"):
            correct_by_type[etype] += 1

    # Update each rule
    for rule in bank:
        rule_types = rule.get("error_types", [])
        if not rule_types:
            continue

        # Average accuracy on the error types this rule targets
        accuracies = []
        for et in rule_types:
            if total_by_type[et] > 0:
                accuracies.append(correct_by_type[et] / total_by_type[et])

        if accuracies:
            new_eff = sum(accuracies) / len(accuracies)
            # Exponential moving average with existing effectiveness
            old_eff = rule.get("effectiveness", 0.5)
            rule["effectiveness"] = round(0.6 * new_eff + 0.4 * old_eff, 3)

    return bank


def evict_weak_rules(bank: list[dict], min_effectiveness: float = 0.2,
                     min_citations: int = 3) -> tuple[list[dict], list[dict]]:
    """Remove rules that have proven ineffective.

    Only evict rules that have been used enough times (citations >= min_citations)
    but still have low effectiveness.
    """
    kept = []
    evicted = []

    for rule in bank:
        citations = rule.get("citations", 0)
        effectiveness = rule.get("effectiveness", 0.5)

        if citations >= min_citations and effectiveness < min_effectiveness:
            evicted.append(rule)
        else:
            kept.append(rule)

    return kept, evicted


def format_bank_for_prompt(bank: list[dict], top_k: int = 5) -> str:
    """Format top-K rules for injection into the verifier's system prompt."""
    # Sort by effectiveness
    sorted_bank = sorted(bank, key=lambda r: r.get("effectiveness", 0),
                         reverse=True)

    lines = ["Verification rules to apply:"]
    for rule in sorted_bank[:top_k]:
        lines.append(f"- [{rule['id']}] {rule['content']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=str, required=True,
                        help="Eval results JSON from eval_seva.py")
    parser.add_argument("--existing-bank", type=str, default=None,
                        help="Existing reasoning bank JSON to update")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-rules", type=int, default=10,
                        help="Number of new rules to extract")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM for rule extraction (requires API key)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K rules for prompt injection")
    parser.add_argument("--epoch", type=int, default=0,
                        help="Current evolution epoch number")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results: {args.results_file}")
    with open(args.results_file) as f:
        results = json.load(f)
    details = results.get("details", [])
    print(f"  {len(details)} samples")

    # Load existing bank
    existing_bank = []
    if args.existing_bank:
        with open(args.existing_bank) as f:
            existing_bank = json.load(f)
        print(f"  Existing bank: {len(existing_bank)} rules")

    # Collect successful chains
    print("\n1. Collecting successful reasoning chains...")
    successes = collect_successful_chains(details)
    print(f"  {len(successes)} high-quality correct predictions")

    # Error type distribution in successes
    etype_dist = Counter(s.get("error_type", "none") for s in successes)
    for etype, count in etype_dist.most_common(10):
        print(f"    {etype:30s}  {count}")

    # Extract rules
    print("\n2. Extracting verification rules...")
    if args.use_llm:
        llm = OpenAILLM(model=args.model, temperature=0.3)
        new_rules = extract_rules_with_llm(
            llm, successes, args.num_rules, existing_bank
        )
        print(f"  LLM extracted: {len(new_rules)} rules")
    else:
        new_rules = []

    # Always add heuristic rules
    heuristic_rules = extract_rules_heuristic(successes)
    print(f"  Heuristic extracted: {len(heuristic_rules)} rules")

    # Merge: avoid duplicate IDs
    existing_ids = {r["id"] for r in existing_bank}
    for r in new_rules + heuristic_rules:
        if r["id"] not in existing_ids:
            r["epoch_created"] = args.epoch
            existing_bank.append(r)
            existing_ids.add(r["id"])

    print(f"  Bank size after merge: {len(existing_bank)} rules")

    # Update effectiveness
    print("\n3. Updating rule effectiveness...")
    existing_bank = update_rule_effectiveness(existing_bank, details)

    for r in sorted(existing_bank, key=lambda x: x.get("effectiveness", 0),
                    reverse=True):
        print(f"  {r['id']:5s}  eff={r.get('effectiveness', 0):.3f}  "
              f"cite={r.get('citations', 0):>3d}  {r['name']}")

    # Evict weak rules
    print("\n4. Evicting weak rules...")
    existing_bank, evicted = evict_weak_rules(existing_bank)
    if evicted:
        for r in evicted:
            print(f"  EVICTED: {r['id']} {r['name']} "
                  f"(eff={r.get('effectiveness', 0):.3f})")
    else:
        print("  No rules evicted (all above threshold or insufficient citations)")

    # Generate prompt injection text
    print("\n5. Generating prompt injection...")
    prompt_text = format_bank_for_prompt(existing_bank, top_k=args.top_k)
    print(f"  Top-{args.top_k} rules for injection:")
    print(f"  {prompt_text[:500]}")

    # Save outputs
    bank_path = output_dir / "bank.json"
    with open(bank_path, "w") as f:
        json.dump(existing_bank, f, indent=2, default=str)

    prompt_path = output_dir / "rules_prompt.txt"
    with open(prompt_path, "w") as f:
        f.write(prompt_text)

    if evicted:
        evicted_path = output_dir / "evicted_rules.json"
        with open(evicted_path, "w") as f:
            json.dump(evicted, f, indent=2, default=str)

    # Save bank summary for paper
    summary = {
        "epoch": args.epoch,
        "total_rules": len(existing_bank),
        "new_rules": len(new_rules) + len(heuristic_rules),
        "evicted_rules": len(evicted),
        "avg_effectiveness": round(
            sum(r.get("effectiveness", 0) for r in existing_bank)
            / max(len(existing_bank), 1), 3
        ),
        "top_rules": [
            {"id": r["id"], "name": r["name"],
             "effectiveness": r.get("effectiveness", 0),
             "error_types": r.get("error_types", [])}
            for r in sorted(existing_bank,
                          key=lambda x: x.get("effectiveness", 0),
                          reverse=True)[:5]
        ],
    }
    with open(output_dir / "bank_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"REFLECT phase complete (epoch {args.epoch})")
    print(f"  Bank: {len(existing_bank)} rules (avg eff: {summary['avg_effectiveness']})")
    print(f"  New:  {summary['new_rules']}  Evicted: {summary['evicted_rules']}")
    print(f"  Output: {output_dir}")
    print(f"  Inject: {prompt_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
