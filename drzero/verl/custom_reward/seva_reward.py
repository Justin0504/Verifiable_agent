"""SEVA v2 Process Reward Function for GRPO training.

Unlike the binary attribution_reward.py, this reward function scores
each component of the structured output INDEPENDENTLY, providing
fine-grained learning signal even when the final label is wrong.

Reward decomposition:
  R_format     (0.10): Valid JSON with required fields
  R_alignment  (0.25): Evidence spans are well-formed and grounded
  R_chain      (0.25): Reasoning chain is coherent and multi-step
  R_label      (0.25): Correct final label
  R_diagnosis  (0.15): Error type + fix suggestion (if applicable)

Key innovation over binary reward:
  - A response with correct alignment but wrong label still gets ~0.35
  - This creates a smooth reward landscape for GRPO exploration
  - The model learns process (HOW to verify) not just outcome (WHAT label)
"""

import json
import re
from typing import Any


# ============================================================
# Valid values
# ============================================================
VALID_LABELS = {"Attributable", "Not Attributable"}
VALID_STATUSES = {"match", "mismatch", "not_found"}
VALID_JUDGMENTS = {"supported", "not_supported", "partially_supported"}
VALID_ERROR_TYPES = {
    "numerical_exaggeration", "negation_flip", "scope_inflation",
    "temporal_shift", "entity_substitution", "fabrication",
}

LABEL_ALIASES = {
    "attributable": "Attributable",
    "not attributable": "Not Attributable",
    "not_attributable": "Not Attributable",
    "supported": "Attributable",
    "not supported": "Not Attributable",
    "yes": "Attributable",
    "no": "Not Attributable",
    "true": "Attributable",
    "false": "Not Attributable",
    "entailment": "Attributable",
    "contradiction": "Not Attributable",
    "neutral": "Not Attributable",
}


# ============================================================
# JSON extraction
# ============================================================
def extract_json_from_response(text: str) -> dict | None:
    """Extract JSON object from model response."""
    text = text.strip()
    # Remove markdown fences
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "")

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def normalize_label(label: str) -> str | None:
    """Normalize label to canonical form."""
    if not label:
        return None
    label_lower = label.strip().lower()
    if label_lower in LABEL_ALIASES:
        return LABEL_ALIASES[label_lower]
    for valid in VALID_LABELS:
        if valid.lower() in label_lower:
            return valid
    return None


# ============================================================
# Component reward functions
# ============================================================
def score_format(parsed: dict | None) -> float:
    """R_format: Does the response parse as valid JSON with required fields?

    Partial credit:
      0.0: not valid JSON
      0.5: valid JSON but missing key fields
      1.0: valid JSON with all required fields
    """
    if parsed is None:
        return 0.0

    required = {"evidence_alignment", "reasoning_chain", "label", "confidence"}
    present = required & set(parsed.keys())

    if len(present) == len(required):
        return 1.0
    elif len(present) >= 2:
        return 0.5
    else:
        return 0.2  # at least it's valid JSON


def score_alignment(parsed: dict) -> float:
    """R_alignment: Quality of evidence_alignment spans.

    Checks:
      - Is it a non-empty list?
      - Do entries have claim_span, source_span, status?
      - Are statuses valid?
      - Are spans non-trivial (not empty, not too long)?

    Returns 0.0-1.0
    """
    alignment = parsed.get("evidence_alignment", [])
    if not isinstance(alignment, list) or len(alignment) == 0:
        return 0.0

    total_score = 0.0
    for entry in alignment:
        entry_score = 0.0
        if not isinstance(entry, dict):
            continue

        # Has required fields
        has_claim = bool(entry.get("claim_span", "").strip())
        has_source = bool(entry.get("source_span", "").strip())
        has_status = entry.get("status", "") in VALID_STATUSES

        if has_claim:
            entry_score += 0.3
        if has_source or entry.get("source_span") == "NOT_FOUND":
            entry_score += 0.3
        if has_status:
            entry_score += 0.2

        # Span quality: not trivially short or absurdly long
        claim_span = entry.get("claim_span", "")
        source_span = entry.get("source_span", "")
        if 3 <= len(claim_span) <= 200:
            entry_score += 0.1
        if (3 <= len(source_span) <= 500) or source_span == "NOT_FOUND":
            entry_score += 0.1

        total_score += entry_score

    # Normalize by number of entries, cap at 1.0
    avg = total_score / len(alignment)
    return min(avg, 1.0)


def score_chain(parsed: dict) -> float:
    """R_chain: Quality of reasoning_chain.

    Checks:
      - Is it a non-empty list with >= 2 steps?
      - Do steps have required fields (judgment, explanation)?
      - Are judgments valid?
      - Are explanations non-trivial (>= 10 chars)?
      - Is there a source_evidence reference?

    Returns 0.0-1.0
    """
    chain = parsed.get("reasoning_chain", [])
    if not isinstance(chain, list) or len(chain) == 0:
        return 0.0

    # Bonus for multi-step reasoning
    length_bonus = min(len(chain) / 3.0, 1.0) * 0.2

    total_score = 0.0
    for step in chain:
        step_score = 0.0
        if not isinstance(step, dict):
            continue

        # Has judgment
        if step.get("judgment", "") in VALID_JUDGMENTS:
            step_score += 0.3

        # Has explanation (non-trivial)
        explanation = step.get("explanation", "")
        if len(explanation) >= 10:
            step_score += 0.3
        elif len(explanation) >= 3:
            step_score += 0.1

        # References source evidence
        source_ev = step.get("source_evidence", "")
        if source_ev and len(source_ev) >= 5:
            step_score += 0.2

        # Has claim_part
        if step.get("claim_part", ""):
            step_score += 0.2

        total_score += step_score

    avg = total_score / len(chain)
    return min(avg + length_bonus, 1.0)


def score_label(parsed: dict, gold_label: str) -> float:
    """R_label: Correct final label.

    Returns:
      1.0: correct label
      0.0: wrong label
    """
    pred_raw = parsed.get("label", "")
    pred = normalize_label(pred_raw)
    if pred is None:
        return 0.0
    return 1.0 if pred == gold_label else 0.0


def score_diagnosis(parsed: dict, gold_label: str) -> float:
    """R_diagnosis: Error type + fix suggestion quality.

    Only applicable when gold_label is "Not Attributable".
    For "Attributable", gives full credit if no error_type is present
    (correctly omitted).

    Returns 0.0-1.0
    """
    if gold_label == "Attributable":
        # Should NOT have error_type → reward for correct omission
        if not parsed.get("error_type"):
            return 1.0
        else:
            return 0.3  # Has error_type when shouldn't, but not catastrophic

    # Gold is "Not Attributable" → should have error_type + fix_suggestion
    error_type = parsed.get("error_type", "")
    fix = parsed.get("fix_suggestion", "")

    score = 0.0

    # Valid error type
    if error_type in VALID_ERROR_TYPES:
        score += 0.6
    elif error_type:
        score += 0.2  # some error type, just not from taxonomy

    # Fix suggestion quality
    if fix and len(fix) >= 10:
        score += 0.4
    elif fix:
        score += 0.1

    return score


# ============================================================
# Confidence calibration bonus
# ============================================================
def score_calibration(parsed: dict, label_correct: bool) -> float:
    """Calibration bonus: reward confidence aligned with correctness.

    Correct + high confidence → positive bonus
    Wrong + high confidence → negative penalty (capped at 0)
    """
    conf = parsed.get("confidence", 0.5)
    if not isinstance(conf, (int, float)):
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.5
    conf = min(max(conf, 0.0), 1.0)

    if label_correct:
        return conf * 0.15  # up to +0.15
    else:
        return -conf * 0.1  # up to -0.1 (penalize overconfident wrong answers)


# ============================================================
# Main reward function
# ============================================================
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """Compute SEVA v2 process reward for a single response.

    The key difference from attribution_reward.py:
    - Each component is scored independently
    - A response with good alignment + chain but wrong label still gets ~0.35
    - This provides smooth gradient for GRPO to learn from

    Args:
        data_source: task identifier
        solution_str: model's raw text output
        ground_truth: {"target": "Attributable" or "Not Attributable"}
        extra_info: optional metadata with "epoch", "total_epochs"

    Returns:
        float: scalar reward in [0, ~1.8]
    """
    gold_label = ground_truth.get("target", "Not Attributable")
    extra_info = extra_info or {}

    # === Weights: process-heavy to ensure smooth reward landscape ===
    # alignment + chain = 0.60 > label = 0.15, so good reasoning with
    # wrong label still beats correct label with no reasoning.
    W = {
        "format": 0.10,
        "alignment": 0.30,
        "chain": 0.30,
        "label": 0.15,
        "diagnosis": 0.15,
    }

    # === Parse response ===
    parsed = extract_json_from_response(solution_str)

    # Component 1: Format
    r_format = W["format"] * score_format(parsed)

    if parsed is None:
        return r_format  # 0.0 — can't score anything else

    # Component 2: Evidence alignment
    r_alignment = W["alignment"] * score_alignment(parsed)

    # Component 3: Reasoning chain
    r_chain = W["chain"] * score_chain(parsed)

    # Component 4: Label accuracy
    label_correct = score_label(parsed, gold_label) == 1.0
    r_label = W["label"] * (1.0 if label_correct else 0.0)

    # Component 5: Diagnosis (error type + fix)
    r_diagnosis = W["diagnosis"] * score_diagnosis(parsed, gold_label)

    # Calibration bonus/penalty (additive, not weighted)
    r_calibration = score_calibration(parsed, label_correct)

    total = r_format + r_alignment + r_chain + r_label + r_diagnosis + r_calibration

    return max(total, 0.0)


def apply_boundary_bonus(scores: list[float], group_size: int = 5) -> list[float]:
    """Group-level boundary-optimal weighting (Dr.Zero).

    Same as attribution_reward.py but adapted for process reward scale.
    """
    if len(scores) < group_size:
        return scores

    bonused = []
    for i in range(0, len(scores), group_size):
        group = scores[i:i + group_size]
        # For process reward, "correct" = got above median reward
        median = sorted(group)[len(group) // 2]
        correct_count = sum(1 for s in group if s > median)
        correct_rate = correct_count / len(group)
        boundary = 1.0 - abs(correct_rate - 0.5) * 2.0
        scale = 0.5 + 0.5 * boundary
        for s in group:
            bonused.append(s * scale)

    remainder = len(scores) % group_size
    if remainder > 0:
        bonused.extend(scores[-remainder:])

    return bonused


def compute_score_batch(
    data_sources, solution_strs, ground_truths, extra_infos, **kwargs
):
    """Batch reward computation for veRL's NaiveRewardManager."""
    scores = []
    for ds, sol, gt, ei in zip(data_sources, solution_strs,
                                ground_truths, extra_infos):
        score = compute_score(ds, sol, gt, ei)
        scores.append(score)

    group_size = kwargs.get("group_size", 5)
    scores = apply_boundary_bonus(scores, group_size=group_size)

    return scores
