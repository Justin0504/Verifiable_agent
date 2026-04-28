"""Reward function for fact verification GRPO training.

Reward components:
1. Format reward (0.1): valid JSON with label ∈ {S, C, N}
2. Accuracy reward (0.0 or 1.0 * class_weight): label matches gold
3. Calibration bonus (-0.2 ~ +0.2): confidence alignment with correctness
4. Reasoning bonus (0.0 or 0.1): non-trivial reasoning provided

Total range: 0.0 (format fail) ~ 1.9 (N-class correct, high confidence, good reasoning)
"""

import json
import re


# N-class is hardest (from Phase 1 analysis), C-class second
CLASS_WEIGHTS = {"S": 1.0, "C": 1.2, "N": 1.5}


def extract_json_from_response(text: str) -> dict | None:
    """Extract JSON object from model response."""
    text = text.strip()
    # Try direct JSON parse
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    label_match = re.search(r'"label"\s*:\s*"([SCN])"', text, re.IGNORECASE)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)

    if label_match:
        return {
            "label": label_match.group(1).upper(),
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "reasoning": reason_match.group(1) if reason_match else "",
        }
    return None


def compute_score(data_source: str, solution_str: str, ground_truth: dict, extra_info: dict = None, **kwargs) -> float:
    """Compute reward for a single verification response.

    Args:
        data_source: task identifier (e.g., "fact_verification")
        solution_str: model's raw text output
        ground_truth: {"target": "S"/"C"/"N"}
        extra_info: optional metadata

    Returns:
        float: scalar reward
    """
    gold_label = ground_truth.get("target", "N")

    # === Component 1: Format reward ===
    parsed = extract_json_from_response(solution_str)
    if parsed is None or parsed.get("label") not in ("S", "C", "N"):
        return 0.0  # Invalid format → zero reward

    r_format = 0.1
    pred_label = parsed["label"]
    confidence = min(max(parsed.get("confidence", 0.5), 0.0), 1.0)
    reasoning = parsed.get("reasoning", "")

    # === Component 2: Accuracy reward (main signal) ===
    correct = pred_label == gold_label
    r_accuracy = 1.0 if correct else 0.0

    # Class weighting: harder classes get higher reward
    weight = CLASS_WEIGHTS.get(gold_label, 1.0)
    r_accuracy *= weight

    # === Component 3: Calibration bonus ===
    if correct:
        r_calibration = 0.2 * confidence  # Correct + confident → bonus
    else:
        r_calibration = -0.2 * confidence  # Wrong + confident → penalty

    # === Component 4: Reasoning bonus ===
    r_reasoning = 0.0
    if reasoning and len(reasoning.split()) >= 5:
        r_reasoning = 0.1  # Non-trivial reasoning provided

    total = r_format + r_accuracy + r_calibration + r_reasoning

    return total


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """Batch reward computation (called by veRL's NaiveRewardManager).

    This function signature matches veRL's custom_reward_function interface.
    """
    scores = []
    for ds, sol, gt, ei in zip(data_sources, solution_strs, ground_truths, extra_infos):
        score = compute_score(ds, sol, gt, ei)
        scores.append(score)
    return scores
