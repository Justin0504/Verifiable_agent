"""Tests for SEVA v2 process reward function.

Verifies that:
1. Process reward gives partial credit (not binary)
2. Good alignment + wrong label > no alignment + wrong label
3. Perfect response gets maximum reward
4. Garbage input gets zero
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "drzero" / "verl" / "custom_reward"))

from seva_reward import (
    compute_score,
    score_format,
    score_alignment,
    score_chain,
    score_label,
    score_diagnosis,
    extract_json_from_response,
    apply_boundary_bonus,
)


# ============================================================
# Test fixtures
# ============================================================
PERFECT_RESPONSE = json.dumps({
    "evidence_alignment": [
        {"claim_span": "60% of participants", "source_span": "60% of subjects", "status": "match"},
        {"claim_span": "significantly improved", "source_span": "NOT_FOUND", "status": "not_found"},
    ],
    "reasoning_chain": [
        {
            "step": 1,
            "claim_part": "60% of participants",
            "source_evidence": "60% of subjects showed improvement",
            "judgment": "supported",
            "explanation": "The percentage matches between claim and source.",
        },
        {
            "step": 2,
            "claim_part": "significantly improved",
            "source_evidence": "showed improvement",
            "judgment": "not_supported",
            "explanation": "The source says 'improvement' but the claim adds 'significantly' which is not in the source.",
        },
    ],
    "label": "Not Attributable",
    "confidence": 0.85,
    "error_type": "scope_inflation",
    "fix_suggestion": "Change 'significantly improved' to 'improved' to match the source.",
})

WRONG_LABEL_GOOD_CHAIN = json.dumps({
    "evidence_alignment": [
        {"claim_span": "60% of participants", "source_span": "60% of subjects", "status": "match"},
    ],
    "reasoning_chain": [
        {
            "step": 1,
            "claim_part": "percentage",
            "source_evidence": "60% of subjects",
            "judgment": "supported",
            "explanation": "Numbers match between claim and source.",
        },
    ],
    "label": "Attributable",  # WRONG
    "confidence": 0.7,
})

LABEL_ONLY = json.dumps({
    "label": "Not Attributable",
    "confidence": 0.5,
})

GARBAGE = "I don't know, maybe it's attributable?"


def gt(label: str) -> dict:
    return {"target": label}


# ============================================================
# Tests
# ============================================================
def test_perfect_response():
    score = compute_score("attr", PERFECT_RESPONSE, gt("Not Attributable"))
    assert score > 0.8, f"Perfect response should score >0.8, got {score}"


def test_wrong_label_partial_credit():
    score = compute_score("attr", WRONG_LABEL_GOOD_CHAIN, gt("Not Attributable"))
    assert 0.2 < score < 0.7, f"Wrong label + good chain should get partial credit, got {score}"


def test_label_only_low():
    score = compute_score("attr", LABEL_ONLY, gt("Not Attributable"))
    # Has correct label but no alignment/chain → should be moderate
    assert 0.1 < score < 0.5, f"Label-only should score low, got {score}"


def test_garbage_zero():
    score = compute_score("attr", GARBAGE, gt("Not Attributable"))
    assert score == 0.0, f"Garbage should score 0, got {score}"


def test_process_reward_is_smooth():
    """Key test: rewards should NOT be binary. There should be
    meaningful differences between partially-correct responses."""
    s_perfect = compute_score("attr", PERFECT_RESPONSE, gt("Not Attributable"))
    s_partial = compute_score("attr", WRONG_LABEL_GOOD_CHAIN, gt("Not Attributable"))
    s_label = compute_score("attr", LABEL_ONLY, gt("Not Attributable"))
    s_garbage = compute_score("attr", GARBAGE, gt("Not Attributable"))

    assert s_perfect > s_partial > s_label > s_garbage, (
        f"Rewards should be ordered: perfect({s_perfect}) > partial({s_partial}) "
        f"> label_only({s_label}) > garbage({s_garbage})"
    )


def test_score_format():
    assert score_format(None) == 0.0
    assert score_format({"label": "X"}) == 0.2  # has 1 field
    assert score_format({"label": "X", "confidence": 0.5,
                         "evidence_alignment": [], "reasoning_chain": []}) == 1.0


def test_score_alignment():
    parsed = {
        "evidence_alignment": [
            {"claim_span": "hello world", "source_span": "hello there", "status": "match"},
        ]
    }
    s = score_alignment(parsed)
    assert s > 0.5, f"Good alignment entry should score >0.5, got {s}"

    assert score_alignment({}) == 0.0
    assert score_alignment({"evidence_alignment": []}) == 0.0


def test_score_chain():
    parsed = {
        "reasoning_chain": [
            {
                "judgment": "supported",
                "explanation": "The numbers match exactly between claim and source.",
                "source_evidence": "60% of subjects",
                "claim_part": "60% of participants",
            },
            {
                "judgment": "not_supported",
                "explanation": "The word significantly is not in the source document.",
                "source_evidence": "showed improvement",
                "claim_part": "significantly improved",
            },
        ]
    }
    s = score_chain(parsed)
    assert s > 0.6, f"Good chain should score >0.6, got {s}"


def test_score_diagnosis_not_attributable():
    parsed = {
        "error_type": "scope_inflation",
        "fix_suggestion": "Remove the word 'significantly' to match the source.",
    }
    s = score_diagnosis(parsed, "Not Attributable")
    assert s == 1.0, f"Perfect diagnosis should score 1.0, got {s}"


def test_score_diagnosis_attributable():
    # Should NOT have error_type
    s1 = score_diagnosis({}, "Attributable")
    assert s1 == 1.0, f"No error_type for Attributable should score 1.0, got {s1}"

    s2 = score_diagnosis({"error_type": "fabrication"}, "Attributable")
    assert s2 < 1.0, f"Error_type for Attributable should penalize, got {s2}"


def test_boundary_bonus():
    # All same score → boundary = 0, scale = 0.5
    scores = [1.0, 1.0, 1.0, 1.0, 1.0]
    bonused = apply_boundary_bonus(scores, group_size=5)
    assert all(b < s for b, s in zip(bonused, scores)), "All-correct should be attenuated"

    # Mixed scores → boundary > 0
    scores2 = [1.0, 0.5, 1.0, 0.0, 0.5]
    bonused2 = apply_boundary_bonus(scores2, group_size=5)
    # Scale should be closer to 1.0 (boundary is high)
    assert sum(bonused2) > sum(apply_boundary_bonus([1.0]*5, 5)), "Mixed group should have higher total"


def test_calibration_effect():
    """Overconfident wrong answer should score lower than low-confidence wrong answer."""
    overconfident_wrong = json.dumps({
        "evidence_alignment": [{"claim_span": "x", "source_span": "y", "status": "match"}],
        "reasoning_chain": [{"judgment": "supported", "explanation": "test explanation here"}],
        "label": "Attributable",
        "confidence": 0.99,
    })
    underconfident_wrong = json.dumps({
        "evidence_alignment": [{"claim_span": "x", "source_span": "y", "status": "match"}],
        "reasoning_chain": [{"judgment": "supported", "explanation": "test explanation here"}],
        "label": "Attributable",
        "confidence": 0.2,
    })

    s_over = compute_score("attr", overconfident_wrong, gt("Not Attributable"))
    s_under = compute_score("attr", underconfident_wrong, gt("Not Attributable"))

    assert s_under > s_over, (
        f"Low-confidence wrong ({s_under}) should beat overconfident wrong ({s_over})"
    )


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
