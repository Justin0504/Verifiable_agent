"""Calculator tool for deterministic numerical verification.

Extracts numerical claims and verifies them via computation.
No LLM involved — purely deterministic.
"""

from __future__ import annotations

import ast
import operator
import re

from .base import BaseTool, ToolResult

# Safe operators for eval
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float | None:
    """Safely evaluate a mathematical expression. Returns None on failure."""
    try:
        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree.body)
    except Exception:
        return None


def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op)}")
        return op_fn(_eval_node(node.operand))
    raise ValueError(f"Unsupported node: {type(node)}")


class CalculatorTool(BaseTool):
    """Verify numerical claims by computation.

    Handles:
    - Age calculations: "X was born in 1879, so they were 76 when they died in 1955"
    - Arithmetic: "The total of 234 and 567 is 801"
    - Date differences: "From 1990 to 2003 is 13 years"
    - Percentages: "15 out of 60 is 25%"
    """

    name = "calculator"
    description = "Deterministic numerical verification"
    deterministic = True

    def query(self, claim: str) -> ToolResult:
        """Extract and verify numerical assertions in the claim."""
        checks = []

        # Pattern: "X to/from Y is Z years/days"
        for m in re.finditer(
            r'(?:from\s+)?(\d{3,4})\s+(?:to|until|through)\s+(\d{3,4})\s+(?:is|was|took|lasted)\s+(\d+)\s*years?',
            claim, re.IGNORECASE
        ):
            start, end, claimed = int(m.group(1)), int(m.group(2)), int(m.group(3))
            actual = end - start
            checks.append({
                "expression": f"{end} - {start}",
                "claimed": claimed,
                "actual": actual,
                "correct": claimed == actual,
            })

        # Pattern: "born in YYYY ... aged/age ZZ" or "born in YYYY ... died in YYYY ... was ZZ"
        born_match = re.search(r'born\s+(?:in\s+)?(\d{4})', claim, re.IGNORECASE)
        died_match = re.search(r'died\s+(?:in\s+)?(\d{4})', claim, re.IGNORECASE)
        age_match = re.search(r'(?:aged?|was)\s+(\d{1,3})', claim, re.IGNORECASE)
        if born_match and died_match and age_match:
            born = int(born_match.group(1))
            died = int(died_match.group(1))
            claimed_age = int(age_match.group(1))
            actual_age = died - born
            checks.append({
                "expression": f"{died} - {born}",
                "claimed": claimed_age,
                "actual": actual_age,
                "correct": claimed_age == actual_age or claimed_age == actual_age - 1,
            })

        # Pattern: "total/sum of X and Y is Z"
        for m in re.finditer(
            r'(?:total|sum)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)',
            claim, re.IGNORECASE
        ):
            a, b, claimed = float(m.group(1)), float(m.group(2)), float(m.group(3))
            actual = a + b
            checks.append({
                "expression": f"{a} + {b}",
                "claimed": claimed,
                "actual": actual,
                "correct": abs(claimed - actual) < 0.01,
            })

        # Pattern: "X out of Y is Z%"
        for m in re.finditer(
            r'(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)%',
            claim, re.IGNORECASE
        ):
            part, whole, claimed_pct = float(m.group(1)), float(m.group(2)), float(m.group(3))
            if whole > 0:
                actual_pct = (part / whole) * 100
                checks.append({
                    "expression": f"({part} / {whole}) * 100",
                    "claimed": claimed_pct,
                    "actual": round(actual_pct, 1),
                    "correct": abs(claimed_pct - actual_pct) < 1.0,
                })

        # Pattern: general "X + Y = Z" or "X - Y = Z" style inline math
        for m in re.finditer(
            r'(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)\s*(?:=|equals?|is)\s*(\d+(?:\.\d+)?)',
            claim
        ):
            a, op, b, claimed = m.group(1), m.group(2), m.group(3), m.group(4)
            op_map = {'+': '+', '-': '-', '*': '*', '/': '/', '×': '*', '÷': '/'}
            expr = f"{a} {op_map.get(op, op)} {b}"
            actual = _safe_eval(expr)
            if actual is not None:
                checks.append({
                    "expression": expr,
                    "claimed": float(claimed),
                    "actual": actual,
                    "correct": abs(float(claimed) - actual) < 0.01,
                })

        if not checks:
            return ToolResult(
                tool_name=self.name,
                query=claim,
                evidence="No numerical assertions detected in this claim.",
                success=False,
            )

        all_correct = all(c["correct"] for c in checks)
        evidence_parts = []
        for c in checks:
            status = "CORRECT" if c["correct"] else "INCORRECT"
            evidence_parts.append(
                f"  {c['expression']} = {c['actual']} (claimed: {c['claimed']}) → {status}"
            )

        return ToolResult(
            tool_name=self.name,
            query=claim,
            evidence="Calculator verification:\n" + "\n".join(evidence_parts),
            success=True,
            confidence=1.0,
            raw_data={"checks": checks, "all_correct": all_correct},
        )

    def is_applicable(self, claim: str) -> bool:
        """Check if claim contains numbers that could be verified."""
        numbers = re.findall(r'\b\d+\b', claim)
        return len(numbers) >= 2
