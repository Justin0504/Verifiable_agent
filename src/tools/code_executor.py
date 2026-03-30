"""Code execution tool for verifying logic, math, and computational claims.

Executes Python code in a restricted subprocess to verify claims
that require computation beyond simple arithmetic.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .base import BaseTool, ToolResult


class CodeExecutorTool(BaseTool):
    """Verify claims by generating and executing Python code.

    Handles:
    - Complex mathematical expressions
    - Date/time calculations
    - Unit conversions
    - Logical reasoning that can be expressed as code
    - Statistical claims (averages, percentages, etc.)
    """

    name = "code_executor"
    description = "Execute Python code for computational verification"
    deterministic = True  # Code execution is deterministic

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def query(self, claim: str) -> ToolResult:
        """Generate and execute verification code for the claim."""
        code = self._generate_verification_code(claim)
        if not code:
            return ToolResult(
                tool_name=self.name, query=claim,
                evidence="Could not generate verification code for this claim.",
                success=False,
            )

        result = self._execute_code(code)
        if result is None:
            return ToolResult(
                tool_name=self.name, query=claim,
                evidence="Code execution failed or timed out.",
                success=False,
            )

        return ToolResult(
            tool_name=self.name,
            query=claim,
            evidence=f"Code execution result:\n{result}",
            success=True,
            confidence=1.0,
            raw_data={"code": code, "output": result},
        )

    def _generate_verification_code(self, claim: str) -> str | None:
        """Generate Python code to verify the claim.

        Pattern-based code generation for common claim types.
        """
        # Date difference calculations
        m = re.search(
            r'(?:from|between|in)\s+(\d{4})\s+(?:to|and|until)\s+(\d{4})',
            claim, re.IGNORECASE,
        )
        if m:
            y1, y2 = m.group(1), m.group(2)
            numbers = re.findall(r'\b(\d+)\s*years?\b', claim)
            if numbers:
                claimed = numbers[0]
                return (
                    f"actual = {y2} - {y1}\n"
                    f"claimed = {claimed}\n"
                    f"print(f'Difference: {{actual}} years')\n"
                    f"print(f'Claimed: {{claimed}} years')\n"
                    f"print(f'CORRECT' if actual == claimed else f'INCORRECT: actual is {{actual}}')"
                )

        # Percentage calculations
        m = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:out of|of|from)\s*(\d+(?:\.\d+)?)\s*(?:is|equals?|=)\s*(\d+(?:\.\d+)?)\s*%',
            claim, re.IGNORECASE,
        )
        if m:
            part, whole, pct = m.group(1), m.group(2), m.group(3)
            return (
                f"actual_pct = ({part} / {whole}) * 100\n"
                f"claimed_pct = {pct}\n"
                f"print(f'Actual: {{actual_pct:.2f}}%')\n"
                f"print(f'Claimed: {{claimed_pct}}%')\n"
                f"print(f'CORRECT' if abs(actual_pct - claimed_pct) < 1 else f'INCORRECT: actual is {{actual_pct:.2f}}%')"
            )

        # Speed/distance/time calculations
        m = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:km|miles?|meters?|m)\s*(?:in|over)\s*(\d+(?:\.\d+)?)\s*(?:hours?|minutes?|seconds?|h|min|s)',
            claim, re.IGNORECASE,
        )
        if m:
            distance, time_val = m.group(1), m.group(2)
            speed_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:km/h|mph|m/s)', claim)
            if speed_match:
                claimed_speed = speed_match.group(1)
                return (
                    f"distance = {distance}\n"
                    f"time = {time_val}\n"
                    f"actual_speed = distance / time\n"
                    f"claimed_speed = {claimed_speed}\n"
                    f"print(f'Actual speed: {{actual_speed:.2f}}')\n"
                    f"print(f'Claimed speed: {{claimed_speed}}')\n"
                    f"print(f'CORRECT' if abs(actual_speed - claimed_speed) < 0.5 else f'INCORRECT')"
                )

        # General arithmetic: "X op Y = Z"
        m = re.search(
            r'(\d+(?:\.\d+)?)\s*([+\-*/×÷^])\s*(\d+(?:\.\d+)?)\s*(?:=|equals?|is)\s*(\d+(?:\.\d+)?)',
            claim,
        )
        if m:
            a, op, b, claimed = m.group(1), m.group(2), m.group(3), m.group(4)
            op_map = {'+': '+', '-': '-', '*': '*', '/': '/', '×': '*', '÷': '/', '^': '**'}
            py_op = op_map.get(op, op)
            return (
                f"actual = {a} {py_op} {b}\n"
                f"claimed = {claimed}\n"
                f"print(f'{{a}} {py_op} {{b}} = {{actual}}')\n"
                f"print(f'Claimed: {{claimed}}')\n"
                f"print(f'CORRECT' if abs(actual - float(claimed)) < 0.01 else f'INCORRECT')"
            )

        return None

    def _execute_code(self, code: str) -> str | None:
        """Execute Python code in a restricted subprocess."""
        # Safety: only allow basic math operations
        forbidden = ["import os", "import sys", "subprocess", "open(", "__import__",
                     "exec(", "eval(", "compile(", "globals", "locals"]
        for f in forbidden:
            if f in code:
                return None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
            Path(tmp_path).unlink(missing_ok=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, Exception):
            Path(tmp_path).unlink(missing_ok=True)
            return None

    def is_applicable(self, claim: str) -> bool:
        """Check if claim contains computational elements."""
        patterns = [
            r'\d+\s*[+\-*/×÷^]\s*\d+',          # arithmetic
            r'\d+\s*%',                            # percentage
            r'from\s+\d{4}\s+to\s+\d{4}',        # date range
            r'\d+\s*(?:km|miles?|meters?)\s*(?:in|over)', # speed
        ]
        return any(re.search(p, claim, re.IGNORECASE) for p in patterns)
