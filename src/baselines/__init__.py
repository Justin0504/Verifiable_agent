"""Baseline methods for NeurIPS comparison.

Implements 5 baselines:
1. SelfCheckGPT — sampling consistency (no external knowledge)
2. FActScore — atomic fact decomposition + knowledge verification
3. SAFE — search-augmented factual evaluation
4. CoVe — chain-of-verification (LLM self-correction)
5. Retrieve+NLI — traditional retrieve-then-classify pipeline
"""

from .base import BaseBaseline, BaselineResult
from .cove import CoVeBaseline
from .factscore_baseline import FActScoreBaseline
from .retrieve_nli import RetrieveNLIBaseline
from .safe_baseline import SAFEBaseline
from .selfcheck import SelfCheckGPTBaseline

__all__ = [
    "BaseBaseline",
    "BaselineResult",
    "SelfCheckGPTBaseline",
    "FActScoreBaseline",
    "SAFEBaseline",
    "CoVeBaseline",
    "RetrieveNLIBaseline",
]
