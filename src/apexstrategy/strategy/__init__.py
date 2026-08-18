"""Strategy layer: turning projections into a recommendation.

The simulator answers "what happens if this plan runs?". This layer asks the question a
pit wall actually has -- "which plan, and how sure are we?" -- by enumerating the legal
and physically plausible plans, simulating them against **common random draws**, and
ranking them on expected race time (D5) with a regret-based risk column beside it.

``strategy.evaluate`` is the entry point the dashboard calls (Part 10: the UI never
computes). Everything it returns is JSON-serialisable, because the showcase front-end is
fed precomputed scenario grids rather than a live Python backend (D9).
"""

from .candidates import fit_max_stint_share, generate_candidates, refine_candidates
from .engine import StrategyEvaluation, StrategyRanking, evaluate, recommend

__all__ = [
    "StrategyEvaluation",
    "StrategyRanking",
    "evaluate",
    "fit_max_stint_share",
    "generate_candidates",
    "recommend",
    "refine_candidates",
]
