"""
Utilities for sampling and structuring error cases for qualitative analysis.
"""

from typing import List, Dict, Any


def select_error_samples(
    predictions: List[Dict[str, Any]],
    k_worst: int = 20,
    k_borderline: int = 20,
    k_random: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Given a list of prediction records (containing gold labels and scores),
    select a few subsets for qualitative inspection.

    Phase 2: you can implement a simple random / naive heuristic and refine later.
    """
    # TODO: implement real logic; for now, return empty buckets
    return {
        "worst": [],
        "borderline": [],
        "random": [],
    }