"""
Input perturbation functions for robustness experiments.

Perturbation types (A0–A4) — Phase 2 implementation:

- clean
    No change to the example.

- shuffle_snippets
    Randomly shuffles the order of context snippets while preserving content.
    This tests how sensitive the model is to discourse / ordering effects.

The following are currently implemented as safe no-ops (placeholders for Phase 3):

- irrelevant_noise
- contradiction
- lexical_noise

All perturbations are deterministic *given* the global random seed,
which is set upstream in `run_experiment.py`.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Iterable, List

Example = Dict[str, Any]


# -------------------------
# Internal helpers
# -------------------------

def _normalize_snippets(ex: Example) -> List[Any]:
    """
    Get the 'snippets' field as a list, or an empty list if missing/invalid.

    We do NOT change the structure here; we just ensure we always get a list
    to operate on. The meaning of individual snippet entries is preserved.
    """
    snippets = ex.get("snippets")
    if snippets is None:
        return []
    if isinstance(snippets, list):
        return snippets
    # If it's some other unexpected type, wrap it into a list so we don't crash.
    return [snippets]


def _set_snippets(ex: Example, snippets: List[Any]) -> None:
    """
    Assign the snippets list back into the example.
    """
    ex["snippets"] = snippets


# -------------------------
# Core perturbations
# -------------------------

def _perturb_clean(example: Example) -> Example:
    """
    A0: clean

    No change: return the example as-is.
    We do NOT deep-copy here to avoid unnecessary allocations;
    the caller is allowed to treat this as pass-through.
    """
    return example


def _perturb_shuffle_snippets(example: Example) -> Example:
    """
    A3: shuffle_snippets

    Deep-copies the example, shuffles the order of 'snippets' if there
    are at least two, and returns the modified copy.

    If there are 0 or 1 snippets, the example is returned unchanged.
    """
    ex = copy.deepcopy(example)
    snippets = _normalize_snippets(ex)

    if len(snippets) <= 1:
        # Nothing to shuffle
        return ex

    random.shuffle(snippets)
    _set_snippets(ex, snippets)
    return ex


def _perturb_irrelevant_noise(example: Example) -> Example:
    """
    A1: irrelevant_noise (Phase 2 placeholder)

    For now, this is implemented as a no-op to keep behavior predictable
    until a noise corpus / injection strategy is designed in Phase 3.
    """
    # TODO (Phase 3): append or prepend irrelevant sentences / snippets
    # sourced from a separate noise pool.
    return example


def _perturb_contradiction(example: Example) -> Example:
    """
    A2: contradiction (Phase 2 placeholder)

    Currently a no-op. Future implementation may inject snippets that
    explicitly contradict the gold answer.
    """
    # TODO (Phase 3): inject contradictory evidence into the context.
    return example


def _perturb_lexical_noise(example: Example) -> Example:
    """
    A4: lexical_noise (Phase 2 placeholder)

    Currently a no-op. Future implementation may add character-level
    or token-level noise (typos, spacing issues, etc.).
    """
    # TODO (Phase 3): apply token-level noise to the question and/or snippets.
    return example


# -------------------------
# Public API
# -------------------------

def apply_perturbation(example: Example, perturbation: str) -> Example:
    """
    Apply a named perturbation to a single QA example.

    Parameters
    ----------
    example : dict
        A BioASQ-style QA example.
    perturbation : str
        One of:
            - "clean"
            - "shuffle_snippets"
            - "irrelevant_noise"
            - "contradiction"
            - "lexical_noise"

    Returns
    -------
    dict
        The perturbed example. For some perturbations (clean, currently
        the placeholder ones), this may be the same object; for others
        (e.g., shuffle_snippets), this is a deep copy.
    """
    p = perturbation.lower().strip()

    if p == "clean":
        return _perturb_clean(example)

    if p == "shuffle_snippets":
        return _perturb_shuffle_snippets(example)

    if p == "irrelevant_noise":
        return _perturb_irrelevant_noise(example)

    if p == "contradiction":
        return _perturb_contradiction(example)

    if p == "lexical_noise":
        return _perturb_lexical_noise(example)

    # Unknown perturbation type: fall back to clean behavior
    return _perturb_clean(example)


def apply_to_dataset(
    examples: Iterable[Example],
    perturbation: str,
) -> List[Example]:
    """
    Apply a perturbation to an entire dataset, returning a new list of examples.

    This is a convenience wrapper; `run_experiment.py` currently applies
    perturbations per-example, but this function is useful for future batch
    operations or scripts.

    Parameters
    ----------
    examples : iterable of dict
    perturbation : str

    Returns
    -------
    list of dict
        New list of examples (copies may be returned for some perturbations).
    """
    return [apply_perturbation(ex, perturbation) for ex in examples]