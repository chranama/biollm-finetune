"""
Run registry utilities.

This module provides a thin abstraction over the on-disk experiment
structure used by the research suite. It knows how to:

  - enumerate experiment runs under a given root directory
  - load their metadata from run_metadata.json
  - provide lightweight helpers for querying/filtering

Directory layout (as produced by scripts/run_experiment.py):

    results/experiments/
      ├── bioasq_TINY_clean_seed42/
      │   ├── config.yaml
      │   ├── run_metadata.json
      │   ├── metrics.json
      │   ├── phenotype_metrics.json
      │   └── ...
      └── bioasq_TINY_shuffle_seed42/
          ├── config.yaml
          ├── run_metadata.json
          ├── metrics.json
          ├── phenotype_metrics.json
          └── ...

The registry does *not* impose any particular naming convention beyond
what the experiment runner already uses; it simply surfaces what is found.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


DEFAULT_EXPERIMENTS_ROOT = Path("results/experiments")


@dataclass
class ExperimentRun:
    """
    Lightweight handle for a single experiment run on disk.

    Attributes
    ----------
    path : Path
        Filesystem path to the experiment directory (e.g.,
        results/experiments/bioasq_TINY_clean_seed42).
    metadata : dict
        Parsed contents of run_metadata.json, or {} if missing/invalid.
    """

    path: Path
    metadata: Dict[str, Any]


# -------------------------
# Internal helpers
# -------------------------

def _load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """
    Load run_metadata.json from an experiment directory, if it exists.

    Returns {} on any error (missing file, invalid JSON, etc.).
    """
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _is_experiment_dir(path: Path) -> bool:
    """
    Heuristic to decide whether a directory is an experiment run.

    Currently:
      - must be a directory
      - must contain run_metadata.json
    """
    return path.is_dir() and (path / "run_metadata.json").exists()


# -------------------------
# Public API
# -------------------------

def iter_experiments(
    root: Path | str = DEFAULT_EXPERIMENTS_ROOT,
) -> Iterator[ExperimentRun]:
    """
    Iterate over all experiment runs under the given root directory.

    Parameters
    ----------
    root : Path or str, default 'results/experiments'
        Root directory containing experiment subdirectories.

    Yields
    ------
    ExperimentRun
        One object per experiment directory with parsed metadata.
    """
    root_path = Path(root)

    if not root_path.exists():
        return iter(())  # empty iterator

    # Only look at immediate subdirectories (no deep recursion for now)
    for child in sorted(root_path.iterdir()):
        if not _is_experiment_dir(child):
            continue
        meta = _load_run_metadata(child)
        yield ExperimentRun(path=child, metadata=meta)


def list_experiments(
    root: Path | str = DEFAULT_EXPERIMENTS_ROOT,
) -> List[ExperimentRun]:
    """
    Materialize the iterator of experiments into a list.

    Useful for interactive use and testing.
    """
    return list(iter_experiments(root))


def find_experiments_by_prefix(
    prefix: str,
    root: Path | str = DEFAULT_EXPERIMENTS_ROOT,
) -> List[ExperimentRun]:
    """
    Find all experiment runs whose directory name starts with a given prefix.

    This is useful for grouping perturbation runs with their corresponding
    clean baselines, assuming a consistent naming convention such as:

        bioasq_TINY_clean_seed42
        bioasq_TINY_shuffle_seed42

    Parameters
    ----------
    prefix : str
        Directory name prefix to match.
    root : Path or str, default 'results/experiments'

    Returns
    -------
    list[ExperimentRun]
    """
    runs: List[ExperimentRun] = []
    root_path = Path(root)

    if not root_path.exists():
        return runs

    for run in iter_experiments(root_path):
        if run.path.name.startswith(prefix):
            runs.append(run)

    return runs


def load_experiment(
    name: str,
    root: Path | str = DEFAULT_EXPERIMENTS_ROOT,
) -> Optional[ExperimentRun]:
    """
    Load a single experiment by its directory name, if it exists.

    Parameters
    ----------
    name : str
        Directory name of the experiment, e.g. 'bioasq_TINY_clean_seed42'.
    root : Path or str, default 'results/experiments'

    Returns
    -------
    ExperimentRun or None
    """
    root_path = Path(root)
    run_dir = root_path / name
    if not _is_experiment_dir(run_dir):
        return None
    meta = _load_run_metadata(run_dir)
    return ExperimentRun(path=run_dir, metadata=meta)