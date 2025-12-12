"""
Robustness computation for clean vs perturbed experiments.

Given two nested metric dictionaries (one from a clean run, one from a
perturbed run), we compute per-metric robustness summaries:

    {
      "some_metric": {
        "clean": 0.80,
        "perturbed": 0.72,
        "delta": -0.08,
        "relative_change": -0.10
      },
      "nested": {
        "metric": {
          "clean": 0.60,
          "perturbed": 0.66,
          "delta": 0.06,
          "relative_change": 0.10
        }
      }
    }

Non-numeric values are passed through as a simple pair:

    {"clean": <value>, "perturbed": <value>}

Missing metrics on either side are annotated with None.
"""

from __future__ import annotations

from typing import Any, Dict, Union


Number = Union[int, float]
MetricsDict = Dict[str, Any]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _compute_leaf(clean_val: Any, pert_val: Any) -> Dict[str, Any]:
    """
    Compute robustness summary for a single metric leaf.

    If both values are numeric, return:
        {
          "clean": ...,
          "perturbed": ...,
          "delta": perturbed - clean,
          "relative_change": (perturbed - clean) / abs(clean)  # if clean != 0
        }

    Otherwise, just return:
        {
          "clean": clean_val,
          "perturbed": pert_val
        }
    """
    if _is_number(clean_val) and _is_number(pert_val):
        delta = pert_val - clean_val
        if clean_val != 0:
            rel = delta / abs(clean_val)
        else:
            rel = None

        return {
            "clean": clean_val,
            "perturbed": pert_val,
            "delta": delta,
            "relative_change": rel,
        }

    # Non-numeric (or mismatched) values: just record them
    return {
        "clean": clean_val,
        "perturbed": pert_val,
    }


def _compute_node(clean_node: Any, pert_node: Any) -> Any:
    """
    Recursively compute robustness across a pair of metric "nodes".

    The structure rules are:

    - If both nodes are dicts:
        Recurse key-wise, returning a dict with the same keys.
    - Otherwise:
        Treat them as leaf values and call _compute_leaf.
    """
    # Both are dicts: recurse
    if isinstance(clean_node, dict) and isinstance(pert_node, dict):
        keys = set(clean_node.keys()) | set(pert_node.keys())
        out: Dict[str, Any] = {}
        for k in sorted(keys):
            c_val = clean_node.get(k)
            p_val = pert_node.get(k)
            out[k] = _compute_node(c_val, p_val)
        return out

    # At least one is not a dict: treat as leaf
    return _compute_leaf(clean_node, pert_node)


def compute_robustness(
    clean_metrics: MetricsDict,
    perturbed_metrics: MetricsDict,
) -> MetricsDict:
    """
    Compute robustness summaries given metrics from a clean and a perturbed run.

    Parameters
    ----------
    clean_metrics : dict
        Metrics from the clean (unperturbed) experiment, typically the
        contents of metrics.json from a '..._clean_...' run.
    perturbed_metrics : dict
        Metrics from the corresponding perturbed experiment.

    Returns
    -------
    dict
        A nested dictionary mirroring the metric structure, where each
        numeric leaf is replaced by:

            {
              "clean": <number>,
              "perturbed": <number>,
              "delta": <number>,
              "relative_change": <number or None>
            }

        Non-numeric leaves are returned as:

            {
              "clean": <value>,
              "perturbed": <value>
            }
    """
    return _compute_node(clean_metrics, perturbed_metrics)