#!/usr/bin/env python3
"""
Validate experiment integrity for the BioLLM robustness suite.

Checks:
- Required files exist per experiment directory
- manifest fields match the config + directory name (seed / perturbation / runtime)
- predictions + inputs lengths are consistent
- clean baseline exists for every (dataset, runtime, seed) used by a perturbed run
- basic "did perturbation actually change inputs?" sanity checks
- optionally spot-check diffs vs clean

Example:
  uv run scripts/validate_experiment_integrity.py \
    --results-root results/experiments \
    --configs-root configs/experiments \
    --spotcheck 3

Exit codes:
- 0: no issues found
- 2: issues found (errors) OR (strict + warnings)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ----------------------------
# Small helpers
# ----------------------------


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def norm_str(x: Any) -> str:
    return str(x).strip().lower()


def count_changed_fields(clean: Dict[str, Any], pert: Dict[str, Any]) -> int:
    """
    Very conservative: only checks the most common fields that perturbations touch.
    """
    changed = 0
    for k in ("body", "question", "snippets"):
        if clean.get(k) != pert.get(k):
            changed += 1
    return changed


# ----------------------------
# Parsing run identifiers
# ----------------------------

# Known runtime tags we expect to appear in run directory names.
# Extend freely as you add runtimes.
KNOWN_RUNTIMES = {
    "mps_fp32",
    "mps_fp32_lora",
    "cpu_fp32",
    "cuda_fp16",
    "cuda_bf16",
    "cuda_fp32",
}


def parse_from_config_expected(
    run_name: str, cfg: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    Use schema-enforced config as the source of truth.

    We *validate* the run_name by checking that it ends with:
        _<runtime>_<perturbation>_seed<seed>
    Then dataset is whatever remains before that suffix (can contain underscores).
    """
    seed = cfg.get("seed", None)
    pert = cfg.get("perturbation", None)
    runtime = safe_get(cfg, "runtime", "name", default=None)
    dataset = safe_get(cfg, "dataset", "name", default=None)

    if not isinstance(seed, int) or not isinstance(pert, str) or not isinstance(runtime, str):
        return (None, None, None, None)

    suffix = f"_{runtime}_{pert}_seed{seed}"
    if not run_name.endswith(suffix):
        # if naming convention changed, still return config-derived values,
        # but dataset parsing from name isn't safe
        return (dataset if isinstance(dataset, str) else None, runtime, pert, seed)

    # dataset prefix in the directory name (can be != cfg dataset name in old dirs)
    dataset_from_name = run_name[: -len(suffix)]
    dataset_from_name = dataset_from_name.strip("_") or None
    return (
        dataset_from_name or (dataset if isinstance(dataset, str) else None),
        runtime,
        pert,
        seed,
    )


def parse_from_name_smart(
    run_name: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    """
    Parse: <dataset>_<runtime>_<perturbation>_seed<seed>
    but allow perturbation to contain underscores (e.g., irrelevant_noise_heavy).

    Strategy:
      1) Require the final token to be seed<INT>
      2) Find a known runtime tag in the remaining string (prefer rightmost match)
      3) Everything between runtime and seed is perturbation (underscores allowed)
      4) Everything before runtime is dataset (underscores allowed)
    """
    if "_seed" not in run_name:
        return (None, None, None, None)

    # Seed
    try:
        seed_str = run_name.split("_")[-1]
        if not seed_str.startswith("seed"):
            return (None, None, None, None)
        seed = int(seed_str.replace("seed", ""))
    except Exception:
        return (None, None, None, None)

    # Strip trailing _seedXX
    base = "_".join(run_name.split("_")[:-1])

    # Find runtime (prefer rightmost occurrence)
    runtime_found = None
    runtime_pos = None
    for rt in sorted(KNOWN_RUNTIMES, key=len, reverse=True):
        needle = f"_{rt}_"
        pos = base.rfind(needle)
        if pos != -1:
            runtime_found = rt
            runtime_pos = pos
            break

    if runtime_found is None or runtime_pos is None:
        # fallback: assume runtime is the last 2 tokens before perturbation
        parts = base.split("_")
        if len(parts) < 3:
            return (None, None, None, seed)
        runtime_found = "_".join(parts[-3:-1])
        dataset = "_".join(parts[:-3]) or None
        perturbation = parts[-1] or None
        return (dataset, runtime_found, perturbation, seed)

    # dataset is before "_<runtime>_"
    dataset = base[:runtime_pos].strip("_") or None

    # remaining after "_<runtime>_"
    after = base[runtime_pos + len(f"_{runtime_found}_") :]
    perturbation = after.strip("_") or None

    return (dataset, runtime_found, perturbation, seed)


def best_parse_run_id(
    run_name: str, cfg_by_name: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int], str]:
    """
    Returns (dataset, runtime, perturbation, seed, source)
    where source is "config" or "name".
    """
    cfg = cfg_by_name.get(run_name)
    if cfg is not None:
        ds, rt, pert, seed = parse_from_config_expected(run_name, cfg)
        if rt and pert and seed is not None:
            return (ds, rt, pert, seed, "config")

    ds, rt, pert, seed = parse_from_name_smart(run_name)
    return (ds, rt, pert, seed, "name")


# ----------------------------
# Data structures
# ----------------------------


@dataclass
class Issue:
    level: str  # "ERROR" | "WARN"
    run: str
    msg: str


# ----------------------------
# Core validator
# ----------------------------

REQUIRED_FILES = (
    "manifest.json",
    "inputs.jsonl",
    "predictions.jsonl",
    "metrics.json",
    "phenotypes.json",
)

WEAK_CHANGE_PERTS = {
    "lexical_noise",
    "lexical_noise_medium",
    "lexical_noise_heavy",
    "contradiction",
    "shuffle_snippets",
}


def validate_run_dir(
    run_dir: Path,
    cfg_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Issue]]:
    issues: List[Issue] = []
    name = run_dir.name

    # Required files
    for fn in REQUIRED_FILES:
        if not (run_dir / fn).exists():
            issues.append(Issue("ERROR", name, f"Missing required file: {fn}"))

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return ({}, issues)

    manifest = read_json(manifest_path)

    # Parse expected values (prefer config; fallback to name)
    ds, rt, pert, seed, src = best_parse_run_id(name, cfg_by_name)

    # Seed check
    man_seed = manifest.get("seed")
    if seed is not None and man_seed is not None:
        try:
            if int(man_seed) != int(seed):
                issues.append(
                    Issue(
                        "ERROR",
                        name,
                        f"Seed mismatch: {src} seed={seed} vs manifest seed={man_seed}",
                    )
                )
        except Exception:
            issues.append(
                Issue(
                    "WARN",
                    name,
                    f"Seed not comparable: {src} seed={seed} vs manifest seed={man_seed}",
                )
            )

    # Perturbation check
    man_pert = manifest.get("perturbation")
    if isinstance(man_pert, str) and isinstance(pert, str):
        if norm_str(man_pert) != norm_str(pert):
            issues.append(
                Issue(
                    "ERROR",
                    name,
                    f"Perturbation mismatch: {src} pert={pert} vs manifest pert={man_pert}",
                )
            )

    # Config cross-check (seed/pert/runtime) if config exists
    cfg = cfg_by_name.get(name)
    if cfg is None:
        # Also attempt matching via manifest["config"] if present
        cfg_path = manifest.get("config")
        if isinstance(cfg_path, str) and cfg_path.strip():
            cfg_stem = Path(cfg_path).stem
            cfg = cfg_by_name.get(cfg_stem)

    if cfg is not None:
        cfg_seed = cfg.get("seed")
        cfg_pert = cfg.get("perturbation")
        cfg_rt = safe_get(cfg, "runtime", "name", default=None)

        if isinstance(cfg_seed, int) and man_seed is not None:
            if int(cfg_seed) != int(man_seed):
                issues.append(
                    Issue(
                        "ERROR",
                        name,
                        f"Seed mismatch: config seed={cfg_seed} vs manifest seed={man_seed}",
                    )
                )
        if isinstance(cfg_pert, str) and isinstance(man_pert, str):
            if norm_str(cfg_pert) != norm_str(man_pert):
                issues.append(
                    Issue(
                        "ERROR",
                        name,
                        f"Perturbation mismatch: config pert={cfg_pert} vs manifest pert={man_pert}",
                    )
                )
        # Runtime isn’t always in manifest; so just validate the directory name parse vs config
        if isinstance(cfg_rt, str) and rt is not None and norm_str(cfg_rt) != norm_str(rt):
            issues.append(
                Issue(
                    "WARN",
                    name,
                    f"Runtime mismatch: parsed runtime={rt} vs config runtime={cfg_rt} (name convention may differ).",
                )
            )
    else:
        issues.append(
            Issue(
                "WARN",
                name,
                "No matching config YAML found; name/manifest checks may be less reliable for older runs.",
            )
        )

    # Inputs/preds length consistency
    inputs_path = run_dir / "inputs.jsonl"
    preds_path = run_dir / "predictions.jsonl"
    if inputs_path.exists() and preds_path.exists():
        inputs = read_jsonl(inputs_path)
        preds = read_jsonl(preds_path)
        if len(inputs) == 0:
            issues.append(Issue("ERROR", name, "inputs.jsonl is empty"))
        if len(preds) == 0:
            issues.append(Issue("ERROR", name, "predictions.jsonl is empty"))
        if len(inputs) != len(preds):
            issues.append(
                Issue("ERROR", name, f"Length mismatch: inputs={len(inputs)} preds={len(preds)}")
            )

    # changed_vs_clean sanity
    n_changed = manifest.get("n_changed_vs_clean")
    n_examples = manifest.get("n_examples")
    man_pert_norm = norm_str(manifest.get("perturbation", "clean"))

    if man_pert_norm == "clean":
        if isinstance(n_changed, int) and n_changed != 0:
            issues.append(
                Issue("WARN", name, f"Clean run has n_changed_vs_clean={n_changed} (expected 0).")
            )
    else:
        if isinstance(n_examples, int) and n_examples > 0 and isinstance(n_changed, int):
            if n_changed == 0:
                issues.append(
                    Issue(
                        "ERROR",
                        name,
                        "Perturbed run reports n_changed_vs_clean=0 (perturbation may not be applied).",
                    )
                )
            elif man_pert_norm not in WEAK_CHANGE_PERTS and n_changed < max(1, n_examples // 10):
                issues.append(
                    Issue(
                        "WARN", name, f"Perturbed run has low change rate: {n_changed}/{n_examples}"
                    )
                )

    # Attach parsed meta for downstream cross-run checks
    manifest["_parsed"] = {
        "dataset": ds,
        "runtime": rt,
        "perturbation": pert,
        "seed": seed,
        "source": src,
    }

    return (manifest, issues)


def validate_clean_baselines(
    manifests: Dict[str, Dict[str, Any]],
) -> List[Issue]:
    issues: List[Issue] = []

    clean_index: Dict[Tuple[str, str, int], str] = {}
    for run_id, m in manifests.items():
        if not m:
            continue
        p = m.get("_parsed") or {}
        pert = norm_str(m.get("perturbation", "")) or norm_str(p.get("perturbation", ""))
        if pert != "clean":
            continue
        ds = norm_str(p.get("dataset", "") or m.get("dataset", ""))
        rt = norm_str(p.get("runtime", ""))
        seed = p.get("seed", m.get("seed", None))
        if isinstance(seed, int) and ds and rt:
            clean_index[(ds, rt, int(seed))] = run_id

    for run_id, m in manifests.items():
        if not m:
            continue
        p = m.get("_parsed") or {}
        pert = norm_str(m.get("perturbation", "")) or norm_str(p.get("perturbation", ""))
        if pert == "clean":
            continue

        ds = norm_str(p.get("dataset", "") or m.get("dataset", ""))
        rt = norm_str(p.get("runtime", ""))
        seed = p.get("seed", m.get("seed", None))

        if not (isinstance(seed, int) and ds and rt):
            issues.append(
                Issue(
                    "WARN",
                    run_id,
                    "Cannot reliably match to a clean baseline (missing dataset/runtime/seed).",
                )
            )
            continue

        key = (ds, rt, int(seed))
        if key not in clean_index:
            issues.append(
                Issue(
                    "ERROR",
                    run_id,
                    f"Missing clean baseline for (dataset={ds}, runtime={rt}, seed={seed})",
                )
            )

    return issues


def spotcheck_diffs(
    results_root: Path,
    manifests: Dict[str, Dict[str, Any]],
    k: int,
) -> List[Issue]:
    issues: List[Issue] = []

    meta: Dict[str, Tuple[str, str, int, str]] = {}
    for run_id, m in manifests.items():
        if not m:
            continue
        p = m.get("_parsed") or {}
        ds = norm_str(p.get("dataset", "") or m.get("dataset", ""))
        rt = norm_str(p.get("runtime", ""))
        pert = norm_str(m.get("perturbation", "") or p.get("perturbation", ""))
        seed = p.get("seed", m.get("seed", None))
        if isinstance(seed, int) and ds and rt and pert:
            meta[run_id] = (ds, rt, int(seed), pert)

    clean_for: Dict[Tuple[str, str, int], str] = {}
    for run_id, (ds, rt, seed, pert) in meta.items():
        if pert == "clean":
            clean_for[(ds, rt, seed)] = run_id

    for run_id, (ds, rt, seed, pert) in meta.items():
        if pert == "clean":
            continue
        clean_id = clean_for.get((ds, rt, seed))
        if not clean_id:
            continue

        clean_inputs_path = results_root / clean_id / "inputs.jsonl"
        pert_inputs_path = results_root / run_id / "inputs.jsonl"
        if not clean_inputs_path.exists() or not pert_inputs_path.exists():
            continue

        clean_rows = read_jsonl(clean_inputs_path)
        pert_rows = read_jsonl(pert_inputs_path)
        n = min(len(clean_rows), len(pert_rows))
        if n == 0:
            continue

        idxs = list(range(min(k, n))) + list(range(max(0, n - k), n))
        idxs = sorted(set(idxs))

        observed_change = any(count_changed_fields(clean_rows[i], pert_rows[i]) > 0 for i in idxs)
        if pert != "clean" and not observed_change:
            issues.append(
                Issue(
                    "WARN",
                    run_id,
                    f"Spotcheck saw no input-field changes vs clean for perturbation='{pert}' "
                    f"(checked indices {idxs}). Double-check perturbation wiring.",
                )
            )

    return issues


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        default="results/experiments",
        help="Root directory containing experiment run dirs.",
    )
    ap.add_argument(
        "--configs-root",
        default="configs/experiments",
        help="Directory containing generated experiment YAMLs.",
    )
    ap.add_argument(
        "--strict", action="store_true", help="Treat WARN as ERROR (exit non-zero on warnings)."
    )
    ap.add_argument(
        "--spotcheck",
        type=int,
        default=0,
        help="If >0, spotcheck k examples at head+tail vs clean.",
    )
    ap.add_argument(
        "--out", default="results/analysis/integrity_report.json", help="Write a JSON report here."
    )
    ap.add_argument(
        "--only-configured",
        action="store_true",
        help="Only validate run dirs that have a matching config YAML.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    configs_root = Path(args.configs_root)
    out_path = Path(args.out)

    if not results_root.exists():
        raise SystemExit(f"[error] results-root not found: {results_root}")

    # Load all configs by name (stem)
    cfg_by_name: Dict[str, Dict[str, Any]] = {}
    if configs_root.exists():
        for p in sorted(list(configs_root.glob("*.y*ml"))):
            try:
                cfg_by_name[p.stem] = read_yaml(p)
            except Exception:
                pass

    # Determine which run dirs to validate
    if args.only_configured and cfg_by_name:
        valid = set(cfg_by_name.keys())
        run_dirs = [p for p in sorted(results_root.iterdir()) if p.is_dir() and p.name in valid]
    else:
        run_dirs = [p for p in sorted(results_root.iterdir()) if p.is_dir()]

    manifests: Dict[str, Dict[str, Any]] = {}
    issues: List[Issue] = []

    for rd in run_dirs:
        m, iss = validate_run_dir(rd, cfg_by_name=cfg_by_name)
        manifests[rd.name] = m
        issues.extend(iss)

    issues.extend(validate_clean_baselines(manifests))

    if args.spotcheck and args.spotcheck > 0:
        issues.extend(spotcheck_diffs(results_root, manifests, k=int(args.spotcheck)))

    n_err = sum(1 for i in issues if i.level == "ERROR")
    n_warn = sum(1 for i in issues if i.level == "WARN")
    status = "ok" if (n_err == 0 and (n_warn == 0 or not args.strict)) else "issues"

    report = {
        "status": status,
        "results_root": str(results_root),
        "configs_root": str(configs_root),
        "n_runs_found": len(run_dirs),
        "n_errors": n_err,
        "n_warnings": n_warn,
        "issues": [{"level": i.level, "run": i.run, "msg": i.msg} for i in issues],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[integrity] runs={len(run_dirs)} errors={n_err} warnings={n_warn} → {out_path}")
    if issues:
        for i in issues[:20]:
            print(f"  [{i.level}] {i.run}: {i.msg}")
        if len(issues) > 20:
            print(f"  ... ({len(issues) - 20} more)")

    if n_err > 0:
        raise SystemExit(2)
    if args.strict and n_warn > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
