#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

# ------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="configs/experiments")
    p.add_argument("--select", action="append", default=[])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------


def get_field(cfg: dict, field: str):
    cur = cfg
    for part in field.split("."):
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def match_selector(cfg: dict, sel: str) -> bool:
    if "!=" in sel:
        field, val = sel.split("!=", 1)
        return get_field(cfg, field) != val

    if "~" in sel:
        field, pat = sel.split("~", 1)
        val = str(get_field(cfg, field) or "")
        return re.search(pat, val) is not None

    if "=" in sel:
        field, val = sel.split("=", 1)
        options = val.split(",")
        return str(get_field(cfg, field)) in options

    raise ValueError(f"Invalid selector: {sel}")


# ------------------------------------------------------------


def main():
    args = parse_args()
    cfg_dir = Path(args.configs)

    yamls = sorted(cfg_dir.glob("*.yaml"))
    selected = []

    for y in yamls:
        cfg = yaml.safe_load(y.read_text())
        if all(match_selector(cfg, s) for s in args.select):
            selected.append(y)

    if not selected:
        print("No experiments matched selectors.")
        return

    print(f"Running {len(selected)} experiments:")
    for y in selected:
        print("  -", y.name)

    for y in selected:
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            str(y),
        ]
        subprocess.check_call(cmd)


# ------------------------------------------------------------

if __name__ == "__main__":
    main()
