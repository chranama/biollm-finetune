#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "proof" / "evidence_manifest.latest.json"

REQUIRED_TOP = [
    "proof_id",
    "run_id",
    "generated_at",
    "repo_commit",
    "status",
    "claims",
    "diagnostics",
]
REQUIRED_CLAIM = ["claim_text", "verification_command", "artifact_paths", "expected_signal"]
EXPECTED_SEEDS = {42, 97, 13}


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def main() -> None:
    if not MANIFEST.exists():
        fail(f"missing manifest: {MANIFEST}")

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key in REQUIRED_TOP:
        if key not in data:
            fail(f"missing top-level key: {key}")

    if data["status"] not in {"pass", "fail"}:
        fail("status must be pass|fail")

    claims = data["claims"]
    if not isinstance(claims, list) or not claims:
        fail("claims must be non-empty list")

    for idx, claim in enumerate(claims, start=1):
        for key in REQUIRED_CLAIM:
            if key not in claim:
                fail(f"claim[{idx}] missing key: {key}")
        for raw in claim["artifact_paths"]:
            p = ROOT / raw
            if not p.exists():
                fail(f"claim[{idx}] missing artifact path: {raw}")

    diagnostics = data.get("diagnostics", {})
    seeds = diagnostics.get("seed_config", [])
    if set(seeds) != EXPECTED_SEEDS:
        fail(f"seed_config must be {sorted(EXPECTED_SEEDS)}, got {seeds}")

    cfg_prefix = diagnostics.get("canonical_configs_prefix", "")
    if not cfg_prefix:
        fail("diagnostics.canonical_configs_prefix is required")

    print("OK: manifest schema-lite checks, artifact paths, and reproducibility metadata validated")


if __name__ == "__main__":
    main()
