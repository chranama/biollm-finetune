#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from biollm_finetune.training.adapter_manifest import write_adapter_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a lightweight PEFT adapter manifest for review artifacts."
    )
    parser.add_argument(
        "--adapter-dir",
        default="results/ckpts/tiny_adapter",
        help="Generated PEFT adapter directory.",
    )
    parser.add_argument(
        "--out",
        default="results/phase4/peft/tiny_adapter_manifest.json",
        help="Reviewer-facing manifest path.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional fine-tuning config path. Defaults to adapter run.json config_path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = write_adapter_manifest(
        adapter_dir=Path(args.adapter_dir),
        out_path=Path(args.out),
        config_path=args.config,
    )
    print(f"[peft] wrote {out}")


if __name__ == "__main__":
    main()
