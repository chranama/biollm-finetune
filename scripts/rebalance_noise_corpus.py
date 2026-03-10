#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebalance a topic-labeled noise corpus JSONL.")
    p.add_argument("--in", dest="inp", required=True, help="Input JSONL (topic-labeled).")
    p.add_argument("--out", required=True, help="Output JSONL.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap", type=int, default=0, help="Per-topic cap. If 0, uses min topic count.")
    p.add_argument(
        "--upsample",
        action="store_true",
        help="If set and cap > topic_count, sample with replacement to reach cap.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            topic = obj.get("topic", "unknown")
            buckets[str(topic)].append(obj)

    counts = {t: len(v) for t, v in buckets.items()}
    min_count = min(counts.values()) if counts else 0
    cap = args.cap if args.cap > 0 else min_count

    print(
        f"[rebalance] topics={len(buckets)} min_count={min_count} cap={cap} upsample={args.upsample}"
    )

    selected: List[Dict[str, Any]] = []
    for topic, items in buckets.items():
        if len(items) >= cap:
            chosen = random.sample(items, cap)
        else:
            if not args.upsample:
                # if strict, keep all we have (or you could drop the topic)
                chosen = items
            else:
                chosen = list(items)
                while len(chosen) < cap:
                    chosen.append(random.choice(items))
        selected.extend(chosen)

    random.shuffle(selected)

    with out.open("w", encoding="utf-8") as f:
        for obj in selected:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Print final counts
    final_counts: Dict[str, int] = defaultdict(int)
    for obj in selected:
        final_counts[str(obj.get("topic", "unknown"))] += 1

    print("[summary] output snippets by topic:")
    for t in sorted(final_counts, key=final_counts.get, reverse=True):
        print(f"  {t}: {final_counts[t]}")
    print(f"[done] wrote {len(selected)} → {out}")


if __name__ == "__main__":
    main()
