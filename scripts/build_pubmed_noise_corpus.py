#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

# Optional dotenv support
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from Bio import Entrez  # biopython

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


DEFAULT_TOPICS = [
    # broad biomedical themes
    "cancer",
    "diabetes",
    "inflammation",
    "cardiovascular",
    "hypertension",
    "randomized controlled trial",
    "systematic review",
    "meta-analysis",
    "gene expression",
    "protein interaction",
    "immunotherapy",
    "antibiotic resistance",
    "pharmacokinetics",
    "adverse effects",
    "biomarker",
    "neurodegeneration",
    "epidemiology",
    "cohort study",
    "clinical guideline",
    "pathway analysis",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a PubMed-derived irrelevant noise snippet corpus (JSONL)."
    )
    p.add_argument(
        "--out", type=str, default="data/noise/irrelevant_snippets.jsonl", help="Output JSONL path."
    )
    p.add_argument("--topics", type=str, default="", help="Optional comma-separated topic queries.")
    p.add_argument("--pmids-per-topic", type=int, default=2000, help="Max PMIDs to pull per topic.")
    p.add_argument("--retmax", type=int, default=200, help="PMIDs per esearch page.")
    p.add_argument("--fetch-batch", type=int, default=200, help="PMIDs per efetch batch.")
    p.add_argument("--min-chars", type=int, default=60, help="Minimum sentence length.")
    p.add_argument("--max-chars", type=int, default=260, help="Maximum sentence length.")
    p.add_argument(
        "--target-snippets", type=int, default=20000, help="Stop after writing this many snippets."
    )
    p.add_argument(
        "--balanced",
        action="store_true",
        help="Enable balanced sampling across topics (round-robin, per-topic quotas).",
    )
    p.add_argument(
        "--per-topic-snippets",
        type=int,
        default=0,
        help="Optional explicit per-topic quota. If 0, uses target_snippets // num_topics.",
    )
    p.add_argument(
        "--max-stall-rounds",
        type=int,
        default=6,
        help="Drop a topic if it yields no new sentences for this many batches.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for topic/selection randomness."
    )
    p.add_argument("--sleep", type=float, default=0.12, help="Sleep between NCBI calls (seconds).")
    return p.parse_args()


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def init_entrez() -> None:
    Entrez.email = require_env("NCBI_EMAIL")
    Entrez.tool = os.getenv("NCBI_TOOL", "biollm-finetune-noise-builder")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def chunked(xs: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield list(xs[i : i + n])


def esearch_pmids(term: str, retmax_total: int, retmax_page: int, sleep_s: float) -> List[str]:
    """
    Search PubMed and return up to retmax_total PMIDs for a query term.
    """
    pmids: List[str] = []
    start = 0

    while start < retmax_total:
        retmax = min(retmax_page, retmax_total - start)
        handle = Entrez.esearch(
            db="pubmed",
            term=term,
            retstart=start,
            retmax=retmax,
            sort="relevance",
        )
        rec = Entrez.read(handle)
        handle.close()

        ids = rec.get("IdList", [])
        if not ids:
            break

        pmids.extend(ids)
        start += len(ids)

        time.sleep(sleep_s)

    # De-dup but preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for pid in pmids:
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def efetch_abstracts(pmids: List[str], sleep_s: float) -> List[str]:
    """
    Fetch abstracts for a list of PMIDs. Returns list of abstract strings.
    """
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    records = Entrez.read(handle)
    handle.close()
    time.sleep(sleep_s)

    abstracts: List[str] = []
    articles = records.get("PubmedArticle", [])
    for art in articles:
        try:
            article = art["MedlineCitation"]["Article"]
            ab = article.get("Abstract", {})
            ab_text = ab.get("AbstractText", [])
            parts: List[str] = []
            if isinstance(ab_text, list):
                for t in ab_text:
                    parts.append(str(t))
            elif ab_text:
                parts.append(str(ab_text))
            text = " ".join(p.strip() for p in parts if p and str(p).strip())
            if text:
                abstracts.append(text)
        except Exception:
            continue

    return abstracts


def split_sentences(text: str) -> List[str]:
    # Simple splitter; good enough for Phase 3
    sents = _SENT_SPLIT_RE.split(text.strip())
    out: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        # remove some common abstract headings that leak into text
        s = re.sub(
            r"^(BACKGROUND|METHODS|RESULTS|CONCLUSION|CONCLUSIONS)\s*:\s*", "", s, flags=re.I
        )
        out.append(s)
    return out


def normalize_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def is_good_sentence(s: str, min_chars: int, max_chars: int) -> bool:
    if len(s) < min_chars or len(s) > max_chars:
        return False
    if s.count(";") > 5:
        return False
    if re.fullmatch(r"[\d\W_]+", s):
        return False
    return True


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    init_entrez()

    out_path = Path(args.out)
    ensure_parent(out_path)

    topics = DEFAULT_TOPICS
    if args.topics.strip():
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]

    if not topics:
        raise RuntimeError("No topics provided.")

    # Shuffle topics so repeated runs vary less by ordering
    random.shuffle(topics)

    written = 0
    seen_norm: Set[str] = set()

    with out_path.open("w", encoding="utf-8") as out_f:
        if not args.balanced:
            # Sequential topics until global target reached
            for topic in topics:
                if written >= args.target_snippets:
                    break

                pmids = esearch_pmids(
                    term=topic,
                    retmax_total=args.pmids_per_topic,
                    retmax_page=args.retmax,
                    sleep_s=args.sleep,
                )
                random.shuffle(pmids)

                for batch in chunked(pmids, args.fetch_batch):
                    if written >= args.target_snippets:
                        break

                    abstracts = efetch_abstracts(batch, sleep_s=args.sleep)
                    for ab in abstracts:
                        for sent in split_sentences(ab):
                            if written >= args.target_snippets:
                                break
                            if not is_good_sentence(sent, args.min_chars, args.max_chars):
                                continue
                            norm = normalize_for_dedup(sent)
                            if norm in seen_norm:
                                continue
                            seen_norm.add(norm)
                            obj = {"text": sent, "source": "pubmed", "topic": topic}
                            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            written += 1

                    if written and written % 2000 == 0:
                        print(f"[progress] wrote {written} snippets (latest topic={topic})")
        else:
            # Balanced: round-robin topics with per-topic quotas + stall detection + spillover fill
            quota = (
                args.per_topic_snippets
                if args.per_topic_snippets > 0
                else max(1, args.target_snippets // len(topics))
            )
            print(
                f"[balanced] topics={len(topics)} target={args.target_snippets} per_topic_quota={quota}"
            )

            written_by_topic: Dict[str, int] = {t: 0 for t in topics}
            stall_rounds: Dict[str, int] = {t: 0 for t in topics}

            pmids_by_topic: Dict[str, List[str]] = {}
            cursor_by_topic: Dict[str, int] = {t: 0 for t in topics}

            def ensure_pmids(topic: str) -> List[str]:
                if topic in pmids_by_topic:
                    return pmids_by_topic[topic]
                pmids = esearch_pmids(
                    term=topic,
                    retmax_total=args.pmids_per_topic,
                    retmax_page=args.retmax,
                    sleep_s=args.sleep,
                )
                random.shuffle(pmids)
                pmids_by_topic[topic] = pmids
                return pmids

            active_topics: List[str] = list(topics)

            # Pass 1: try to fill each topic to quota
            while active_topics and written < args.target_snippets:
                next_active: List[str] = []
                for topic in active_topics:
                    if written >= args.target_snippets:
                        break

                    if written_by_topic[topic] >= quota:
                        continue

                    pmids = ensure_pmids(topic)
                    cur = cursor_by_topic[topic]
                    if cur >= len(pmids):
                        continue

                    batch_pmids = pmids[cur : cur + args.fetch_batch]
                    cursor_by_topic[topic] = cur + len(batch_pmids)

                    before = written
                    abstracts = efetch_abstracts(batch_pmids, sleep_s=args.sleep)
                    for ab in abstracts:
                        for sent in split_sentences(ab):
                            if written >= args.target_snippets or written_by_topic[topic] >= quota:
                                break
                            if not is_good_sentence(sent, args.min_chars, args.max_chars):
                                continue
                            norm = normalize_for_dedup(sent)
                            if norm in seen_norm:
                                continue
                            seen_norm.add(norm)
                            obj = {"text": sent, "source": "pubmed", "topic": topic}
                            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            written += 1
                            written_by_topic[topic] += 1

                    gained = written - before
                    if gained == 0:
                        stall_rounds[topic] += 1
                    else:
                        stall_rounds[topic] = 0

                    if (
                        written_by_topic[topic] < quota
                        and stall_rounds[topic] < args.max_stall_rounds
                        and cursor_by_topic[topic] < len(pmids)
                    ):
                        next_active.append(topic)

                    if written and written % 2000 == 0:
                        print(f"[progress] wrote {written} snippets (latest topic={topic})")

                active_topics = next_active

            # Pass 2: spillover to hit global target (if any topics underfilled)
            if written < args.target_snippets:
                spill_topics = [
                    t for t in topics if cursor_by_topic.get(t, 0) < len(ensure_pmids(t))
                ]
                print(
                    f"[spillover] written={written} remaining_target={args.target_snippets - written} spill_topics={len(spill_topics)}"
                )

                while spill_topics and written < args.target_snippets:
                    next_spill: List[str] = []
                    for topic in spill_topics:
                        if written >= args.target_snippets:
                            break

                        pmids = ensure_pmids(topic)
                        cur = cursor_by_topic[topic]
                        if cur >= len(pmids):
                            continue

                        batch_pmids = pmids[cur : cur + args.fetch_batch]
                        cursor_by_topic[topic] = cur + len(batch_pmids)

                        before = written
                        abstracts = efetch_abstracts(batch_pmids, sleep_s=args.sleep)
                        for ab in abstracts:
                            for sent in split_sentences(ab):
                                if written >= args.target_snippets:
                                    break
                                if not is_good_sentence(sent, args.min_chars, args.max_chars):
                                    continue
                                norm = normalize_for_dedup(sent)
                                if norm in seen_norm:
                                    continue
                                seen_norm.add(norm)
                                obj = {"text": sent, "source": "pubmed", "topic": topic}
                                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                written += 1
                                written_by_topic[topic] += 1

                        gained = written - before
                        if gained == 0:
                            stall_rounds[topic] += 1
                        else:
                            stall_rounds[topic] = 0

                        if stall_rounds[topic] < args.max_stall_rounds and cursor_by_topic[
                            topic
                        ] < len(pmids):
                            next_spill.append(topic)

                        if written and written % 2000 == 0:
                            print(f"[progress] wrote {written} snippets (latest topic={topic})")

                    spill_topics = next_spill

            print("[summary] snippets by topic:")
            for t in sorted(written_by_topic, key=written_by_topic.get, reverse=True):
                print(f"  {t}: {written_by_topic[t]}")

    print(f"[done] wrote {written} snippets → {out_path}")


if __name__ == "__main__":
    main()
