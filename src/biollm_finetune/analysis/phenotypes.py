"""
Phenotype tagging for BioASQ-style biomedical QA questions.

Phase 3 builds on the Phase 2 lightweight phenotypes by:
- making the phenotype schema explicit and stable
- adding dict-based tagging outputs for aggregation
- preserving the Phase 2 list-of-tags API for backward compatibility

Current phenotypes:
- B1: long_question
- B2: long_context
- B7: multi_answer_list

Design goals:
- deterministic and lightweight (no model calls, no external deps)
- robust to BioASQ format variations
- analysis-friendly outputs (boolean phenotype dicts)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

PhenotypeTags = List[str]
PhenotypeDict = Dict[str, bool]
Example = Dict[str, Any]


# -------------------------
# Phenotype schema
# -------------------------

PHENOTYPE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "long_question": {
        "code": "B1",
        "description": "Question length exceeds token threshold.",
        "type": "boolean",
    },
    "long_context": {
        "code": "B2",
        "description": "Total context/snippet text length exceeds character threshold.",
        "type": "boolean",
    },
    "multi_answer_list": {
        "code": "B7",
        "description": "List question with at least N distinct exact answers.",
        "type": "boolean",
    },
}

PHENOTYPE_KEYS: List[str] = list(PHENOTYPE_DEFINITIONS.keys())


# -------------------------
# Tunable thresholds
# -------------------------

# Question is "long" if it has at least this many tokens
LONG_QUESTION_TOKENS: int = 20

# Context is "long" if total snippet text length exceeds this many characters
LONG_CONTEXT_CHARS: int = 800

# List question is "multi-answer" if it has at least this many distinct answers
MULTI_ANSWER_MIN: int = 3


# -------------------------
# Internal helpers
# -------------------------

def get_example_id(ex: Example, idx: int) -> str:
    """
    Return a stable id for an example.

    Prefer:
      - 'id'
      - '_id'
    Fallback:
      - 'idx_{i}'
    """
    raw_id = ex.get("id") or ex.get("_id")
    if raw_id is None:
        return f"idx_{idx}"
    return str(raw_id)


def _get_question_text(ex: Example) -> str:
    """
    Extract the main question text from a BioASQ-style example.

    Prefer fields in this order:
      - 'body'       (BioASQ convention)
      - 'question'   (fallback)
    """
    text = ex.get("body") or ex.get("question") or ""
    if not isinstance(text, str):
        return str(text)
    return text


def _normalize_snippet(snippet: Any) -> str:
    """
    Normalize a single snippet entry to plain text.

    Supports:
      - plain strings
      - dicts with 'text' or 'snippet'
      - other types are converted to str() as a last resort
    """
    if isinstance(snippet, str):
        return snippet

    if isinstance(snippet, dict):
        for key in ("text", "snippet", "document", "context"):
            value = snippet.get(key)
            if isinstance(value, str):
                return value
        return str(snippet)

    return str(snippet)


def _get_context_text(ex: Example) -> str:
    """
    Concatenate all snippet/context text into a single string.

    BioASQ variations:
      - 'snippets': List[dict] with 'text' or 'snippet'
      - 'snippets': List[str]
      - other structures; we normalize defensively.
    """
    snippets = ex.get("snippets") or ex.get("documents") or []
    if not isinstance(snippets, list):
        return ""

    pieces: List[str] = []
    for s in snippets:
        pieces.append(_normalize_snippet(s))

    return " ".join(pieces)


def _flatten_exact_answer(ans: Any) -> List[str]:
    """
    Flatten the BioASQ 'exact_answer' field into a list of strings.

    'exact_answer' can be:
      - a single string
      - a list of strings
      - a list of lists (especially for list questions)
    """
    if ans is None:
        return []

    if isinstance(ans, str):
        return [ans]

    if isinstance(ans, (int, float, bool)):
        return [str(ans)]

    if isinstance(ans, list):
        flat: List[str] = []
        for item in ans:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, (int, float, bool)):
                flat.append(str(item))
            elif isinstance(item, list):
                for sub in item:
                    flat.append(str(sub))
            else:
                flat.append(str(item))
        return flat

    return [str(ans)]


def _get_exact_answers(ex: Example) -> List[str]:
    ans = ex.get("exact_answer") or ex.get("answers") or None
    return _flatten_exact_answer(ans)


def _get_question_type(ex: Example) -> str:
    """
    Extract a normalized question type.

    Expected BioASQ types:
      - 'yesno'
      - 'factoid'
      - 'list'
      - 'summary'
    """
    q_type = ex.get("type") or ex.get("question_type") or ""
    if not isinstance(q_type, str):
        q_type = str(q_type)
    return q_type.lower().strip()


# -------------------------
# Core phenotype logic
# -------------------------

def _is_long_question(ex: Example) -> bool:
    text = _get_question_text(ex)
    tokens = text.split()
    return len(tokens) >= LONG_QUESTION_TOKENS


def _is_long_context(ex: Example) -> bool:
    ctx_text = _get_context_text(ex)
    return len(ctx_text) >= LONG_CONTEXT_CHARS


def _is_multi_answer_list(ex: Example) -> bool:
    q_type = _get_question_type(ex)
    if q_type != "list":
        return False

    answers = _get_exact_answers(ex)
    norm = {a.strip() for a in answers if isinstance(a, str) and a.strip()}
    return len(norm) >= MULTI_ANSWER_MIN


# -------------------------
# Public APIs
# -------------------------

def tag_example(example: Example) -> PhenotypeTags:
    """
    Phase 2-compatible API: returns a list of phenotype tags.
    """
    tags: PhenotypeTags = []

    if _is_long_question(example):
        tags.append("long_question")

    if _is_long_context(example):
        tags.append("long_context")

    if _is_multi_answer_list(example):
        tags.append("multi_answer_list")

    return tags


def tag_example_dict(example: Example) -> PhenotypeDict:
    """
    Phase 3 API: returns a boolean dict over the canonical phenotype keys.

    Example:
      {"long_question": True, "long_context": False, "multi_answer_list": False}
    """
    tag_list = set(tag_example(example))
    return {k: (k in tag_list) for k in PHENOTYPE_KEYS}


def tag_dataset(examples: Sequence[Example]) -> Dict[str, PhenotypeTags]:
    """
    Phase 2-compatible dataset API:
      question_id -> [phenotype_tag, ...]
    """
    labeled: Dict[str, PhenotypeTags] = {}
    for idx, ex in enumerate(examples):
        qid = get_example_id(ex, idx)
        labeled[qid] = tag_example(ex)
    return labeled


def tag_dataset_dict(examples: Sequence[Example]) -> Dict[str, PhenotypeDict]:
    """
    Phase 3 dataset API:
      question_id -> {phenotype_key -> bool}
    """
    labeled: Dict[str, PhenotypeDict] = {}
    for idx, ex in enumerate(examples):
        qid = get_example_id(ex, idx)
        labeled[qid] = tag_example_dict(ex)
    return labeled