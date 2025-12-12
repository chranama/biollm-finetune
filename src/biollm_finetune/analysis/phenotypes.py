"""
Phenotype tagging for BioASQ-style biomedical QA questions.

Phase 2 focuses on simple, length- and structure-based phenotypes:

- B1: long_question
- B2: long_context
- B7: multi_answer_list

These tags are intentionally lightweight and deterministic, intended to:
- provide immediate, interpretable structure for error analysis
- serve as a foundation for more sophisticated phenotypes in later phases
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

PhenotypeTags = List[str]
Example = Dict[str, Any]

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
        # Common BioASQ fields
        for key in ("text", "snippet", "document", "context"):
            value = snippet.get(key)
            if isinstance(value, str):
                return value
        # Fallback: stringify dict
        return str(snippet)

    # Fallback: stringify everything else
    return str(snippet)


def _get_context_text(ex: Example) -> str:
    """
    Concatenate all snippet/context text into a single string.

    BioASQ variations:
      - 'snippets': List[dict] with 'text' or 'snippet'
      - 'snippets': List[str]
      - occasionally other structures; we normalize defensively.
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
                # As a last resort
                flat.append(str(item))
        return flat

    # Any other type: just stringify
    return [str(ans)]


def _get_exact_answers(ex: Example) -> List[str]:
    """
    Extract exact answers from an example, flattened to a simple list of strings.
    """
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
    """
    B1: long_question

    A question is considered long if it has at least LONG_QUESTION_TOKENS tokens.
    """
    text = _get_question_text(ex)
    tokens = text.split()
    return len(tokens) >= LONG_QUESTION_TOKENS


def _is_long_context(ex: Example) -> bool:
    """
    B2: long_context

    Context is long if the concatenated snippet text exceeds LONG_CONTEXT_CHARS characters.
    """
    ctx_text = _get_context_text(ex)
    return len(ctx_text) >= LONG_CONTEXT_CHARS


def _is_multi_answer_list(ex: Example) -> bool:
    """
    B7: multi_answer_list

    A question qualifies if:
      - its type is 'list'
      - it has at least MULTI_ANSWER_MIN distinct exact answers
    """
    q_type = _get_question_type(ex)
    if q_type != "list":
        return False

    answers = _get_exact_answers(ex)
    # Normalize answers (strip whitespace, drop empties)
    norm = {a.strip() for a in answers if isinstance(a, str) and a.strip()}
    return len(norm) >= MULTI_ANSWER_MIN


def tag_example(example: Example) -> PhenotypeTags:
    """
    Assign phenotype tags to a single QA example.

    Current tags (Phase 2):
      - 'long_question'     (B1)
      - 'long_context'      (B2)
      - 'multi_answer_list' (B7)
    """
    tags: PhenotypeTags = []

    if _is_long_question(example):
        tags.append("long_question")

    if _is_long_context(example):
        tags.append("long_context")

    if _is_multi_answer_list(example):
        tags.append("multi_answer_list")

    return tags


def tag_dataset(examples: Sequence[Example]) -> Dict[str, PhenotypeTags]:
    """
    Tag an entire dataset of examples, returning a mapping:

        question_id -> [phenotype_tag, ...]

    If an example has no 'id' (or '_id'), a synthetic id based on its index
    (e.g., 'idx_0', 'idx_1', ...) is used.
    """
    labeled: Dict[str, PhenotypeTags] = {}

    for idx, ex in enumerate(examples):
        raw_id = ex.get("id") or ex.get("_id")
        if raw_id is None:
            qid = f"idx_{idx}"
        else:
            qid = str(raw_id)

        labeled[qid] = tag_example(ex)

    return labeled