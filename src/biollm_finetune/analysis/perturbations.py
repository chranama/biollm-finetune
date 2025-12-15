"""
Perturbations for robustness experiments.

Implements:
- clean
- shuffle_snippets
- irrelevant_noise (uses a local balanced PubMed-derived corpus JSONL)
- irrelevant_noise_heavy
- lexical_noise (light/medium/heavy, taxonomy-driven)
- contradiction (yes/no focused, template-based)

This module is designed to be deterministic given upstream seeding of `random`.
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

Example = Dict[str, Any]

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_NOISE_CORPUS = Path("data/noise/irrelevant_snippets_balanced.jsonl")
DEFAULT_NOISE_FALLBACK = Path("data/noise/irrelevant_snippets.jsonl")

_NOISE_POOL: Optional[List[Dict[str, Any]]] = None
_NOISE_POOL_PATH: Optional[Path] = None

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")

# Tokens we consider "safe to drop" for token-deletion noise
SAFE_DROP_TOKENS = {
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "and", "or", "but",
}

# Simple vowel substitution map (very conservative)
VOWELS = "aeiou"
VOWEL_SUBS = {v: [x for x in VOWELS if x != v] for v in VOWELS}

# Small keyboard-adjacent-ish map (minimal, conservative)
KEY_NEIGHBORS = {
    "a": ["s", "q", "w", "z"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "e": ["w", "s", "d", "r"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["u", "j", "k", "o"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k"],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "k", "l", "p"],
    "p": ["o", "l"],
    "q": ["w", "a"],
    "r": ["e", "d", "f", "t"],
    "s": ["a", "w", "e", "d", "x", "z"],
    "t": ["r", "f", "g", "y"],
    "u": ["y", "h", "j", "i"],
    "v": ["c", "f", "g", "b"],
    "w": ["q", "a", "s", "e"],
    "x": ["z", "s", "d", "c"],
    "y": ["t", "g", "h", "u"],
    "z": ["a", "s", "x"],
}

# ---------------------------------------------------------------------
# Public API (single perturbation)
# ---------------------------------------------------------------------

def apply_perturbation(example: Example, perturbation: str, config: Optional[Dict[str, Any]] = None) -> Example:
    """
    Apply a single perturbation by name.

    Supported:
      - clean
      - shuffle_snippets
      - irrelevant_noise
      - irrelevant_noise_heavy
      - lexical_noise
      - lexical_noise_medium
      - lexical_noise_heavy
      - contradiction
      - contradiction_prepend (optional alias)
    """
    cfg = config or {}
    p = perturbation.lower().strip()

    if p == "clean":
        return example

    if p == "shuffle_snippets":
        return _perturb_shuffle_snippets(example)

    if p == "irrelevant_noise":
        return _perturb_irrelevant_noise(example, cfg, heavy=False)

    if p == "irrelevant_noise_heavy":
        return _perturb_irrelevant_noise(example, cfg, heavy=True)

    if p in {"lexical_noise", "lexical_noise_light"}:
        return _perturb_lexical_noise(example, cfg, budget="low")

    if p == "lexical_noise_medium":
        return _perturb_lexical_noise(example, cfg, budget="medium")

    if p == "lexical_noise_heavy":
        return _perturb_lexical_noise(example, cfg, budget="high")

    if p == "contradiction":
        return _perturb_contradiction(example, cfg, position="append")

    if p == "contradiction_prepend":
        return _perturb_contradiction(example, cfg, position="prepend")

    # Unknown => no-op (safer in large experiment grids)
    return example


def apply_to_dataset(examples: Iterable[Example], perturbation: str, config: Optional[Dict[str, Any]] = None) -> List[Example]:
    return [apply_perturbation(ex, perturbation, config=config) for ex in examples]


# ---------------------------------------------------------------------
# Helpers: question/snippets access
# ---------------------------------------------------------------------

def _normalize_snippets(ex: Example) -> List[Any]:
    s = ex.get("snippets")
    if s is None:
        return []
    if isinstance(s, list):
        return s
    return [s]


def _set_snippets(ex: Example, snippets: List[Any]) -> None:
    ex["snippets"] = snippets


def _get_question_text(ex: Example) -> str:
    text = ex.get("body") or ex.get("question") or ""
    if not isinstance(text, str):
        return str(text)
    return text


def _set_question_text(ex: Example, text: str) -> None:
    if "body" in ex:
        ex["body"] = text
    elif "question" in ex:
        ex["question"] = text
    else:
        ex["body"] = text


def _get_question_type(ex: Example) -> str:
    qt = ex.get("type") or ex.get("question_type") or ""
    if not isinstance(qt, str):
        qt = str(qt)
    return qt.lower().strip()


def _flatten_exact_answer(ans: Any) -> List[str]:
    if ans is None:
        return []
    if isinstance(ans, str):
        return [ans]
    if isinstance(ans, (int, float, bool)):
        return [str(ans)]
    if isinstance(ans, list):
        out: List[str] = []
        for it in ans:
            if isinstance(it, list):
                out.extend([str(x) for x in it])
            else:
                out.append(str(it))
        return out
    return [str(ans)]


# ---------------------------------------------------------------------
# Shuffle snippets
# ---------------------------------------------------------------------

def _perturb_shuffle_snippets(example: Example) -> Example:
    ex = copy.deepcopy(example)
    snippets = _normalize_snippets(ex)
    if len(snippets) > 1:
        random.shuffle(snippets)
        _set_snippets(ex, snippets)
    return ex


# ---------------------------------------------------------------------
# Irrelevant noise (balanced corpus)
# ---------------------------------------------------------------------

def _load_noise_pool(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Loads JSONL lines; each line must contain at least {"text": "..."}.
    Keeps optional fields like "topic"/"source".
    """
    global _NOISE_POOL, _NOISE_POOL_PATH

    chosen = path or DEFAULT_NOISE_CORPUS
    if not chosen.exists() and DEFAULT_NOISE_FALLBACK.exists():
        chosen = DEFAULT_NOISE_FALLBACK

    if _NOISE_POOL is not None and _NOISE_POOL_PATH == chosen:
        return _NOISE_POOL

    pool: List[Dict[str, Any]] = []
    if not chosen.exists():
        _NOISE_POOL = pool
        _NOISE_POOL_PATH = chosen
        return pool

    with chosen.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"text": line}
            if isinstance(obj, str):
                obj = {"text": obj}
            if isinstance(obj, dict):
                text = obj.get("text")
                if isinstance(text, str) and text.strip():
                    pool.append(obj)
    _NOISE_POOL = pool
    _NOISE_POOL_PATH = chosen
    return pool


def _sample_noise_snippets(n: int, corpus_path: Optional[str] = None) -> List[Dict[str, Any]]:
    pool = _load_noise_pool(Path(corpus_path) if corpus_path else None)
    if not pool or n <= 0:
        return []
    # sample with replacement
    return [random.choice(pool) for _ in range(n)]


def _perturb_irrelevant_noise(example: Example, cfg: Dict[str, Any], heavy: bool) -> Example:
    """
    Injects irrelevant (in-domain) biomedical snippets from a balanced corpus.

    Config keys (optional):
      - noise_corpus_path: str
      - noise_n: int (overrides range sampling)
      - noise_position: "append"|"prepend"|"random"
      - noise_snippet_format: "dict"|"text" (default dict)
        * dict => {"text": "...", "source": "noise", "topic": "..."}
        * text => "..."
    """
    ex = copy.deepcopy(example)
    snippets = _normalize_snippets(ex)

    n_override = cfg.get("noise_n")
    if isinstance(n_override, int) and n_override >= 0:
        n = n_override
    else:
        n = random.randint(3, 4) if heavy else random.randint(1, 2)

    corpus_path = cfg.get("noise_corpus_path")
    sampled = _sample_noise_snippets(n, corpus_path=corpus_path)
    if not sampled:
        return ex

    fmt = (cfg.get("noise_snippet_format") or "dict").lower()
    noise_items: List[Any] = []
    for obj in sampled:
        text = obj.get("text", "")
        if fmt == "text":
            noise_items.append(str(text))
        else:
            noise_items.append(
                {
                    "text": str(text),
                    "source": "noise",
                    "topic": obj.get("topic"),
                    "noise_source": obj.get("source", "pubmed"),
                }
            )

    pos = (cfg.get("noise_position") or "append").lower()
    if pos == "prepend":
        snippets = noise_items + snippets
    elif pos == "random":
        # insert each item at a random position
        for item in noise_items:
            idx = random.randint(0, len(snippets))
            snippets.insert(idx, item)
    else:
        snippets = snippets + noise_items

    _set_snippets(ex, snippets)
    return ex


# ---------------------------------------------------------------------
# Lexical noise (taxonomy-driven)
# ---------------------------------------------------------------------

def _is_protected_token(tok: str) -> bool:
    """
    Protect tokens that are likely biomedical identifiers or numeric forms:
      - contains digits (IL6, p53, 5-HT)
      - all caps short tokens (DNA, RNA, TNF)
      - greek letters common in biomedical text (α, β, γ)
    """
    if any(ch.isdigit() for ch in tok):
        return True
    if tok.isupper() and 2 <= len(tok) <= 6:
        return True
    if any(ch in tok for ch in ("α", "β", "γ", "δ")):
        return True
    return False


def _tokenize(text: str) -> List[str]:
    # keep punctuation as separate tokens for safer swaps
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _detokenize(tokens: List[str]) -> str:
    # simple detokenizer: spaces between alnum tokens; no space before punctuation
    out: List[str] = []
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t)
            continue
        prev = tokens[i - 1]
        if re.fullmatch(r"[^\w\s]", t):
            # punctuation attaches
            out.append(t)
        elif re.fullmatch(r"[^\w\s]", prev):
            out.append(t if prev in ("(", "[", "{") else " " + t)
        else:
            out.append(" " + t)
    return "".join(out)


def _op_case_flip(tokens: List[str]) -> List[str]:
    # Flip case of a random character in a random non-protected word token
    idxs = [i for i, t in enumerate(tokens) if re.fullmatch(r"\w+", t) and not _is_protected_token(t) and any(ch.isalpha() for ch in t)]
    if not idxs:
        return tokens
    i = random.choice(idxs)
    t = list(tokens[i])
    alpha_pos = [j for j, ch in enumerate(t) if ch.isalpha()]
    if not alpha_pos:
        return tokens
    j = random.choice(alpha_pos)
    t[j] = t[j].swapcase()
    tokens[i] = "".join(t)
    return tokens


def _op_char_delete(tokens: List[str]) -> List[str]:
    idxs = [i for i, t in enumerate(tokens) if re.fullmatch(r"\w+", t) and not _is_protected_token(t) and len(t) >= 5]
    if not idxs:
        return tokens
    i = random.choice(idxs)
    t = tokens[i]
    # don't delete first char
    pos = random.randint(1, len(t) - 2)
    tokens[i] = t[:pos] + t[pos + 1 :]
    return tokens


def _op_char_duplicate(tokens: List[str]) -> List[str]:
    idxs = [i for i, t in enumerate(tokens) if re.fullmatch(r"\w+", t) and not _is_protected_token(t) and len(t) >= 4]
    if not idxs:
        return tokens
    i = random.choice(idxs)
    t = tokens[i]
    pos = random.randint(1, len(t) - 2)
    tokens[i] = t[:pos] + t[pos] + t[pos:]
    return tokens


def _op_char_substitute(tokens: List[str]) -> List[str]:
    idxs = [i for i, t in enumerate(tokens) if re.fullmatch(r"\w+", t) and not _is_protected_token(t) and any(ch.isalpha() for ch in t)]
    if not idxs:
        return tokens
    i = random.choice(idxs)
    t = list(tokens[i].lower())
    alpha_pos = [j for j, ch in enumerate(t) if ch.isalpha()]
    if not alpha_pos:
        return tokens
    j = random.choice(alpha_pos)
    ch = t[j]
    # vowel substitution or keyboard neighbor
    if ch in VOWEL_SUBS and random.random() < 0.6:
        t[j] = random.choice(VOWEL_SUBS[ch])
    else:
        neigh = KEY_NEIGHBORS.get(ch)
        if neigh:
            t[j] = random.choice(neigh)
        else:
            # fallback: minor vowel swap if possible
            if ch in VOWEL_SUBS:
                t[j] = random.choice(VOWEL_SUBS[ch])
    tokens[i] = "".join(t)
    return tokens


def _op_token_duplicate(tokens: List[str]) -> List[str]:
    idxs = [i for i, t in enumerate(tokens) if re.fullmatch(r"\w+", t) and not _is_protected_token(t) and len(t) >= 3]
    if not idxs:
        return tokens
    i = random.choice(idxs)
    tokens.insert(i, tokens[i])
    return tokens


def _op_token_delete(tokens: List[str]) -> List[str]:
    idxs = [i for i, t in enumerate(tokens) if t.lower() in SAFE_DROP_TOKENS]
    if not idxs:
        return tokens
    i = random.choice(idxs)
    del tokens[i]
    return tokens


def _op_token_swap(tokens: List[str]) -> List[str]:
    # swap two adjacent word tokens; don't cross punctuation
    candidates = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if re.fullmatch(r"\w+", a) and re.fullmatch(r"\w+", b) and not _is_protected_token(a) and not _is_protected_token(b):
            candidates.append(i)
    if not candidates:
        return tokens
    i = random.choice(candidates)
    tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    return tokens


def _op_whitespace_noise(text: str) -> str:
    # either add an extra space or remove one space
    if " " not in text:
        return text
    if random.random() < 0.5:
        # add double space somewhere
        idx = random.randint(0, len(text) - 2)
        return text[:idx] + "  " + text[idx:]
    # remove one space occurrence
    space_positions = [m.start() for m in re.finditer(r"\s", text)]
    if not space_positions:
        return text
    idx = random.choice(space_positions)
    return text[:idx] + text[idx + 1:]


def _op_punct_noise(tokens: List[str]) -> List[str]:
    # remove or duplicate commas/periods; keep final question mark if present
    punct_idxs = [i for i, t in enumerate(tokens) if t in {",", "."}]
    if not punct_idxs:
        return tokens
    i = random.choice(punct_idxs)
    if random.random() < 0.5:
        # delete punctuation
        del tokens[i]
    else:
        # duplicate punctuation
        tokens.insert(i, tokens[i])
    return tokens


LEX_OPS = {
    "case_flip": _op_case_flip,
    "char_delete": _op_char_delete,
    "char_duplicate": _op_char_duplicate,
    "char_substitute": _op_char_substitute,
    "token_duplicate": _op_token_duplicate,
    "token_delete": _op_token_delete,
    "token_swap": _op_token_swap,
    "punct_noise": _op_punct_noise,
}

# weights by "safety" (higher = more likely)
LEX_WEIGHTS = {
    "case_flip": 3.0,
    "char_substitute": 2.5,
    "char_duplicate": 2.0,
    "char_delete": 2.0,
    "token_swap": 1.2,
    "token_duplicate": 1.0,
    "token_delete": 0.7,
    "punct_noise": 0.8,
}


def _choose_ops(budget: str) -> List[str]:
    # Budget defines how many ops; we avoid repeated ops by default.
    if budget == "low":
        k = 1
    elif budget == "high":
        k = random.randint(4, 6)
    else:
        k = random.randint(2, 3)

    ops = list(LEX_OPS.keys())
    weights = [LEX_WEIGHTS.get(o, 1.0) for o in ops]

    chosen: List[str] = []
    # sample without replacement using weighted picks
    available = ops[:]
    avail_w = weights[:]
    for _ in range(min(k, len(available))):
        # weighted choice
        total = sum(avail_w)
        r = random.random() * total
        acc = 0.0
        pick_idx = 0
        for i, w in enumerate(avail_w):
            acc += w
            if acc >= r:
                pick_idx = i
                break
        chosen.append(available[pick_idx])
        # remove
        del available[pick_idx]
        del avail_w[pick_idx]
    return chosen


def _perturb_lexical_noise(example: Example, cfg: Dict[str, Any], budget: str) -> Example:
    """
    Applies taxonomy-driven lexical noise primarily to the question text.

    Config keys (optional):
      - lexical_target: "question"|"snippets_first"|"snippets_all"  (default "question")
      - lexical_budget: "low"|"medium"|"high" (overrides `budget` arg)
      - lexical_ops: list[str] (explicit ops to apply instead of sampling)
      - lexical_include_whitespace: bool (default: medium/high True)
      - lexical_include_punct: bool (default: medium/high True)
    """
    ex = copy.deepcopy(example)

    target = (cfg.get("lexical_target") or "question").lower()
    budget = (cfg.get("lexical_budget") or budget).lower()

    ops_override = cfg.get("lexical_ops")
    if isinstance(ops_override, list) and all(isinstance(x, str) for x in ops_override):
        ops = [x.strip().lower() for x in ops_override if x.strip()]
    else:
        ops = _choose_ops(budget)

    include_ws = cfg.get("lexical_include_whitespace")
    include_punct = cfg.get("lexical_include_punct")
    if include_ws is None:
        include_ws = budget in {"medium", "high"}
    if include_punct is None:
        include_punct = budget in {"medium", "high"}

    def apply_to_text(text: str) -> Tuple[str, List[str]]:
        if not text:
            return text, []
        tokens = _tokenize(text)
        applied: List[str] = []

        for op in ops:
            if op == "punct_noise" and not include_punct:
                continue
            fn = LEX_OPS.get(op)
            if fn is None:
                continue
            try:
                tokens = fn(tokens)
                applied.append(op)
            except Exception:
                continue

        new_text = _detokenize(tokens)
        if include_ws and random.random() < 0.5:
            new_text = _op_whitespace_noise(new_text)
            applied.append("whitespace_noise")

        # keep trailing '?' if original had one
        if text.strip().endswith("?") and not new_text.strip().endswith("?"):
            new_text = new_text.rstrip() + "?"
        new_text = _WS_RE.sub(" ", new_text).strip()
        return new_text, applied

    if target == "question":
        q = _get_question_text(ex)
        new_q, applied = apply_to_text(q)
        _set_question_text(ex, new_q)
        ex.setdefault("meta", {})
        ex["meta"]["lexical_noise"] = {"budget": budget, "ops": applied}
        return ex

    snippets = _normalize_snippets(ex)
    if not snippets:
        return ex

    def apply_to_snippet_item(item: Any) -> Any:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            txt, applied = apply_to_text(item["text"])
            new_item = dict(item)
            new_item["text"] = txt
            new_item.setdefault("meta", {})
            new_item["meta"]["lexical_noise"] = {"budget": budget, "ops": applied}
            return new_item
        if isinstance(item, str):
            txt, _ = apply_to_text(item)
            return txt
        return item

    if target == "snippets_first":
        snippets[0] = apply_to_snippet_item(snippets[0])
    elif target == "snippets_all":
        snippets = [apply_to_snippet_item(s) for s in snippets]

    _set_snippets(ex, snippets)
    ex.setdefault("meta", {})
    ex["meta"]["lexical_noise"] = {"budget": budget, "ops": ops}
    return ex


# ---------------------------------------------------------------------
# Contradiction (yes/no focused, template-based)
# ---------------------------------------------------------------------

YESNO_TRUE = {"yes", "true"}
YESNO_FALSE = {"no", "false"}

ASSOCIATION_TRIGGERS = [
    "associated with", "correlated with", "linked to", "related to",
]
CAUSAL_TRIGGERS = [
    "causes", "cause", "leads to", "results in", "induces", "triggers",
]
TREATMENT_TRIGGERS = [
    "used to treat", "treats", "effective for", "recommended for",
]


def _extract_subject_object(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = (question or "").strip()
    q_low = q.lower()

    for trig in ASSOCIATION_TRIGGERS + CAUSAL_TRIGGERS + TREATMENT_TRIGGERS:
        if trig in q_low:
            parts = q_low.split(trig, 1)
            if len(parts) == 2:
                left = parts[0].strip(" ?.,;:()[]{}\"'")
                right = parts[1].strip(" ?.,;:()[]{}\"'")
                return left if left else None, right if right else None
    return None, None


def _generate_contradiction_yesno(
    gold_polarity: str,
    subject: Optional[str],
    obj: Optional[str],
    strength: str,
    style: str,
) -> str:
    # flip stance
    flip = "neg" if gold_polarity == "pos" else "pos"

    core = f"between {subject} and {obj}" if subject and obj else "with respect to the relationship under investigation"

    strength = (strength or "medium").lower()
    style = (style or "paper").lower()

    # style knob (very light-touch)
    if style == "guideline":
        if flip == "neg":
            base = f"Current guidance does not support a consistent association {core}."
        else:
            base = f"Current guidance suggests an association {core} may be considered in some cases."
        return base

    if strength == "strong":
        return f"Multiple studies report {'no significant' if flip == 'neg' else 'a significant'} association {core}."

    if strength == "weak":
        if flip == "neg":
            return f"Some reports suggest the association {core} may not be consistently observed."
        return f"Some evidence suggests an association {core} may exist in specific contexts."

    # medium
    if flip == "neg":
        return f"Several studies indicate that the association {core} is not consistently supported."
    return f"Several studies indicate that an association {core} has been observed in some settings."


def _perturb_contradiction(example: Example, cfg: Dict[str, Any], position: str = "append") -> Example:
    """
    Inject a contradictory snippet; Phase 3 scope: yes/no questions.

    Config keys (optional):
      - contradiction_strength: "weak"|"medium"|"strong" (default "medium")
      - contradiction_style: "paper"|"guideline"|"hedged" (default "paper")
      - contradiction_position: "append"|"prepend" (overrides position arg)
    """
    ex = copy.deepcopy(example)

    qtype = _get_question_type(ex)
    if qtype != "yesno":
        return ex

    answers = _flatten_exact_answer(ex.get("exact_answer"))
    if not answers:
        return ex
    a = answers[0].lower().strip()
    if a in YESNO_TRUE:
        polarity = "pos"
    elif a in YESNO_FALSE:
        polarity = "neg"
    else:
        return ex

    q_text = _get_question_text(ex)
    subj, obj = _extract_subject_object(q_text)

    strength = cfg.get("contradiction_strength", "medium")
    style = cfg.get("contradiction_style", "paper")
    pos = (cfg.get("contradiction_position") or position).lower()

    text = _generate_contradiction_yesno(polarity, subj, obj, strength=strength, style=style)

    snippet = {
        "text": text,
        "source": "contradiction",
        "meta": {"kind": "yesno", "strength": strength, "style": style},
    }

    snippets = _normalize_snippets(ex)
    if pos == "prepend":
        snippets = [snippet] + snippets
    else:
        snippets = snippets + [snippet]
    _set_snippets(ex, snippets)
    return ex