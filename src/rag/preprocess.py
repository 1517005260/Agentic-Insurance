"""Query preprocessing — two LLM calls in parallel + lang sanity check.

* ``hyde_call`` — Hypothetical Document Embeddings: generate a plausible
  answer document so the semantic / BM25 channels can match content
  the bare question wouldn't (HyDE: Gao et al., 2022).
* ``rewrite_regex_call`` — paraphrase + lang tag + regex patterns for the
  regex scan channel.

Both prompts are written in English but include explicit "match the user's
language" instructions: zh question → zh outputs; en → en; mixed → mixed.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List

import regex as ureg

from model_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- prompts ----

_HYDE_PROMPT = """\
Write a concise hypothetical answer (about 80-150 words) for the question \
below. The text is for retrieval, not for the user — pack it with the \
specific terms, entities, numbers, and phrasing that would appear in a real \
answer. Every sentence should add a retrievable cue; no filler, no preamble \
("This document covers…"), no bullet headings.

LANGUAGE MATCHING (critical):
- Detect the language(s) of the question.
- Write the answer in the SAME language(s) as the question.
- Chinese question -> Chinese answer. English -> English. Mixed -> mixed.
- Do NOT translate; preserve domain terms in their original script.

If uncertain, write the most plausible answer based on common domain \
knowledge — never say "I don't know".

Question: {query}
Hypothetical answer:"""


_REWRITE_PROMPT = """\
Analyze the question and produce a strict JSON object with three fields.

LANGUAGE MATCHING (critical):
- All natural-language outputs (rewrite, regex literals) MUST match the \
language(s) of the question.
- Chinese question -> Chinese rewrite + Chinese regex literals.
- English -> English. Mixed -> mixed.

Output schema (return ONLY this JSON, no prose, no code fences):
{{
  "rewrite": "<paraphrase of the question, same language(s) as input, \
preserving entities/numbers/proper nouns verbatim>",
  "lang": "zh" | "en" | "mixed",
  "regexes": [
    {{
      "pattern": "<Python re-module syntax regex likely to match relevant \
phrasings in document text; Unicode-aware where applicable>",
      "weight": <float in [0.3, 1.0] — how diagnostic this pattern is>,
      "rationale": "<one short sentence>"
    }}
  ]
}}

Regex generation guidelines:
- Produce 3-6 patterns covering different angles of the question.
- Avoid bare \\d+ / .* / .+ — always anchor with literal terms or context.
- For numeric values, anchor with the surrounding label, e.g. \
"保险金额[:：]?\\\\s*\\\\d+" or "(?i)deductible[:\\\\s]+\\\\$?\\\\d+".
- For Chinese, use literal characters; mix Han + ASCII as needed.
- weight 0.9-1.0: highly diagnostic literal anchor.
- weight 0.5-0.8: moderate.
- weight 0.3-0.5: speculative / fallback.

Question: {query}
JSON:"""


# ----------------------------------------------------------------- types ----


@dataclass
class RegexSpec:
    pattern: str
    weight: float
    rationale: str = ""


@dataclass
class QueryContext:
    """Everything the channels need from the preprocessing step."""

    query: str
    hyde: str
    rewrite: str
    lang: str  # "zh" | "en" | "mixed"
    regexes: List[RegexSpec] = field(default_factory=list)
    file_ids: List[str] | None = None


# ------------------------------------------------------------ language ----


def detect_lang_local(text: str) -> str:
    """Han ideograph in text → ``zh``; otherwise ``en``."""
    return "zh" if ureg.search(r"\p{Han}", text or "") else "en"


def sanity_check_lang(query: str, llm_lang: str) -> str:
    """Trust ``llm_lang == "mixed"`` (LLM saw both); otherwise the local
    detection wins because the query string is the ground truth."""
    if llm_lang == "mixed":
        return "mixed"
    local = detect_lang_local(query)
    return local


# ----------------------------------------------------------------- calls ----


def hyde_call(query: str, llm: LLMClient) -> str:
    """Return the LLM's hypothetical answer document, plain text."""
    result = llm.chat(
        messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        temperature=0.7,  # a little entropy helps surface vocabulary breadth
    )
    return result["message"].get("content", "").strip()


# Strip a leading code fence (```json ... ```), tolerating both ``` and ```json.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def rewrite_regex_call(query: str, llm: LLMClient) -> Dict[str, Any]:
    """Return ``{"rewrite": str, "lang": str, "regexes": [...]}``.

    LLMs occasionally wrap JSON in code fences despite instructions; we
    strip a single fence non-destructively and then parse. On parse
    failure we degrade to a safe empty result rather than crashing the
    pipeline — the BM25/semantic/PPR channels still work.
    """
    raw = llm.chat(
        messages=[{"role": "user", "content": _REWRITE_PROMPT.format(query=query)}],
        temperature=0.0,
    )["message"].get("content", "").strip()

    fenced = _FENCE_RE.match(raw)
    if fenced:
        raw = fenced.group(1)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("rewrite_regex_call: JSON parse failed (%s); falling back", exc)
        return {"rewrite": query, "lang": detect_lang_local(query), "regexes": []}

    rewrite = (parsed.get("rewrite") or query).strip()
    lang = parsed.get("lang") or detect_lang_local(query)
    regexes: List[RegexSpec] = []
    for item in parsed.get("regexes") or []:
        # One malformed entry shouldn't kill preprocessing — skip and log.
        try:
            pattern = (item.get("pattern") or "").strip()
            if not pattern:
                continue
            weight_raw = item.get("weight")
            try:
                weight = float(weight_raw) if weight_raw is not None else 0.5
            except (TypeError, ValueError):
                weight = 0.5
            weight = max(0.3, min(1.0, weight))
            regexes.append(
                RegexSpec(
                    pattern=pattern,
                    weight=weight,
                    rationale=(item.get("rationale") or "").strip(),
                )
            )
        except Exception as exc:
            logger.warning("rewrite_regex_call: dropping malformed regex spec %r (%s)", item, exc)
    return {"rewrite": rewrite, "lang": lang, "regexes": regexes}


# ------------------------------------------------------------- preprocess ----


def preprocess(
    query: str,
    llm: LLMClient,
    file_ids: List[str] | None = None,
) -> QueryContext:
    """Run HyDE + rewrite/regex calls in parallel, then synthesize QueryContext."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_hyde = pool.submit(hyde_call, query, llm)
        fut_rewrite = pool.submit(rewrite_regex_call, query, llm)
        hyde = fut_hyde.result()
        meta = fut_rewrite.result()

    lang = sanity_check_lang(query, meta["lang"])
    return QueryContext(
        query=query,
        hyde=hyde,
        rewrite=meta["rewrite"],
        lang=lang,
        regexes=meta["regexes"],
        file_ids=file_ids,
    )
