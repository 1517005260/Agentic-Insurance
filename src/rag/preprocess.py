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
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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

    # PPR-only knob: when True, GraphPPRChannel adds a query-side seed
    # fallback (literal gazetteer scan ➜ entity-embedding top-K) for
    # questions where NER finds no entity. The 4-channel RAG path
    # leaves this off (returning empty preserves RRF channel
    # independence). The graph_explore agent tool flips it on because
    # PPR is the only signal available there. See graph_ppr._seed_entities.
    enable_ppr_seed_fallback: bool = False


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


def _format_history_block(history_user_only: List[str]) -> str:
    """Format prior user queries as a coreference-resolution prefix.

    Only the user side: rewrite is about resolving "它 / that one /
    上一个" references in the *current* question, not reasoning about
    the assistant's prior answers. Including the assistant text would
    bloat the cheap rewrite call without changing the rewrite output
    on questions that aren't ambiguous in the first place.
    """
    if not history_user_only:
        return ""
    lines = [f"  T{i+1}: {q}" for i, q in enumerate(history_user_only)]
    return (
        "Conversation history (user questions only, oldest → newest):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def rewrite_regex_call(
    query: str,
    llm: LLMClient,
    *,
    history_user_only: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return ``{"rewrite": str, "lang": str, "regexes": [...]}``.

    LLMs occasionally wrap JSON in code fences despite instructions; we
    strip a single fence non-destructively and then parse. On parse
    failure we degrade to a safe empty result rather than crashing the
    pipeline — the BM25/semantic/PPR channels still work.

    ``history_user_only`` (optional) prepends prior user questions so
    the rewrite can resolve references ("它的免赔额" → "AXA 盛利II
    的免赔额"). Default ``None`` keeps single-turn behaviour.
    """
    history_block = _format_history_block(history_user_only or [])
    prompt = (
        history_block
        + (
            "Resolve any references in the *current* question to entities mentioned above, "
            "writing the rewrite as a self-contained, standalone question.\n\n"
            if history_block
            else ""
        )
        + _REWRITE_PROMPT.format(query=query)
    )
    raw = llm.chat(
        messages=[{"role": "user", "content": prompt}],
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


# How many chars of the HyDE / rewrite outputs to ship in the SSE
# events. Full HyDE can be 500+ chars; the frontend only needs enough
# to show "what query rewrite the system is using" — a preview.
_PREPROCESS_PREVIEW_CHARS = 240

# Wall-clock budget for the *parallel* HyDE + rewrite phase. The
# upstream LLM client uses a (10s, 120s) connect/read timeout,
# but a stalled relay that ships a single keep-alive byte every 119s
# would still drag this stage out indefinitely. When the wall fires
# we cancel any in-flight future, emit synthetic ``done`` events with
# safe fallbacks (rewrite = original query, empty regex list), and
# let the rest of the pipeline run instead of hanging the whole
# request. The user-visible failure mode flips from "RAG silently
# never produces tokens" to "answer is computed against the un-rewritten
# query" — degraded but responsive.
_PREPROCESS_TIMEOUT_S: float = 130.0


def preprocess(
    query: str,
    llm: LLMClient,
    file_ids: List[str] | None = None,
    *,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    history_user_only: Optional[List[str]] = None,
) -> QueryContext:
    """Run HyDE + rewrite/regex calls in parallel, then synthesize QueryContext.

    ``on_event`` (optional) receives one ``preprocess`` event per
    sub-step transition so a streaming UI can show two parallel
    spinners ("HyDE" and "rewrite + regex"), each reaching its own
    completion independently. Default ``None`` keeps the experiment
    path silent — same behavior as before.

    Emitted events (raw signature; pipeline.py wraps it via the same
    safe-emit helper used elsewhere):

    * ``preprocess {"step":"hyde","phase":"start"}``
    * ``preprocess {"step":"rewrite","phase":"start"}``
    * ``preprocess {"step":"hyde","phase":"done","elapsed_ms":N,"hyde_preview":"..."}``
    * ``preprocess {"step":"rewrite","phase":"done","elapsed_ms":N,
        "lang":..., "rewrite":..., "regexes":[{"pattern","weight","rationale"}]}``
    """
    def _emit(event: str, data: Dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(event, data)
        except Exception:
            logger.exception("preprocess on_event failed")

    # Manual lifecycle (no ``with`` block): the context-manager exit
    # calls ``shutdown(wait=True)``, which would block on hung LLM
    # futures and defeat the wall-clock fallback below. We instead
    # call ``shutdown(wait=False)`` from the ``finally`` so the pool
    # background-cleans the cancelled futures while the caller
    # proceeds with the safe-fallback ``meta`` dict.
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        # We push start events BEFORE submitting so the frontend can
        # show "HyDE 启动 / rewrite 启动" right away — submit() is
        # near-instant but the LLM round-trip is 1-3s, so seeing the
        # spinners light up matters.
        _emit("preprocess", {"step": "hyde", "phase": "start"})
        _emit("preprocess", {"step": "rewrite", "phase": "start"})
        t0_hyde = time.perf_counter()
        t0_rewrite = time.perf_counter()
        fut_hyde = pool.submit(hyde_call, query, llm)
        # Only the rewrite path consumes history (coreference resolution);
        # HyDE is a hypothetical answer for embedding anchoring and would
        # see no benefit, only added prompt cost.
        fut_rewrite = pool.submit(
            rewrite_regex_call,
            query,
            llm,
            history_user_only=history_user_only,
        )

        # as_completed lets the faster path's "done" event fire as
        # soon as it finishes, even if the slower one hasn't returned
        # yet — that's the whole point of running them in parallel.
        # The ``timeout`` argument is a wall-clock fence on the *whole*
        # iteration: when it fires (e.g. one of the LLM calls has hung
        # past the upstream read timeout's keep-alive window) we
        # convert any unfinished futures into safe-fallback events so
        # the rest of the pipeline sees usable data.
        fut_to_step = {fut_hyde: "hyde", fut_rewrite: "rewrite"}
        fut_t0 = {fut_hyde: t0_hyde, fut_rewrite: t0_rewrite}
        remaining = {fut_hyde, fut_rewrite}
        hyde: str = ""
        meta: Optional[Dict[str, Any]] = None
        try:
            for fut in as_completed([fut_hyde, fut_rewrite], timeout=_PREPROCESS_TIMEOUT_S):
                remaining.discard(fut)
                step = fut_to_step[fut]
                elapsed_ms = int((time.perf_counter() - fut_t0[fut]) * 1000)
                if step == "hyde":
                    try:
                        hyde = fut.result()
                    except Exception as exc:
                        # A blow-up inside the LLM call (HTTP 500,
                        # parse error, …) shouldn't kill RAG — log
                        # and proceed with empty HyDE so the dense
                        # channel still queries against the bare
                        # question.
                        logger.warning("preprocess hyde call raised: %s", exc)
                        hyde = ""
                    _emit(
                        "preprocess",
                        {
                            "step": "hyde",
                            "phase": "done",
                            "elapsed_ms": elapsed_ms,
                            "hyde_preview": (hyde or "")[:_PREPROCESS_PREVIEW_CHARS],
                            "hyde_chars": len(hyde or ""),
                        },
                    )
                else:
                    try:
                        meta = fut.result()
                    except Exception as exc:
                        logger.warning("preprocess rewrite call raised: %s", exc)
                        meta = {
                            "rewrite": query,
                            "lang": detect_lang_local(query),
                            "regexes": [],
                        }
                    _emit(
                        "preprocess",
                        {
                            "step": "rewrite",
                            "phase": "done",
                            "elapsed_ms": elapsed_ms,
                            "lang": meta.get("lang"),
                            "rewrite": meta.get("rewrite", ""),
                            "regexes": [
                                {
                                    "pattern": r.pattern,
                                    "weight": r.weight,
                                    "rationale": r.rationale,
                                }
                                for r in meta.get("regexes", [])
                            ],
                        },
                    )
        except FutTimeout:
            # Wall-clock fired with one or both futures still running.
            # Emit synthetic done events so the SSE consumer sees
            # closure on every step it saw a "start" for, then
            # continue with safe fallbacks. ``cancel()`` is best-effort
            # — a future already executing the LLM HTTP call can't
            # actually stop the call, but it does prevent the result
            # from being silently consumed later.
            for fut in remaining:
                step = fut_to_step[fut]
                fut.cancel()
                elapsed_ms = int((time.perf_counter() - fut_t0[fut]) * 1000)
                if step == "hyde":
                    hyde = ""
                    _emit(
                        "preprocess",
                        {
                            "step": "hyde",
                            "phase": "done",
                            "elapsed_ms": elapsed_ms,
                            "hyde_preview": "",
                            "hyde_chars": 0,
                            "error": "timeout",
                        },
                    )
                else:
                    meta = {
                        "rewrite": query,
                        "lang": detect_lang_local(query),
                        "regexes": [],
                    }
                    _emit(
                        "preprocess",
                        {
                            "step": "rewrite",
                            "phase": "done",
                            "elapsed_ms": elapsed_ms,
                            "lang": meta["lang"],
                            "rewrite": meta["rewrite"],
                            "regexes": [],
                            "error": "timeout",
                        },
                    )
            logger.warning(
                "preprocess hit %.1fs wall: %d future(s) cancelled with fallbacks",
                _PREPROCESS_TIMEOUT_S, len(remaining),
            )
    finally:
        # ``wait=False`` is the whole point: a hung LLM future would
        # block ``shutdown(wait=True)`` (the implicit ``with`` exit)
        # and re-stretch the wall-clock fence we just enforced. The
        # background thread leaks until the LLM call eventually times
        # out at the chat-client layer (timeout=(10, 120) + read=0
        # retries → ≤ ~130 s), then the pool is GC'd cleanly.
        pool.shutdown(wait=False)

    if meta is None:
        # Both futures hit the wall (or the rewrite future raised before
        # ever returning a dict). Make sure downstream code never reads
        # ``meta["lang"]`` against a None.
        meta = {
            "rewrite": query,
            "lang": detect_lang_local(query),
            "regexes": [],
        }
    lang = sanity_check_lang(query, meta["lang"])
    return QueryContext(
        query=query,
        hyde=hyde,
        rewrite=meta["rewrite"],
        lang=lang,
        regexes=meta["regexes"],
        file_ids=file_ids,
    )
