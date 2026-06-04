"""Adapter for HippoRAG-format cross-document multi-hop corpora.

Covers 2WikiMultihopQA / MuSiQue / HotpotQA, which ship each retrieval unit
as one short passage and reference gold support across *different* passages
(true cross-document multi-hop, unlike Double-Bench's single-document multi
page). Two file shapes appear:

* ``*_corpus.json`` — the deduplicated global pool: ``[{title, text, idx}]``.
* ``*.json`` (QA) — per-question items. Field shapes differ by dataset:
    - 2Wiki / HotpotQA: ``context = [[title, [sentence, ...]], ...]`` plus
      ``supporting_facts = [[title, sent_idx], ...]`` and ``evidences`` (gold
      ``[subject, relation, object]`` hop triples).
    - MuSiQue: ``paragraphs = [{title, paragraph_text, is_supporting, idx}]``
      plus ``question_decomposition`` (gold hop chain).

**One file_id per passage** (see :func:`corpus_file_id`). The build links
adjacent pages only *within* a file_id; collapsing the whole corpus under one
file_id would chain unrelated passages with ordering-artifact edges. With one
passage per file_id that linking is inert, so cross-passage multi-hop is
carried entirely by shared entity nodes (same surface → same hashed node
across file_ids) and the sentence co-occurrence graph.

The emitted PageAsset matches the :mod:`ingestion.dbocr_pageassets` contract
exactly so the agent's read tool and graph tool work unchanged:
``page_number=1``; ``text_markdown`` and ``layout_blocks`` render from one
shared block list (closure gate substring-matches both); the article title
renders as an ATX heading so the title surface — frequently the multi-hop
bridge entity — is NER-visible, retrievable, and citable; no tables / images
/ raster.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from ingestion.page_assets import persist_and_warm_page_assets
from ingestion.page_mode import PageModeSignals, classify_page_mode
from storage.page_store import PageAsset

_PASSAGE_CHAR_BUDGET = 1200
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Open-set GLiNER label palette for general-encyclopedic (Wikipedia) text.
# Lives with the dataset adapter (not in the core LinearRAGConfig default,
# which is insurance/legal-tuned) and is injected via
# ``LinearRAGConfig(gliner_labels=...)`` by the build driver — the algorithm
# layer never hard-codes a domain palette. Kept short and English (GLiNER's
# mT5 backbone tokenises English label tokens most stably). Covers the entity
# types 2Wiki / MuSiQue / HotpotQA questions hinge on.
WIKI_GLINER_LABELS: List[str] = [
    "person",
    "location",
    "organization",
    "date",
    "creative work",
    "event",
    "group",
    "nationality",
    "award",
    "occupation",
]

# Build-stage knobs for cross-document wiki corpora, passed to the build driver
# as ``LinearRAGConfig(gliner_labels=WIKI_GLINER_LABELS, **WIKI_BUILD_KNOBS)``.
# Wikipedia entity surfaces are clean (exact shared surfaces already hash to one
# node), so alias ER is both the build's dominant cost and a false-bridge risk →
# off. Each passage is its own single-page file_id, so adjacent-passage edges
# would only link arbitrary corpus neighbours → off. Literal backfill off keeps
# the PPR baseline simple (titles are already in the passage text); flip it on
# (it runs once at final flush) if a later ablation wants it.
WIKI_BUILD_KNOBS = {
    "alias_edges_enabled": False,
    "adjacent_passage_edges_enabled": False,
    "literal_backfill_enabled": False,
}


def corpus_file_id(dataset: str, idx: Any) -> str:
    """Stable per-passage file_id, e.g. ``2wiki_000042``.

    ``idx`` is the corpus item's own index (int) when present so the identity
    is reproducible across rebuilds; non-int ids are slugged verbatim.
    """
    try:
        return f"{dataset}_{int(idx):06d}"
    except (TypeError, ValueError):
        return f"{dataset}_{re.sub(r'[^0-9A-Za-z]+', '_', str(idx))}"


def _split_to_budget(text: str, budget: int = _PASSAGE_CHAR_BUDGET) -> List[str]:
    """Split prose into <=``budget``-char chunks on sentence boundaries
    (hard-splitting any single oversized sentence). Wiki passages are short so
    this is usually a single chunk; long MuSiQue paragraphs split."""
    text = text.strip()
    if len(text) <= budget:
        return [text] if text else []
    chunks: List[str] = []
    cur = ""
    for sent in _SENTENCE_SPLIT_RE.split(text):
        if not sent:
            continue
        if len(sent) > budget:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(sent), budget):
                chunks.append(sent[i : i + budget])
            continue
        if cur and len(cur) + 1 + len(sent) > budget:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def build_pageasset_from_corpus_item(
    title: str,
    text: str,
    file_id: str,
    *,
    persist: bool = False,
) -> List[PageAsset]:
    """Build the single-page PageAsset list for one corpus passage.

    ``persist=False`` by default: a bulk corpus driver writes the page_assets
    JSON itself and warms the inventory once at the end, because
    ``persist_and_warm_page_assets`` re-scans every page_assets file on each
    call (O(N²) over a 6k–12k passage corpus). Single-item callers can pass
    ``persist=True``.
    """
    title = (title or "").strip()
    body = (text or "").strip()

    blocks: List[Dict[str, Any]] = []
    md_parts: List[str] = []

    def _add(label: str, content: str) -> None:
        bid = len(blocks)
        blocks.append(
            {
                "block_label": label,
                "block_content": content,
                "block_id": bid,
                "block_order": bid,
            }
        )
        md_parts.append(content)

    if title:
        _add("paragraph_title", f"# {title}")
    for chunk in _split_to_budget(body):
        _add("text", chunk)

    text_markdown = "\n\n".join(md_parts).strip()

    signals = PageModeSignals(
        has_table=False,
        has_chart=False,
        has_figure=False,
        low_ocr_confidence=False,
        scanned_page=False,
        dense_layout=False,
    )
    page = PageAsset(
        page_id="p_0001",
        file_id=file_id,
        page_number=1,
        text_markdown=text_markdown,
        page_image_path=None,
        table_blocks=[],
        image_blocks=[],
        layout_blocks=blocks,
        page_mode=classify_page_mode(signals),
        quality_flags={
            "has_table": False,
            "has_chart": False,
            "has_figure": False,
            "low_ocr_confidence": False,
            "scanned_page": False,
            "dense_layout": False,
        },
    )
    pages = [page]
    if persist:
        persist_and_warm_page_assets(file_id, pages)
    return pages


def load_corpus(corpus_json: Path) -> List[Dict[str, Any]]:
    """Load a HippoRAG ``*_corpus.json`` (list of ``{title, text, idx}``)."""
    data = json.loads(Path(corpus_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{corpus_json}: expected a list of corpus items")
    return data


def load_qa(qa_json: Path) -> List[Dict[str, Any]]:
    """Load a HippoRAG QA file (list of per-question dicts)."""
    data = json.loads(Path(qa_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{qa_json}: expected a list of QA items")
    return data


def corpus_text(item: Dict[str, Any]) -> str:
    """Extract the passage body from a corpus item across field-name variants."""
    return item.get("text") or item.get("paragraph_text") or item.get("passage") or ""


def iter_question_paragraphs(q: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Yield ``(title, text)`` candidate passages for one QA item.

    Handles both HippoRAG QA shapes:
    * 2Wiki / HotpotQA ``context = [[title, [sentence, ...]], ...]`` (or a
      ``[title, "joined text"]`` variant, or a ``{title, sentences|text}`` dict).
    * MuSiQue ``paragraphs = [{title, text|paragraph_text}, ...]``.
    """
    for p in q.get("paragraphs", []) or []:
        t = corpus_text(p)
        if t.strip():
            yield p.get("title", ""), t
    for c in q.get("context", []) or []:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            title, body = c[0], c[1]
            text = " ".join(body) if isinstance(body, (list, tuple)) else str(body)
            if text.strip():
                yield str(title), text
        elif isinstance(c, dict):
            text = corpus_text(c) or " ".join(c.get("sentences", []) or [])
            if text.strip():
                yield c.get("title", ""), text
