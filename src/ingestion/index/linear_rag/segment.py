"""Segment a document's ``combined.md`` into the LinearRAG tri-graph units,
carrying exact source offsets so EvidenceFS can address evidence by line.

The pipeline used to treat one PaddleOCR page asset as one passage. Here we
segment ``combined.md`` itself — the canonical, agent-readable text — so every
passage and sentence knows its ``[start_line, end_line]`` / byte span back into
that file. Offsets are captured at cut time (no post-hoc anchoring), which is
both exact and immune to OCR-table reflow.

Structure (relation-free, vanilla LinearRAG):

    combined.md
      └─ page block        (split on the page-boundary marker)
           └─ passage      (paragraph: a run between blank lines)
                └─ sentence (pysbd, zh/en auto)

Entity extraction (GLiNER) runs downstream on the passage text; this module
only produces the *spans*.
"""
import bisect
import re
from dataclasses import dataclass, field

from ingestion.index._sentence import split_sentences

# combined.md separates consecutive pages with this HTML comment (one per page
# transition; N markers => N+1 pages). Confirmed against meta.json.total_pages.
PAGE_MARKER = "<!-- agentic:batch_boundary -->"

# A passage boundary is a run of blank lines. OCR'd HTML tables are emitted as a
# single physical line, so blank-line splitting keeps a table intact as one
# passage rather than shredding it.
_PARA_SPLIT = re.compile(r"\n[ \t]*\n")


@dataclass
class SentenceSpan:
    text: str
    start_line: int
    end_line: int
    start_char: int      # byte offset into combined.md
    end_char: int


@dataclass
class PassageSpan:
    text: str
    page_number: int
    start_line: int
    end_line: int
    start_char: int
    end_char: int
    sentences: list[SentenceSpan] = field(default_factory=list)


def _line_of(nls: list[int], char_pos: int) -> int:
    """1-based line number of ``char_pos`` via precomputed newline offsets.

    ``nls`` is the sorted list of newline byte positions in combined.md; the
    line number is one more than the count of newlines strictly before
    ``char_pos`` — an ``O(log lines)`` bisect rather than an ``O(char_pos)``
    prefix scan (which made the whole segmentation O(doc²)).
    """
    return bisect.bisect_left(nls, char_pos) + 1


def _split_sentences_with_spans(
    nls: list[int], p_start: int, p_text: str,
) -> list[SentenceSpan]:
    """Locate each pysbd sentence inside the passage and lift to md offsets.

    A monotonic cursor handles repeated identical fragments (e.g. table
    headers). A sentence that fails to locate (rare reflow) degrades to a
    zero-width span at the cursor rather than dropping the unit.
    """
    spans: list[SentenceSpan] = []
    cursor = 0
    for sent in split_sentences(p_text):
        local = p_text.find(sent, cursor)
        if local < 0:
            local = cursor
            cursor_advance = 0
        else:
            cursor_advance = len(sent)
        sc = p_start + local
        ec = sc + len(sent)
        spans.append(SentenceSpan(
            text=sent, start_char=sc, end_char=ec,
            start_line=_line_of(nls, sc),
            end_line=_line_of(nls, max(sc, ec - 1)),
        ))
        cursor = local + cursor_advance
    return spans


def segment_combined_md(md: str) -> list[PassageSpan]:
    """Segment one ``combined.md`` into offset-carrying passages + sentences.

    O(n) in the size of the document: a single pass over page blocks, each
    split once into paragraphs; sentence location is a forward-cursor scan.
    """
    passages: list[PassageSpan] = []
    # Precompute newline offsets once (O(L)); line lookups are then O(log L).
    nls = [m.start() for m in re.finditer("\n", md)]
    # Page blocks: split on the marker; block i is page i+1. We keep absolute
    # offsets by walking the marker positions rather than re-finding substrings.
    page_no = 0
    block_start = 0
    n = len(md)
    marker_len = len(PAGE_MARKER)
    search_from = 0
    while True:
        idx = md.find(PAGE_MARKER, search_from)
        block_end = idx if idx >= 0 else n
        page_no += 1
        _emit_block(nls, md[block_start:block_end], block_start, page_no, passages)
        if idx < 0:
            break
        block_start = idx + marker_len
        search_from = block_start
    return passages


def _emit_block(
    nls: list[int], block: str, block_start: int, page_no: int,
    out: list[PassageSpan],
) -> None:
    """Split one page block into paragraph passages, carry offsets."""
    pos = 0
    for chunk in _PARA_SPLIT.split(block):
        # advance to this chunk's real position (skips the blank-line gap)
        local = block.find(chunk, pos) if chunk else -1
        if local < 0:
            pos += len(chunk)
            continue
        pos = local + len(chunk)
        text = chunk.strip()
        if not text:
            continue
        # tighten span to the stripped text
        lead = chunk.find(text)
        sc = block_start + local + (lead if lead >= 0 else 0)
        ec = sc + len(text)
        passage = PassageSpan(
            text=text, page_number=page_no, start_char=sc, end_char=ec,
            start_line=_line_of(nls, sc), end_line=_line_of(nls, max(sc, ec - 1)),
        )
        passage.sentences = _split_sentences_with_spans(nls, sc, text)
        out.append(passage)
