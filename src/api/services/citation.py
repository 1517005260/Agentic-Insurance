"""Citation builder — turn reranked pages into a [^k] legend, parse responses.

Web-layer service shared by RAG and agent runners. The runner:

1. Builds a ``CitationBuilder`` from the reranked pages
   (RAG) or claim list (agent).
2. Calls :meth:`render_legend_for_prompt` to produce a text block that
   gets appended to the user message, plus :meth:`render_pages_block`
   to render each page header with its ``[^k]`` label inline so the
   model sees the label co-located with the page content.
3. After streaming finishes, calls :meth:`parse_response` to extract
   the ``CitationItem`` list that gets emitted as the SSE
   ``citations`` event and persisted in ``chat_messages.metadata_json``.

Algorithm-side ``rag/answer.py`` does NOT depend on this module — the
experiment path uses its plain prompt and never numbers citations.
"""
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

from rag.rerank import RerankedPage


logger = logging.getLogger(__name__)


@dataclass
class CitationItem:
    """One entry in the SSE ``citations`` event payload.

    ``sup`` is the 1-based label the LLM emits as ``[^sup]``; the
    frontend renders it as a ``<sup>`` linking to a side drawer that
    opens ``file_id`` at ``page_number``. ``page_preview`` is a short
    excerpt to show in the drawer header before the PDF render kicks
    in. ``observation_id`` is populated for agent paths (links the
    citation to the read observation), None for RAG.
    """

    sup: int
    file_id: str
    page_id: str
    page_number: Optional[int] = None
    page_preview: Optional[str] = None
    observation_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# [^k] markers in the response. Tolerant of nested brackets but not
# mid-word junk: must follow a non-word boundary or BOS, ``k`` is digits.
_SUP_RE = re.compile(r"\[\^(\d+)\]")

# Module-level default cap on inline page-preview length. The runner
# threads ``preview_chars`` into :meth:`from_reranked_pages` so the
# admin config-center value wins per request; this default applies
# only when nothing was specified (smoke-route path with no config).
DEFAULT_PREVIEW_CHARS = 240


@dataclass
class CitationBuilder:
    """Holds a numbered list of pages and parses LLM responses against it."""

    items: List[CitationItem] = field(default_factory=list)

    @classmethod
    def from_reranked_pages(
        cls,
        pages: Sequence[RerankedPage],
        *,
        preview_chars: Optional[int] = None,
    ) -> "CitationBuilder":
        """Build a builder from rerank output. ``sup`` is rerank rank, 1-based.

        ``preview_chars`` (None → :data:`DEFAULT_PREVIEW_CHARS`) caps
        each ``CitationItem.page_preview`` length so the side drawer
        opens with a snippet rather than the full page Markdown.
        """
        cap = preview_chars if preview_chars is not None else DEFAULT_PREVIEW_CHARS
        items: List[CitationItem] = []
        for idx, r in enumerate(pages, start=1):
            p = r.page
            preview = (p.text_markdown or "").strip().replace("\n", " ")
            items.append(
                CitationItem(
                    sup=idx,
                    file_id=p.file_id,
                    page_id=p.page_id,
                    page_number=p.page_number,
                    page_preview=preview[:cap] if preview else None,
                )
            )
        return cls(items=items)

    # ------------------------------------------------------------------ render

    def render_legend_for_prompt(self) -> str:
        """Render the ``[^k] -> (file_id, page_number)`` block for the prompt.

        Empty list → empty string (caller should branch and pass a
        bare "no pages found" instead of an empty legend).

        Wording (``page_number=...``) matches what the business prompt
        tells the model to expect; keep these aligned when editing.
        """
        if not self.items:
            return ""
        lines = ["Cited pages (use ONLY these [^k] labels in your answer):"]
        for it in self.items:
            label = f"[^{it.sup}]"
            page = it.page_number if it.page_number is not None else it.page_id
            lines.append(f"  {label} file_id={it.file_id} page_number={page}")
        return "\n".join(lines)

    def render_pages_block(self, pages: Sequence[RerankedPage]) -> str:
        """Render the page-content block with each page header carrying ``[^k]``.

        Co-locating the label with the actual page content makes the
        mapping unambiguous to the model — it doesn't have to triangulate
        the legend against header order. Pages missing from ``self.items``
        (which shouldn't happen, since the builder is constructed from
        the same page list) emit an unlabeled header instead of crashing.
        """
        if not pages:
            return "(no pages found)"
        index_by_global = {(it.file_id, it.page_id): it for it in self.items}
        parts: List[str] = []
        for r in pages:
            p = r.page
            it = index_by_global.get((p.file_id, p.page_id))
            label = f"[^{it.sup}] " if it is not None else ""
            parts.append(
                f"----- {label}Page file_id={p.file_id} page_id={p.page_id} -----\n"
                f"{p.text_markdown}\n"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------- parse

    def parse_response(self, text: str) -> tuple[str, List[CitationItem]]:
        """Extract ``[^k]`` references from ``text``.

        Returns ``(text_unmodified, list_of_items_actually_cited)``. The
        text is intentionally not mutated — the frontend renders the
        ``[^k]`` markers as clickable sups via a markdown plugin, so
        stripping them here would break that.

        Unknown labels (k not in the legend) are silently dropped:
        they're typically the model hallucinating a higher index than
        we provided, and showing a broken link to the user is worse
        than dropping the cite.
        """
        cited_sups: List[int] = []
        for match in _SUP_RE.finditer(text):
            try:
                k = int(match.group(1))
            except ValueError:
                continue
            if k not in cited_sups:
                cited_sups.append(k)

        legend_by_sup = {it.sup: it for it in self.items}
        cited_items: List[CitationItem] = []
        unknown: List[int] = []
        for k in cited_sups:
            if k in legend_by_sup:
                cited_items.append(legend_by_sup[k])
            else:
                unknown.append(k)
        if unknown:
            logger.warning(
                "citation parser: model cited labels not in legend: %s (legend=%s)",
                unknown,
                sorted(legend_by_sup.keys()),
            )
        return text, cited_items
