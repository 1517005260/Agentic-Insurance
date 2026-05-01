"""Regex channel — LLM-generated patterns scanned over per-page Markdown.

Patterns come from the rewrite/regex preprocess step (each carries an
LLM-emitted ``weight`` ∈ [0.3, 1.0]). We scan the canonical per-page
Markdown stored in ``STORAGE_PATH/page_assets/<file_id>.json`` (same content
as ``paddle_ocr/<file_id>/combined.md`` but already split per page, so we
get accurate page boundaries for free).

Per-match score (codex-recommended scheme, see ``docs/rag_pipeline.md`` §4.4)::

    s_i = w_r * idf_r * quality(m, line) * g(m) * h(r)

where ``idf_r = log((N+1)/(df_r+1))`` is computed from the candidate page
universe in this query (so generic patterns naturally lose weight),
``quality`` rewards full-line matches over single-token slivers, ``g``
penalizes digit/punct-only matches and ``h`` penalizes overly broad
patterns (mostly wildcards / few literal characters).

Per-page dedup by ``(pattern, normalize(matched_text))`` collapses repeat
matches (saturated TF, capped at ``regex_dedup_cap``).
"""

import logging
import math
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import regex as ureg

from config import RAGConfig
from config.settings import page_assets_root
from rag.channels.base import BaseChannel, ChannelHit, RawHit, aggregate_per_page
from rag.preprocess import QueryContext, RegexSpec
from storage.page_store import PageAsset, PageStore


logger = logging.getLogger(__name__)


_DIGIT_PUNCT_RE = ureg.compile(r"^[\d\p{P}\p{S}\s]+$")

# "Too-broad" pattern heuristic: strip away meta-syntax and escape sequences,
# then count what's left. Anything resembling a real anchor (an English
# word, a Han character) survives; nothing but ``\d+\s*.*`` collapses to an
# empty / near-empty residue.
_META_STRIP_RE = re.compile(r"\\.|[.*+?(){}\[\]|^$]")


def _is_broad_pattern(pattern: str) -> bool:
    stripped = _META_STRIP_RE.sub("", pattern)
    return len(stripped) < 3


def _normalize_match(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("·,.;:!?，。；：！？—-_/\\\"'()[]{}<>")


def _quality(match_text: str, line: str) -> float:
    q = math.log1p(len(match_text)) / max(math.log1p(len(line) or 1), 1e-6)
    return max(0.1, min(1.0, q))


def _g_factor(match_text: str) -> float:
    return 0.2 if _DIGIT_PUNCT_RE.match(match_text) else 1.0


class RegexChannel(BaseChannel):
    name = "regex"

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        page_store: Optional[PageStore] = None,
        page_assets_dir: Optional[Path] = None,
    ):
        self.config = config or RAGConfig()
        self._page_store = page_store
        self._page_assets_dir = page_assets_dir or page_assets_root()
        # Debug snapshot — populated each retrieve() call.
        self.last_debug: Dict[str, object] = {}

    @property
    def page_store(self) -> Optional[PageStore]:
        if self._page_store is None and self._page_assets_dir.is_dir():
            self._page_store = PageStore(self._page_assets_dir)
        return self._page_store

    def retrieve(self, ctx: QueryContext) -> List[ChannelHit]:
        self.last_debug = {}
        store = self.page_store
        if store is None or len(store) == 0 or not ctx.regexes:
            self.last_debug["status"] = "no_store_or_no_patterns"
            return []

        # Compile all patterns first; bad ones get logged + skipped.
        compiled: List[Tuple[RegexSpec, ureg.Pattern]] = []
        compile_failures: List[str] = []
        for spec in ctx.regexes:
            try:
                compiled.append((spec, ureg.compile(spec.pattern, ureg.IGNORECASE)))
            except Exception as exc:
                logger.warning("regex_scan: failed to compile %r: %s", spec.pattern, exc)
                compile_failures.append(spec.pattern)

        # Scope to candidate pages — file_ids filter at the page-iteration step.
        pages = self._candidate_pages(store, ctx.file_ids)
        if not pages or not compiled:
            self.last_debug["status"] = "no_pages_or_no_valid_patterns"
            self.last_debug["compile_failures"] = compile_failures
            return []

        # df_r computed against the candidate pool so per-query idf adapts
        # to the active corpus rather than a global precomputed table.
        df = self._document_frequency(compiled, pages)
        n_pages = len(pages)

        self.last_debug["n_pages_scanned"] = n_pages
        self.last_debug["compile_failures"] = compile_failures
        self.last_debug["pattern_stats"] = [
            {
                "pattern": spec.pattern,
                "weight": spec.weight,
                "df": df.get(spec.pattern, 0),
                "idf": round(math.log((n_pages + 1) / (df.get(spec.pattern, 0) + 1)), 4),
                "broad": _is_broad_pattern(spec.pattern),
            }
            for spec, _ in compiled
        ]

        with ThreadPoolExecutor(max_workers=min(8, len(pages))) as pool:
            results = list(
                pool.map(
                    lambda p: self._scan_page(p, compiled, df, n_pages),
                    pages,
                )
            )
        raw: List[RawHit] = [hit for page_hits in results for hit in page_hits]
        return aggregate_per_page(raw, top_k=self.config.regex_channel_topk)

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _candidate_pages(
        store: PageStore, file_ids: Optional[List[str]]
    ) -> List[PageAsset]:
        all_pages = [store.get(gid) for gid in store.ids()]
        all_pages = [p for p in all_pages if p is not None]
        if not file_ids:
            return all_pages
        wanted = set(file_ids)
        return [p for p in all_pages if p.file_id in wanted]

    @staticmethod
    def _document_frequency(
        compiled: List[Tuple[RegexSpec, ureg.Pattern]],
        pages: Iterable[PageAsset],
    ) -> Dict[str, int]:
        df: Dict[str, int] = defaultdict(int)
        for page in pages:
            text = page.text_markdown or ""
            for spec, regex_obj in compiled:
                if regex_obj.search(text):
                    df[spec.pattern] += 1
        return df

    def _scan_page(
        self,
        page: PageAsset,
        compiled: List[Tuple[RegexSpec, ureg.Pattern]],
        df: Dict[str, int],
        n_pages: int,
    ) -> List[RawHit]:
        cfg = self.config
        text = page.text_markdown or ""
        if not text:
            return []
        lines = text.splitlines() or [text]

        # Collect per-match (score, evidence) tuples; a (pattern, normalized
        # match) pair seen on this page only counts once, so the same string
        # repeated across boilerplate doesn't dominate.
        scored: List[Tuple[float, Dict[str, object]]] = []
        seen: set = set()

        for spec, regex_obj in compiled:
            df_r = df.get(spec.pattern, 0)
            if df_r == 0:
                continue
            idf = math.log((n_pages + 1) / (df_r + 1))
            h_factor = 0.3 if _is_broad_pattern(spec.pattern) else 1.0
            for line in lines:
                for m in regex_obj.finditer(line):
                    matched = m.group(0)
                    if not matched:
                        continue
                    key = (spec.pattern, _normalize_match(matched))
                    if key in seen:
                        continue
                    seen.add(key)
                    score = (
                        spec.weight
                        * idf
                        * _quality(matched, line)
                        * _g_factor(matched)
                        * h_factor
                    )
                    scored.append(
                        (score, {"pattern": spec.pattern, "match": matched[:80]})
                    )

        if not scored:
            return []

        # Saturated TF: keep the K strongest unique matches.
        scored.sort(key=lambda t: t[0], reverse=True)
        capped = scored[: cfg.regex_dedup_cap]
        # One RawHit per kept match so ``aggregate_per_page`` applies the
        # uniform sum/sqrt(N+1) within-channel formula.
        return [
            RawHit(file_id=page.file_id, page_id=page.page_id, score=s, evidence=ev)
            for s, ev in capped
        ]
