"""Emit EvidenceFS — a shell-operable evidence filesystem — from the
exact-offset segmentation of each document's ``combined.md``.

The ingestion pipeline produces a relation-free Tri-Graph (surface /
sentence / passage anchors, no relation extraction, no entity merge): the
in-memory NER caches map passage-hash → surfaces and sentence-text →
surfaces. This module re-materializes that graph as ordinary files an agent
can ``grep`` / ``awk`` / ``join`` / ``sed`` over, with each document kept as
its original ``combined.md`` (copied verbatim) so the agent reads
source context to interpret meaning.

Unlike the earlier standalone compiler — which re-derived byte/line offsets
by anchoring stored verbatim text back into ``combined.md`` — every passage
and sentence span here comes straight from :func:`segment_combined_md`, which
captures offsets at cut time. The spans are exact by construction; nothing in
this module re-runs NER, an embedding model, or a substring ``find`` for
positions.

Layout produced (v1, lexical / relation-free; no semantic layer):

    evidence_fs/
      README.md  EXAMPLES.md  manifest.json
      documents/<doc_id>/combined.md          (verbatim copy of the OCR source)
      documents/<doc_id>/pages/page_NNNN.md   (sliced on the page marker)
      nodes/   documents.tsv passages.tsv sentences.tsv surfaces.tsv mentions.tsv
      edges/   surface_sentence.tsv sentence_surface.tsv surface_passage.tsv
               passage_surface.tsv passage_sentence.tsv sentence_passage.tsv
               passage_next.tsv surface_surface_sentence.tsv
      views/   *.sorted.tsv (join-ready) + surface_index / *_text / high_df_surfaces

The shell ABI (``find_surface`` / ``expand_surface`` / ``bridge_surfaces`` /
``show_passage`` / ``show_sentence`` / ``grep_passages``) is NOT emitted here:
it lives in the repo-side ``agent_scripts/`` folder so a corpus rebuild never
rewrites the agent's tooling. The README points at it.
"""
import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config.settings import page_assets_path
from ingestion.index.linear_rag.segment import PAGE_MARKER, segment_combined_md

logger = logging.getLogger(__name__)

# Tunables. Hard-coded here is a temporary measure; these belong in the web
# config centre so they can be injected per build.  # TODO admin panel
MAX_COOC_SURFACES = 12      # skip sentence co-occurrence on >N-surface sentences
HIGH_DF_QUANTILE = 0.95     # flag (never delete) surfaces above this passage-df
PREVIEW_CHARS = 200         # passage text_preview length in the shell view


# --------------------------------------------------------------------------- #
# TSV / text hygiene — tab and newline would break awk/join, so collapse them.
# --------------------------------------------------------------------------- #
def _clean(value) -> str:
    return re.sub(r"[\t\r\n]+", " ", str(value)).strip()


def _write_tsv(path: Path, header, rows) -> int:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(_clean(c) for c in row) + "\n")
    return len(rows)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# In-memory node records.
# --------------------------------------------------------------------------- #
@dataclass
class _Doc:
    doc_id: str
    file_id: str
    title: str
    source_path: str
    md: str
    n_pages: int
    n_lines: int
    sha256: str


@dataclass
class _Passage:
    passage_id: str
    doc_id: str
    order: int
    page: str
    text: str
    start_line: int
    end_line: int
    byte_start: int
    byte_end: int
    sha256: str


@dataclass
class _Sentence:
    sentence_id: str
    passage_id: str
    doc_id: str
    order: int
    text: str
    start_line: int
    end_line: int
    start_char: int
    end_char: int


@dataclass
class _Surface:
    surface_id: str
    surface_norm: str
    surface_raw: str = ""
    mention_count: int = 0
    sentence_df: int = 0
    passage_df: int = 0
    doc_df: int = 0
    raw_votes: dict = field(default_factory=lambda: defaultdict(int))


# --------------------------------------------------------------------------- #
# Compiler — same node/edge/view shapes as the standalone materializer, but
# every offset comes from the exact segmentation rather than anchoring.
# --------------------------------------------------------------------------- #
class _EvidenceFSCompiler:
    def __init__(
        self,
        *,
        file_ids,
        read_md: Callable[[str], str],
        passage_to_entities: dict,
        sentence_to_entities: dict,
        hash_for: Callable[[str], str],
        corpus_root: Path,
        surface_to_label: dict | None = None,
    ):
        self.file_ids = list(file_ids)
        self.read_md = read_md
        self.p2e = passage_to_entities
        self.s2e = sentence_to_entities
        self.hash_for = hash_for
        self.corpus_root = corpus_root
        # norm_surface -> GLiNER label, for surfaces.tsv ``ner_label``.
        self.surface_to_label = dict(surface_to_label or {})

        self.docs: dict[str, _Doc] = {}
        self.passages: list[_Passage] = []
        self.sentences: list[_Sentence] = []
        self.surfaces: dict[str, _Surface] = {}   # norm -> _Surface
        self.mentions: list[tuple] = []

        # edges
        self.surf_sent: list[tuple] = []                           # (norm, sid, count)
        self.surf_pass_count: dict[tuple, int] = defaultdict(int)  # (norm, pid)->cnt
        self.cooc: list[tuple] = []          # (norm_a, norm_b, sid, pid, did, degree)
        self.passage_next_pairs: list[tuple] = []

        # stats for the manifest / honest reporting
        self.stats: dict[str, int] = defaultdict(int)

    # -- node construction ------------------------------------------------- #
    def _surface(self, norm: str) -> _Surface:
        sf = self.surfaces.get(norm)
        if sf is None:
            sf = _Surface(surface_id="", surface_norm=norm)  # id assigned later
            self.surfaces[norm] = sf
        return sf

    def _doc_meta(self, file_id: str) -> tuple[str, str, int]:
        """(title, source_path, n_pages) from the corpus ``meta.json`` if present."""
        title, source_path, n_pages = file_id, file_id, 0
        meta_path = self.corpus_root / file_id / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text("utf-8"))
            except Exception:
                return title, source_path, n_pages
            source_path = meta.get("source_path", file_id)
            title = Path(source_path).stem or file_id
            n_pages = int(meta.get("total_pages", 0) or 0)
        return title, source_path, n_pages

    def build(self) -> None:
        p_counter = s_counter = 0
        for i, file_id in enumerate(sorted(self.file_ids), 1):
            md = self.read_md(file_id)
            title, source_path, n_pages = self._doc_meta(file_id)
            doc = _Doc(
                doc_id=f"d_{i:04d}", file_id=file_id, title=title,
                source_path=source_path, md=md,
                n_pages=n_pages or (md.count(PAGE_MARKER) + 1),
                n_lines=md.count("\n") + 1, sha256=_sha256(md),
            )
            self.docs[file_id] = doc

            prev_pid = None
            for span in segment_combined_md(md):
                p_counter += 1
                pid = f"p_{p_counter:06d}"
                self.passages.append(_Passage(
                    passage_id=pid, doc_id=doc.doc_id, order=p_counter,
                    page=str(span.page_number), text=span.text,
                    start_line=span.start_line, end_line=span.end_line,
                    byte_start=span.start_char, byte_end=span.end_char,
                    sha256=_sha256(span.text),
                ))
                if prev_pid is not None:
                    self.passage_next_pairs.append((prev_pid, pid))
                prev_pid = pid

                # surface↔passage from the passage-level NER set (authoritative)
                for norm in self.p2e.get(self.hash_for(span.text), []):
                    self._surface(norm)
                    self.surf_pass_count[(norm, pid)] += 1

                for sentence in span.sentences:
                    s_counter += 1
                    sid = f"s_{s_counter:06d}"
                    self.sentences.append(_Sentence(
                        sentence_id=sid, passage_id=pid, doc_id=doc.doc_id,
                        order=s_counter, text=sentence.text,
                        start_line=sentence.start_line, end_line=sentence.end_line,
                        start_char=sentence.start_char, end_char=sentence.end_char,
                    ))
                    self._link_sentence(sid, pid, doc.doc_id, sentence.text)

        self._fold_sentence_counts_into_passages()
        self._finalize_surfaces()

    def _link_sentence(self, sid: str, pid: str, did: str, stext: str) -> None:
        """surface↔sentence edges, mentions and co-occurrence for one sentence."""
        surfs = self.s2e.get(stext, [])
        present = []
        # Per-mention offset recovery scans the sentence once per surface. Cap
        # it at MAX_COOC_SURFACES so a pathological flattened OCR-table row
        # (one very long "sentence" with many surfaces) stays O(cap * len), not
        # O(surfaces * len); over-cap sentences keep their surface↔sentence
        # edges but record sentence-level provenance only (-1 char offsets).
        recover = len(surfs) <= MAX_COOC_SURFACES
        for norm in surfs:
            sf = self._surface(norm)
            # count raw occurrences (case-insensitive); records original casing
            hits = (list(re.finditer(re.escape(norm), stext, re.IGNORECASE))
                    if recover else [])
            count = len(hits) or 1  # CJK t2s-normalized surfaces may not match raw
            self.surf_sent.append((norm, sid, count))
            present.append(norm)
            if hits:
                for h in hits:
                    raw = stext[h.start():h.end()]
                    sf.raw_votes[raw] += 1
                    self.mentions.append((sf, raw, did, pid, sid, h.start(), h.end()))
            else:
                self.stats["mention_offset_miss"] += 1
                self.mentions.append((sf, norm, did, pid, sid, -1, -1))

        # relation-free co-occurrence bridge (sentence-level, degree-capped)
        uniq = sorted(set(present))
        if 2 <= len(uniq) <= MAX_COOC_SURFACES:
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    self.cooc.append((uniq[i], uniq[j], sid, pid, did, len(uniq)))
        elif len(uniq) > MAX_COOC_SURFACES:
            self.stats["cooc_sentence_skipped"] += 1

    def _fold_sentence_counts_into_passages(self) -> None:
        """Fold sentence-derived surface counts into the surface→passage weights
        (stronger signal where present), mirroring the standalone compiler."""
        sent_pass = {s.sentence_id: s.passage_id for s in self.sentences}
        agg: dict[tuple, int] = defaultdict(int)
        for norm, sid, cnt in self.surf_sent:
            agg[(norm, sent_pass[sid])] += cnt
        for key, cnt in agg.items():
            if key in self.surf_pass_count:
                self.surf_pass_count[key] = max(self.surf_pass_count[key], cnt)
            else:
                # a sentence-level surface whose passage-level NER set missed it
                self.surf_pass_count[key] = cnt

    def _finalize_surfaces(self) -> None:
        for i, norm in enumerate(sorted(self.surfaces), 1):
            self.surfaces[norm].surface_id = f"sf_{i:06d}"
        pass_doc = {p.passage_id: p.doc_id for p in self.passages}
        s_df: dict[str, set] = defaultdict(set)
        for norm, sid, _ in self.surf_sent:
            s_df[norm].add(sid)
        p_df: dict[str, set] = defaultdict(set)
        d_df: dict[str, set] = defaultdict(set)
        mc: dict[str, int] = defaultdict(int)
        for (norm, pid), cnt in self.surf_pass_count.items():
            p_df[norm].add(pid)
            d_df[norm].add(pass_doc[pid])
            mc[norm] += cnt
        for norm, sf in self.surfaces.items():
            sf.sentence_df = len(s_df[norm])
            sf.passage_df = len(p_df[norm])
            sf.doc_df = len(d_df[norm])
            sf.mention_count = mc[norm] or sf.sentence_df
            sf.surface_raw = (max(sf.raw_votes, key=sf.raw_votes.get)
                              if sf.raw_votes else norm)

    # -- emit -------------------------------------------------------------- #
    def write(self, out: Path) -> dict:
        for sub in ("nodes", "edges", "views", "documents"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        sid = lambda norm: self.surfaces[norm].surface_id

        self._write_documents(out)
        self._write_nodes(out)
        surf_sorted = sorted(self.surfaces.values(), key=lambda s: s.surface_id)
        ss, sp_rows = self._write_edges(out, sid)
        self._write_views(out, surf_sorted, sp_rows, ss)
        return self._write_manifest(out)

    def _write_documents(self, out: Path) -> None:
        doc_rows = []
        for doc in self.docs.values():
            dest = out / "documents" / doc.doc_id
            dest.mkdir(parents=True, exist_ok=True)
            # Copy combined.md as a REAL file (it is small, ~0.6 MB). A symlink
            # would save negligible space — the heavy raw OCR assets are never
            # copied — but ``rg``/``grep -r`` skip symlinked files by default,
            # which silently breaks the agent's primary locator (recursive
            # search). A real file keeps the FS self-contained and greppable.
            dest_md = dest / "combined.md"
            if dest_md.is_symlink() or dest_md.exists():
                dest_md.unlink()
            shutil.copyfile(self.corpus_root / doc.file_id / "combined.md", dest_md)
            # paginated reads: slice combined.md on the page marker.
            self._write_pages(dest / "pages", doc.md)
            # multimodal: copy each page's rendered image next to its .md.
            self._copy_page_images(dest / "pages", doc.file_id)
            doc_rows.append((doc.doc_id, doc.title, doc.file_id,
                             f"documents/{doc.doc_id}/combined.md",
                             doc.n_pages, doc.n_lines, doc.sha256))
        _write_tsv(out / "nodes" / "documents.tsv",
                   ["doc_id", "title", "source_path", "combined_path",
                    "n_pages", "n_lines", "sha256"], doc_rows)

    @staticmethod
    def _write_pages(pages_dir: Path, md: str) -> None:
        pages_dir.mkdir(parents=True, exist_ok=True)
        for n, page in enumerate(md.split(PAGE_MARKER), 1):
            (pages_dir / f"page_{n:04d}.md").write_text(page, encoding="utf-8")

    def _copy_page_images(self, pages_dir: Path, file_id: str) -> None:
        """Copy each page's rendered image next to its ``page_NNNN.md`` as
        ``page_NNNN.<ext>`` so a vision agent can ``view_page`` it.

        Real copies, not symlinks — the FS stays self-contained and an image
        path never escapes to the raw OCR tree (and, as with ``combined.md``,
        ``rg``/``grep`` skip symlinks). Page assets are emitted in reading order,
        the same order ``combined.md`` was assembled and sliced, so enumerate
        index == page_number. Silently a no-op for a text-only corpus (no
        manifest, or a page with no rendered image on disk)."""
        manifest = page_assets_path(file_id)
        if not manifest.is_file():
            return
        try:
            assets = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        file_root = self.corpus_root / file_id
        for n, asset in enumerate(assets, 1):
            rel_img = asset.get("page_image_path")
            if not rel_img:
                continue
            src = file_root / rel_img
            if not src.is_file():
                continue
            dest = pages_dir / f"page_{n:04d}{Path(rel_img).suffix.lower()}"
            if not dest.exists():
                shutil.copyfile(src, dest)

    def _write_nodes(self, out: Path) -> None:
        _write_tsv(out / "nodes" / "passages.tsv",
                   ["passage_id", "doc_id", "passage_order", "page_start",
                    "page_end", "start_line", "end_line", "byte_start",
                    "byte_end", "sha256"],
                   [(p.passage_id, p.doc_id, p.order, p.page, p.page,
                     p.start_line, p.end_line, p.byte_start, p.byte_end,
                     p.sha256) for p in self.passages])

        _write_tsv(out / "nodes" / "sentences.tsv",
                   ["sentence_id", "passage_id", "doc_id", "sentence_order",
                    "start_line", "end_line", "start_char", "end_char", "text"],
                   [(s.sentence_id, s.passage_id, s.doc_id, s.order,
                     s.start_line, s.end_line, s.start_char, s.end_char,
                     s.text) for s in self.sentences])

        surf_sorted = sorted(self.surfaces.values(), key=lambda s: s.surface_id)
        _write_tsv(out / "nodes" / "surfaces.tsv",
                   ["surface_id", "surface_raw", "surface_norm", "ner_label",
                    "mention_count", "sentence_df", "passage_df", "doc_df"],
                   [(s.surface_id, s.surface_raw, s.surface_norm,
                     self.surface_to_label.get(s.surface_norm, ""),
                     s.mention_count, s.sentence_df, s.passage_df, s.doc_df)
                    for s in surf_sorted])

        _write_tsv(out / "nodes" / "mentions.tsv",
                   ["mention_id", "surface_id", "surface_raw", "ner_label",
                    "doc_id", "passage_id", "sentence_id", "start_char",
                    "end_char"],
                   [(f"m_{i:07d}", sf.surface_id, raw, "", did, pid, s, a, b)
                    for i, (sf, raw, did, pid, s, a, b)
                    in enumerate(self.mentions, 1)])

    def _write_edges(self, out: Path, sid):
        ss = sorted(((sid(n), s, c) for n, s, c in self.surf_sent))
        _write_tsv(out / "edges" / "surface_sentence.tsv",
                   ["surface_id", "sentence_id", "mention_count"], ss)
        _write_tsv(out / "edges" / "sentence_surface.tsv",
                   ["sentence_id", "surface_id", "mention_count"],
                   sorted((s, sf, c) for sf, s, c in ss))

        sp_total: dict[str, int] = defaultdict(int)
        for (norm, pid), cnt in self.surf_pass_count.items():
            sp_total[pid] += cnt
        sp_rows = sorted(
            (sid(norm), pid, cnt,
             round(cnt / sp_total[pid], 6) if sp_total[pid] else 0.0)
            for (norm, pid), cnt in self.surf_pass_count.items())
        _write_tsv(out / "edges" / "surface_passage.tsv",
                   ["surface_id", "passage_id", "mention_count",
                    "weight_in_passage"], sp_rows)
        _write_tsv(out / "edges" / "passage_surface.tsv",
                   ["passage_id", "surface_id", "mention_count",
                    "weight_in_passage"],
                   sorted((p, s, c, w) for s, p, c, w in sp_rows))

        _write_tsv(out / "edges" / "sentence_passage.tsv",
                   ["sentence_id", "passage_id", "sentence_order"],
                   [(s.sentence_id, s.passage_id, s.order) for s in self.sentences])
        _write_tsv(out / "edges" / "passage_sentence.tsv",
                   ["passage_id", "sentence_id", "sentence_order"],
                   sorted((s.passage_id, s.sentence_id, s.order)
                          for s in self.sentences))

        next_rows = []
        for a, b in self.passage_next_pairs:
            next_rows.append((a, b, "next_in_doc"))
            next_rows.append((b, a, "prev_in_doc"))
        _write_tsv(out / "edges" / "passage_next.tsv",
                   ["passage_id", "next_passage_id", "relation"],
                   sorted(next_rows))

        _write_tsv(out / "edges" / "surface_surface_sentence.tsv",
                   ["surface_a", "surface_b", "sentence_id", "passage_id",
                    "doc_id", "sent_degree"],
                   sorted((sid(a), sid(b), s, p, d, deg)
                          for a, b, s, p, d, deg in self.cooc))
        return ss, sp_rows

    def _write_views(self, out, surf_sorted, sp_rows, ss) -> None:
        v = out / "views"
        _write_tsv(v / "surface_index.tsv",
                   ["surface_norm", "surface_id", "surface_raw", "ner_label",
                    "mention_count", "passage_df"],
                   sorted((s.surface_norm, s.surface_id, s.surface_raw, "",
                           s.mention_count, s.passage_df) for s in surf_sorted))
        _write_tsv(v / "surface_passages.sorted.tsv",
                   ["surface_id", "passage_id", "mention_count",
                    "weight_in_passage"], sp_rows)
        _write_tsv(v / "passage_surfaces.sorted.tsv",
                   ["passage_id", "surface_id", "mention_count",
                    "weight_in_passage"],
                   sorted((p, s, c, w) for s, p, c, w in sp_rows))
        _write_tsv(v / "surface_sentences.sorted.tsv",
                   ["surface_id", "sentence_id", "mention_count"], ss)
        _write_tsv(v / "sentence_surfaces.sorted.tsv",
                   ["sentence_id", "surface_id", "mention_count"],
                   sorted((s, sf, c) for sf, s, c in ss))
        _write_tsv(v / "passage_text.tsv",
                   ["passage_id", "doc_id", "start_line", "end_line",
                    "text_preview"],
                   [(p.passage_id, p.doc_id, p.start_line, p.end_line,
                     p.text[:PREVIEW_CHARS]) for p in self.passages])
        _write_tsv(v / "sentence_text.tsv",
                   ["sentence_id", "passage_id", "doc_id", "text"],
                   [(s.sentence_id, s.passage_id, s.doc_id, s.text)
                    for s in self.sentences])
        dfs = sorted(s.passage_df for s in surf_sorted)
        cutoff = dfs[int(len(dfs) * HIGH_DF_QUANTILE)] if dfs else 0
        _write_tsv(v / "high_df_surfaces.tsv",
                   ["surface_id", "surface_norm", "passage_df"],
                   sorted(((s.surface_id, s.surface_norm, s.passage_df)
                           for s in surf_sorted if s.passage_df >= cutoff and cutoff),
                          key=lambda r: -r[2]))

    def _write_manifest(self, out: Path) -> dict:
        manifest = {
            "name": "evidencefs",
            "version": "v1-lexical",
            "relation_free": True,
            "entity_resolution": False,
            "counts": {
                "documents": len(self.docs),
                "passages": len(self.passages),
                "sentences": len(self.sentences),
                "surfaces": len(self.surfaces),
                "mentions": len(self.mentions),
                "surface_sentence_edges": len(self.surf_sent),
                "surface_passage_edges": len(self.surf_pass_count),
                "cooccurrence_edges": len(self.cooc),
            },
            "reconstruction": dict(self.stats),
            "tunables": {
                "MAX_COOC_SURFACES": MAX_COOC_SURFACES,
                "HIGH_DF_QUANTILE": HIGH_DF_QUANTILE,
                "PREVIEW_CHARS": PREVIEW_CHARS,
            },
            "notes": [
                "Passage / sentence spans come from exact-offset segmentation of "
                "combined.md (segment_combined_md); md[start_char:end_char] == text "
                "by construction — no anchoring / find.",
                "Surfaces are normalized anchors (OpenCC t2s + NFKC + lower); "
                "no entity resolution / merge is performed.",
                "ner_label carries the GLiNER label of the surface's first "
                "occurrence (surfaces.tsv); blank where the cache has no label.",
                "Mention char offsets are sentence-local (within the sentence "
                "text), best-effort: -1 where the t2s-normalized surface does not "
                "match raw CJK. Sentence-level provenance is always exact.",
                "Shell ABI (find_surface / expand_surface / bridge_surfaces / "
                "show_passage / show_sentence / grep_passages) is NOT bundled here "
                "— it ships in the repo-side agent_scripts/ folder.",
            ],
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        (out / "README.md").write_text(_README, "utf-8")
        (out / "EXAMPLES.md").write_text(_EXAMPLES, "utf-8")
        return manifest


# --------------------------------------------------------------------------- #
# Pure entry point + LinearRAG adapter.
# --------------------------------------------------------------------------- #
def build_evidence_fs(
    *,
    file_ids,
    read_md: Callable[[str], str],
    passage_to_entities: dict,
    sentence_to_entities: dict,
    hash_for: Callable[[str], str],
    out_dir: Path,
    corpus_root: Path,
    surface_to_label: dict | None = None,
) -> dict:
    """Compile EvidenceFS from exact segmentation spans. Pure / unit-testable.

    ``read_md(file_id) -> str`` returns the document's combined.md text.
    ``passage_to_entities`` maps passage-hash → [normalized surface];
    ``sentence_to_entities`` maps sentence-text → [normalized surface];
    ``hash_for(text) -> str`` is the passage-hash function. ``surface_to_label``
    maps normalized surface → GLiNER label, populating ``surfaces.tsv``'s
    ``ner_label`` column (absent / unknown surfaces stay ``""``). Per file,
    segment combined.md into exact passage / sentence spans, attach surfaces by
    hash / text lookup, and emit the EvidenceFS layout (no ``scripts/``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compiler = _EvidenceFSCompiler(
        file_ids=file_ids,
        read_md=read_md,
        passage_to_entities=passage_to_entities,
        sentence_to_entities=sentence_to_entities,
        hash_for=hash_for,
        corpus_root=Path(corpus_root),
        surface_to_label=surface_to_label,
    )
    compiler.build()
    return compiler.write(out_dir)


def write_evidence_fs(lr, out_dir: Path, corpus_root: Path) -> dict:
    """Thin LinearRAG adapter: pull the file_id set + NER caches + hash fn off
    ``lr`` and drive :func:`build_evidence_fs`.

    ``file_ids`` are the unique, sorted ``file_id`` meta values of the passage
    embedding store. ``read_md`` reads ``corpus_root/<file_id>/combined.md``.
    """
    corpus_root = Path(corpus_root)
    file_ids = sorted(
        {fid for fid in lr.passage_embedding_store.meta_column("file_id")
         if fid is not None}
    )

    def read_md(file_id: str) -> str:
        return (corpus_root / file_id / "combined.md").read_text("utf-8")

    return build_evidence_fs(
        file_ids=file_ids,
        read_md=read_md,
        passage_to_entities=lr._passage_to_entities,
        sentence_to_entities=lr._sentence_to_entities,
        hash_for=lr.passage_embedding_store.hash_for,
        out_dir=out_dir,
        corpus_root=corpus_root,
        surface_to_label=getattr(lr, "_entity_to_label", {}),
    )


# --------------------------------------------------------------------------- #
# Static assets: schema doc + recipes (the shell ABI lives in agent_scripts/).
# --------------------------------------------------------------------------- #
_README = """\
# EvidenceFS — a shell-operable evidence filesystem

A corpus compiled into a relation-free Tri-Graph you operate with ordinary
shell programs. Surfaces are anchors, sentences are bridges, passages carry the
evidence, documents hold the original text. No entity merge, no relation
extraction: relation meaning is recovered at runtime by reading the source.

## Node identity
- `d_*` document, `p_*` passage, `s_*` sentence, `sf_*` surface, `m_*` mention.

## Files
- `documents/<doc_id>/combined.md` — original OCR'd text (a verbatim copy,
  greppable; read with `sed -n`). `documents/<doc_id>/pages/page_NNNN.md`
  are per-page slices for paginated reads.
- `nodes/` — one row per node. `passages.tsv`/`sentences.tsv` carry
  `start_line`/`end_line` (and exact char spans) into `combined.md`.
  `surfaces.tsv` is a normalized anchor (not a resolved entity); `mentions.tsv`
  is per-occurrence provenance.
- `edges/` — `surface_sentence` / `surface_passage` / `passage_sentence` /
  `passage_next`, both directions materialized, plus
  `surface_surface_sentence` (relation-free co-occurrence; NOT a relation).
- `views/` — `*.sorted.tsv` are pre-sorted on their first column for `join`;
  `surface_index.tsv` is sorted by `surface_norm` for `rg`; `*_text.tsv` give
  greppable text; `high_df_surfaces.tsv` flags corpus-frequent anchors.

## Programs (on your PATH; run by bare name from this directory)
Lexical (string + graph-file ops):
- `find_surface TERM` → surfaces (`sf_*`) matching a term.
- `expand_surface sf_*` → its passages (+ doc + line range).
- `bridge_surfaces sf_*` → surfaces sharing a sentence with it (co-occurrence,
  not a relation; read the example sentence to recover the relation).
- `show_passage p_* [--context N]` → the passage's source-text window.
- `show_sentence s_*` → a sentence + its passage/doc + window.
- `grep_passages 'PATTERN' p_* …` → grep inside specific passages only.
Embedding-ranked (graph agent only; slower — use when exact search isn't enough):
- `seed_surfaces --query "..."` → surfaces nearest in meaning to the query.
- `rank_passages --query "..."` → graph/PPR page ranking.
- `search_dense --query "..."` → dense (embedding) page ranking.
- `candidate_aliases sf_*` → surfaces nearest a given one (possible synonyms).
- `semantic_bridge sf_* --query "..."` → surfaces co-mentioned with it in the
  query-most-relevant sentences (+ the bridging `s_*`).

## How to search
1. `find_surface TERM` (or `seed_surfaces --query "..."`) → `sf_*`.
2. `expand_surface sf_*` → its passages (+ line ranges).
3. `show_passage p_* --context N` → read original text.
4. multi-hop: `bridge_surfaces sf_*` → surfaces sharing a sentence.

Read `EXAMPLES.md` for full recipes.
"""

_EXAMPLES = """\
# EvidenceFS recipes

All tables are tab-separated. `T=$'\\t'`. The programs below are on your PATH;
run them by bare name from this directory (the FS root).

## 1. find_surface → passages
```bash
find_surface "surrender"            # -> sf_000231 ...
expand_surface sf_000231            # -> passages + line ranges
```

## 2. bridge two surfaces (multi-hop via shared sentence)
```bash
bridge_surfaces sf_000046           # surfaces co-occurring with it
# then read a bridging sentence:
grep -P "^sf_000046\\t" edges/surface_surface_sentence.tsv | head
```

## 3. inspect a passage window
```bash
show_passage p_000123 --context 3
```

## 4. list every mention of a surface
```bash
awk -F"\\t" -v s=sf_000046 '$2==s' edges/sentence_surface.tsv \\
  | cut -f1 | sort -u                            # sentence ids
grep -P "^sf_000046\\t" nodes/mentions.tsv       # raw mention rows w/ offsets
```

## 5. trace a sentence to its document
```bash
show_sentence s_009812              # text + passage + doc + window
```

## 6. local grep inside candidate passages (avoid full-corpus scan)
```bash
grep_passages "not cover|excluded|免责|不赔" p_000123 p_000124
```

## 7. embedding-ranked retrieval (graph agent only; when exact search isn't enough)
```bash
seed_surfaces --query "ways to pay the premium"   # synonym -> sf_* to expand
rank_passages --query "..." --top-k 10            # graph/PPR page ranking
search_dense  --query "..." --top-k 10            # dense page ranking
# both print: rank  doc_id  page  score  page_file  preview  -> read page_file to confirm
```
"""
