"""Unit-aware read.

One tool, three observation flavours, all backed by the same scope
filters as the search tools (file_ids / section_ids / page_range)
plus an optional ``unit_ids`` for direct addressing:

* ``unit_type="page"``   → ``PageReadObservation`` with markdown + tables
                           + (page-only) optional VLM summary, AND a
                           ``children`` block listing the passage_ids
                           and table_row_ids on each page so the agent
                           can drill into fine-grained evidence without
                           a second discovery step.
* ``unit_type="passage"`` → ``PassageReadObservation`` carrying just
                           the cleaned passage text + parent section.
* ``unit_type="table_row"`` → ``TableRowReadObservation`` carrying
                           html + flattened text + parent section.

Plant only mints WitnessClaim / ValueClaim from a *ReadObservation
whose ``unit_type`` matches the obligation's. Returned units are
sorted in document order; the agent can rely on that for
"contiguous, in-order" reading even across multiple unit_ids.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from config.shared import shared_tiktoken_encoder

from agentic.tools.acquisition._common import Scope, err, ok, parse_scope
from agentic.tools.acquisition._vlm import VlmReader, default_vlm_reader
from agentic.tools.base import BaseTool
from storage.inventory_store import InventoryStore
from storage.page_store import PageAsset, PageStore, make_global_id

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from rag.channels.graph_ppr import GraphPPRChannel


logger = logging.getLogger(__name__)


_VALID_UNIT_TYPES = {"page", "passage", "table_row"}
_VALID_MODES = {"text", "text_with_img"}
_DEFAULT_VLM_PARALLELISM = 4

# Per-page annotation caps for graph-aware reads (kept low; the agent
# benefits from a tight key-entity list and a couple of neighbour
# pointers, not a corpus dump).
_READ_TOP_ENTITIES = 8
_READ_TOP_NEIGHBOURS = 3
# Cluster-size guard for the neighbour computation: pages whose top
# entities all belong to giant hub clusters (e.g. one Pew Research
# Center entity that incidentally appears on 200 pages) would dominate
# the neighbour list and crowd out genuine cross-references. Skip
# entities sitting in clusters above this size when picking neighbours.
_READ_NEIGHBOUR_HUB_CUTOFF = 40


class ReadTool(BaseTool):
    def __init__(
        self,
        page_store: PageStore,
        inventory: InventoryStore,
        vlm_reader: Optional[VlmReader] = None,
        vlm_parallelism: int = _DEFAULT_VLM_PARALLELISM,
        graph_channel: Optional["GraphPPRChannel"] = None,
    ) -> None:
        self.page_store = page_store
        self.inventory = inventory
        self.vlm_reader = vlm_reader if vlm_reader is not None else default_vlm_reader()
        self.vlm_parallelism = max(1, int(vlm_parallelism))
        self.tokenizer = shared_tiktoken_encoder("gpt-4o")
        # When wired (graph_agent path), each page-level read attaches
        # two graph-derived annotations: ``entities`` (top mention-
        # weighted entities on this page; key-entity table prompts the
        # reader to ground synthesis) and ``neighbour_pages`` (other
        # pages sharing this page's anchor entities; lets the agent
        # plan the next read without a fresh PPR query). None means
        # no graph wiring (regex / web / default agents), in which
        # case the original page observation shape is preserved.
        self.graph_channel = graph_channel

    @property
    def name(self) -> str:
        return "read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read",
                "description": (
                    "Read units (pages / passages / table_rows) verbatim, in "
                    "document order. You MUST provide one of:\n"
                    "- `unit_ids=['file_id/p_NNNN', ...]` (precise — preferred), or\n"
                    "- `file_ids=[...] + page_range=[start, end]` (inclusive range), or\n"
                    "- `file_ids=[...] + section_ids=['<file_id>:sec_NNN']`.\n"
                    "Bare `file_ids` is rejected (would overflow context). "
                    "Pages include table blocks and `children.passage_ids` / "
                    "`.table_row_ids` for drill-down."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "unit_type": {
                            "type": "string",
                            "enum": sorted(_VALID_UNIT_TYPES),
                            "default": "page",
                        },
                        "unit_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Precise unit ids; pages use 'FILE_ID/PAGE_ID' (separator '/', not '#').",
                        },
                        "file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File allow-list (only when unit_ids omitted).",
                        },
                        "section_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Section allow-list ('<file_id>:sec_NNN' from `toc`).",
                        },
                        "page_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Inclusive [start, end] page-number filter.",
                        },
                        "mode": {
                            "type": "string",
                            "enum": sorted(_VALID_MODES),
                            "default": "text",
                            "description": "text | text_with_img (VLM read of rendered page image, page only).",
                        },
                    },
                    "required": [],
                },
            },
        }

    # ---------------------------------------------------------- execute

    def execute(
        self,
        context: "AgentContext",
        unit_type: str = "page",
        unit_ids: Optional[List[str]] = None,
        file_ids: Optional[List[str]] = None,
        section_ids: Optional[List[str]] = None,
        page_range: Optional[List[int]] = None,
        mode: str = "text",
    ) -> Tuple[str, Dict[str, Any]]:
        if unit_type not in _VALID_UNIT_TYPES:
            return _bad_arg(f"unit_type must be one of {sorted(_VALID_UNIT_TYPES)}")
        if mode not in _VALID_MODES:
            return _bad_arg(f"mode must be one of {sorted(_VALID_MODES)}")

        if unit_ids:
            target_ids = []
            for u in unit_ids:
                s = str(u).strip()
                if not s:
                    continue
                # Defensive: some LLMs emit a markdown-anchor '#' where the
                # canonical id uses '/' (e.g. 'db_en_0579#p_0011' instead of
                # 'db_en_0579/p_0011'). Treat the first '#' as the file/page
                # separator iff no '/' is present, preserving any legitimate
                # passage/row '#suffix' on already-canonical ids.
                if "/" not in s and "#" in s:
                    s = s.replace("#", "/", 1)
                target_ids.append(s)
            if not target_ids:
                return _bad_arg(
                    "unit_ids was provided but every entry was empty after "
                    "trimming. Pass at least one valid unit_id such as "
                    "'db_en_X/p_0001'."
                )
        else:
            has_page_spec = (page_range is not None) or bool(section_ids)
            if not has_page_spec:
                return _bad_arg(
                    "read requires a page-spec — bare file_ids is rejected "
                    "(would dump entire documents and overflow the LLM "
                    "context). Pass one of:\n"
                    "  (a) unit_ids=['<file_id>/<page_id>', ...]  e.g. "
                    "['db_en_0579/p_0001','db_en_0579/p_0003']\n"
                    "  (b) file_ids=['<file_id>'] + page_range=[start, end]  "
                    "(inclusive, e.g. page_range=[1,5])\n"
                    "  (c) file_ids=['<file_id>'] + section_ids=['<sec>', ...]"
                )
            scope, scope_err = parse_scope(file_ids, page_range, section_ids, inventory=self.inventory)
            if scope_err is not None:
                return _bad_arg(scope_err)
            target_ids = sorted(self.inventory.units(
                unit_type,
                file_ids=list(scope.file_ids) if scope.file_ids else None,
                section_ids=list(scope.section_ids) if scope.section_ids else None,
            ))
            if scope.page_range is not None:
                target_ids = self._filter_by_page_range(target_ids, unit_type, scope.page_range)

        if not target_ids:
            return _empty(unit_type)

        if unit_type == "page":
            return self._read_pages(context, target_ids, mode=mode)
        if unit_type == "passage":
            return self._read_passages(context, target_ids)
        return self._read_table_rows(context, target_ids)

    # ---------------------------------------------------------- page mode

    def _read_pages(
        self,
        context: "AgentContext",
        global_ids: List[str],
        *,
        mode: str,
    ) -> Tuple[str, Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        new_pages, already_read, not_found = [], [], []
        vlm_targets: List[Tuple[int, PageAsset]] = []
        total_tokens = 0

        for gid in global_ids:
            page = self.page_store.get(gid)
            if page is None:
                not_found.append(gid)
                units.append({"unit_id": gid, "status": "not_found"})
                continue
            entry = self._page_unit_entry(page)
            if context.is_page_read(gid):
                entry["status"] = "already_read"
                already_read.append(gid)
            else:
                new_pages.append(gid)
                if mode == "text_with_img":
                    vlm_targets.append((len(units), page))
            units.append(entry)

        if vlm_targets and self.vlm_reader is not None:
            workers = min(self.vlm_parallelism, len(vlm_targets))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._vlm_one, page): idx for idx, page in vlm_targets}
                for fut, idx in futures.items():
                    try:
                        units[idx].update(fut.result())
                    except Exception as exc:
                        logger.warning("read: VLM future raised: %s", exc)
                        units[idx].update({"vlm_error": str(exc)})

        for u in units:
            if u.get("status") in {"not_found", "already_read"}:
                continue
            total_tokens += len(self.tokenizer.encode(u.get("text", "") or ""))
            if u.get("vlm_summary"):
                total_tokens += len(self.tokenizer.encode(u["vlm_summary"]))
            context.mark_page_as_read(u["unit_id"])

        units.sort(key=_page_unit_sort_key)

        context.add_retrieval_log(
            tool_name="read",
            tokens=total_tokens,
            metadata={
                "unit_type": "page",
                "requested": len(global_ids),
                "new": len(new_pages),
                "already_read": len(already_read),
                "not_found": len(not_found),
            },
        )

        return (
            ok(
                "PageReadObservation",
                unit_type="page",
                units=units,
                summary={
                    "requested": len(global_ids),
                    "new": len(new_pages),
                    "already_read": len(already_read),
                    "not_found": len(not_found),
                },
            ),
            {
                "retrieved_tokens": total_tokens,
                "unit_type": "page",
                "new_pages_count": len(new_pages),
            },
        )

    def _page_unit_entry(self, page: PageAsset) -> Dict[str, Any]:
        passages = []
        try:
            passages = [
                p.passage_id
                for p in self.inventory.passage_store.passages_for_page(page.file_id, page.page_id)
            ]
        except Exception:
            pass
        rows = []
        try:
            rows = [
                r.table_row_id
                for r in self.inventory.table_row_store.rows_for_file(page.file_id)
                if r.page_id == page.page_id
            ]
        except Exception:
            pass
        entry: Dict[str, Any] = {
            "unit_id": page.global_id,
            "file_id": page.file_id,
            "page_id": page.page_id,
            "page_number": page.page_number,
            "text": page.text_markdown or "",
            "table_blocks": page.table_blocks,
            "image_refs": [
                {"image_id": b.get("image_id"), "path": b.get("path"), "type": b.get("type")}
                for b in page.image_blocks
            ],
            "vlm_summary": "",
            "vlm_extracted_items": [],
            "children": {
                "passage_ids": passages,
                "table_row_ids": rows,
            },
        }
        self._attach_graph_annotations(entry, page)
        return entry

    def _attach_graph_annotations(
        self, entry: Dict[str, Any], page: PageAsset
    ) -> None:
        """Add ``entities`` + ``neighbour_pages`` to a page-read entry
        when the GraphPPRChannel is wired (graph_agent factory path).

        ``entities``: top-K (surface, cluster_id, weight) on this page
        — a structured key-entity table that prompts the reader to
        ground synthesis. Multi-hop misses where the reader read all
        gold pages but answered wrong often look like the reader
        staring at a long Markdown page without the named hooks; this
        is the entity-chained reading trick from prior work.

        ``neighbour_pages``: top-K other pages sharing this page's
        specific (non-hub) anchor entities. The agent's next-step
        planning gets a direct sibling pointer without issuing a
        fresh PPR query. Hub entities (cluster_size > cutoff) are
        skipped so the neighbour list isn't dominated by corpus-wide
        mentions of e.g. organization names.

        All reverse maps (``page_meta_to_hash``, ``page_hash_to_meta``,
        ``entity_to_passages``) live on the channel and are built once
        with the channel's ``_call_lock``, then served in O(1) here —
        no per-read full scans of ``passage_store.hash_ids``.
        """
        channel = self.graph_channel
        if channel is None:
            return
        try:
            channel._build_entity_passage_indexes()
            _, m2c = channel._load_clusters_cached()
        except Exception:
            return

        key = (page.file_id, int(page.page_number) if page.page_number is not None else None)
        passage_hash = channel.page_meta_to_hash().get(key)
        if passage_hash is None:
            return

        ent_list = (channel._passage_entities or {}).get(passage_hash, [])
        if not ent_list:
            return

        ent_text = channel.entity_store.hash_id_to_text
        entities_out: List[Dict[str, Any]] = []
        for ent_hash, w in ent_list[:_READ_TOP_ENTITIES]:
            cid = m2c.get(ent_hash, ent_hash)
            ent_entry: Dict[str, Any] = {
                "surface": ent_text.get(ent_hash, ""),
                "cluster_id": cid,
                "weight": round(float(w), 4),
            }
            # Multi-hop bridge hint via the tri-graph's sentence layer.
            # Our entity graph has NO relation edges; predicates live
            # in the sentences where two entities co-occur. Surfacing
            # the top-2 sentence-co-occurring partners per entity
            # lets the agent see the bridge directly on the read
            # observation — no extra chain call needed to discover
            # which entities are "documented near" this one. Capped
            # at 2 surfaces per entity (~40 tokens/page total) to
            # keep the read observation focused.
            try:
                co_partners = channel.entity_top_co_occurring(ent_hash, top_n=2)
            except Exception:
                co_partners = []
            if co_partners:
                ent_entry["co_occurring"] = [
                    ent_text.get(other_h, "") for other_h, _ in co_partners
                    if ent_text.get(other_h, "")
                ]
            entities_out.append(ent_entry)
        entry["entities"] = entities_out

        # Neighbour pages: union of other passages sharing this page's
        # non-hub anchor entities. Score each neighbour by Σ shared-
        # entity weight; surface top K.
        entity_to_passages = channel.entity_to_passages()
        neighbour_score: Dict[str, float] = {}
        for ent_hash, w in ent_list[:_READ_TOP_ENTITIES]:
            cid = m2c.get(ent_hash, ent_hash)
            try:
                if channel.cluster_passage_count(cid) > _READ_NEIGHBOUR_HUB_CUTOFF:
                    continue
            except Exception:
                pass
            for other_ph, other_w in entity_to_passages.get(ent_hash, []):
                if other_ph == passage_hash:
                    continue
                neighbour_score[other_ph] = neighbour_score.get(other_ph, 0.0) + float(other_w)
        if not neighbour_score:
            entry["neighbour_pages"] = []
            return

        hash_to_meta = channel.page_hash_to_meta()
        ranked = sorted(neighbour_score.items(), key=lambda kv: kv[1], reverse=True)
        neighbours: List[Dict[str, Any]] = []
        for ph_other, sc in ranked[: _READ_TOP_NEIGHBOURS * 3]:
            meta = hash_to_meta.get(ph_other)
            if meta is None:
                continue
            fid_o, pn_o = meta
            if not fid_o or pn_o is None:
                continue
            # ``same_doc`` flag lets the agent see at a glance whether
            # the neighbour is an intra-doc sibling (likely an
            # unread section of the same article) or a cross-doc
            # bridge. Intra-doc siblings are the dominant path on
            # multi-page-gold questions.
            neighbours.append({
                "file_id": fid_o,
                "page_number": pn_o,
                "shared_entity_weight": round(sc, 4),
                "same_doc": fid_o == page.file_id,
            })
            if len(neighbours) >= _READ_TOP_NEIGHBOURS:
                break
        entry["neighbour_pages"] = neighbours

    def _vlm_one(self, page: PageAsset) -> Dict[str, Any]:
        if self.vlm_reader is None or not page.page_image_path:
            return {"vlm_summary": "", "vlm_error": None if page.page_image_path else "no_page_image"}
        try:
            result = self.vlm_reader(page.page_image_path, page) or {}
        except Exception as exc:
            return {"vlm_summary": "", "vlm_error": "exception", "vlm_error_message": str(exc)}
        out = {
            "vlm_summary": result.get("summary", "") or "",
            "vlm_extracted_items": result.get("items", []) or [],
        }
        if result.get("error"):
            out["vlm_error"] = result["error"]
            if result.get("error_message"):
                out["vlm_error_message"] = result["error_message"]
        return out

    # ---------------------------------------------------------- passage / row modes

    def _read_passages(
        self,
        context: "AgentContext",
        passage_ids: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        not_found: List[str] = []
        total = 0
        store = self.inventory.passage_store
        for pid in passage_ids:
            atom = store.get(pid)
            if atom is None:
                not_found.append(pid)
                units.append({"unit_id": pid, "status": "not_found"})
                continue
            text = atom.text or ""
            units.append(
                {
                    "unit_id": atom.passage_id,
                    "file_id": atom.file_id,
                    "page_id": atom.page_id,
                    "page_number": atom.page_number,
                    "parent_section_id": atom.parent_section_id,
                    "block_label": atom.block_label,
                    "text": text,
                }
            )
            total += len(self.tokenizer.encode(text))

        units.sort(key=_atom_unit_sort_key)
        context.add_retrieval_log(
            tool_name="read",
            tokens=total,
            metadata={"unit_type": "passage", "requested": len(passage_ids), "not_found": len(not_found)},
        )
        return (
            ok(
                "PassageReadObservation",
                unit_type="passage",
                units=units,
                summary={"requested": len(passage_ids), "not_found": len(not_found)},
            ),
            {"retrieved_tokens": total, "unit_type": "passage"},
        )

    def _read_table_rows(
        self,
        context: "AgentContext",
        row_ids: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        not_found: List[str] = []
        total = 0
        store = self.inventory.table_row_store
        for rid in row_ids:
            atom = store.get(rid)
            if atom is None:
                not_found.append(rid)
                units.append({"unit_id": rid, "status": "not_found"})
                continue
            text = atom.text or ""
            units.append(
                {
                    "unit_id": atom.table_row_id,
                    "file_id": atom.file_id,
                    "page_id": atom.page_id,
                    "page_number": atom.page_number,
                    "parent_section_id": atom.parent_section_id,
                    "table_index": atom.table_index,
                    "row_index": atom.row_index,
                    "is_header_row": atom.is_header_row,
                    "html": atom.html,
                    "text": text,
                }
            )
            total += len(self.tokenizer.encode(text))

        units.sort(key=_table_row_unit_sort_key)
        context.add_retrieval_log(
            tool_name="read",
            tokens=total,
            metadata={"unit_type": "table_row", "requested": len(row_ids), "not_found": len(not_found)},
        )
        return (
            ok(
                "TableRowReadObservation",
                unit_type="table_row",
                units=units,
                summary={"requested": len(row_ids), "not_found": len(not_found)},
            ),
            {"retrieved_tokens": total, "unit_type": "table_row"},
        )

    # ---------------------------------------------------------- helpers

    def _filter_by_page_range(
        self,
        target_ids: List[str],
        unit_type: str,
        page_range: Tuple[int, int],
    ) -> List[str]:
        lo, hi = page_range
        out: List[str] = []
        for uid in target_ids:
            page_no = self._unit_page_number(uid, unit_type)
            if page_no is not None and lo <= page_no <= hi:
                out.append(uid)
        return out

    def _unit_page_number(self, unit_id: str, unit_type: str) -> Optional[int]:
        if unit_type == "page":
            page = self.page_store.get(unit_id)
            return page.page_number if page else None
        if unit_type == "passage":
            atom = self.inventory.passage_store.get(unit_id)
            return atom.page_number if atom else None
        atom = self.inventory.table_row_store.get(unit_id)
        return atom.page_number if atom else None


# ---------------------------------------------------------------- module helpers


def _bad_arg(message: str) -> Tuple[str, Dict[str, Any]]:
    return (
        err("invalid_argument", message),
        {"error": "invalid_argument"},
    )


def _empty(unit_type: str) -> Tuple[str, Dict[str, Any]]:
    obs = {
        "page": "PageReadObservation",
        "passage": "PassageReadObservation",
        "table_row": "TableRowReadObservation",
    }[unit_type]
    return (
        ok(obs, unit_type=unit_type, units=[], summary={"requested": 0}),
        {"retrieved_tokens": 0, "unit_type": unit_type},
    )


def _page_unit_sort_key(u: Dict[str, Any]) -> Tuple[Any, ...]:
    return (u.get("file_id") or "", u.get("page_number") or 0, u.get("unit_id") or "")


def _atom_unit_sort_key(u: Dict[str, Any]) -> Tuple[Any, ...]:
    return (u.get("file_id") or "", u.get("page_number") or 0, u.get("unit_id") or "")


def _table_row_unit_sort_key(u: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        u.get("file_id") or "",
        u.get("page_number") or 0,
        u.get("table_index") or 0,
        u.get("row_index") or 0,
    )


__all__ = ["ReadTool"]
