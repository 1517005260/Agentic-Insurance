"""Unified faiss-backed embedding store.

One store = one directory containing:

    index.faiss     IndexFlatIP, vectors L2-normalized → IP == cos sim
    meta.parquet    row_idx aligned to faiss; cols: hash_id, text, [+caller-supplied]
    config.json     {dim, namespace, metric, size}

Hash-keyed dedup via md5 of normalized text. Stores are designed to be
**global** — new files append into the same store; ``file_id`` is just a
column on meta.parquet, used for filtering, not for partitioning.

**Process-level cache**: the lifespan PPR channel and per-ingest LinearRAG
both target the same on-disk artifact. Constructing independent
``EmbeddingStore`` handles would double the resident set, since each
store's ``faiss.read_index`` materialises 100 MB - 1 GB depending on
corpus. Always go through :func:`get_or_create_store` (alias
:func:`shared_store`) instead of constructing ``EmbeddingStore`` directly;
the helper deduplicates by ``(canonical-dir-path, namespace)`` so the
cached instance is reused.

**Concurrency model** (since the cache made one store visible to both
ingest writers and query readers): every method that mutates or reads
``self._index`` / ``self._meta`` / ``self._hash_id_to_idx`` runs under
``self._lock`` (a re-entrant ``threading.RLock``). The lock is per-
instance so two distinct stores don't serialise. ``add()`` is the only
true writer; readers (``topk`` / ``all_similarities`` / metadata
properties) take the lock to atomic-snapshot index size and meta state
together, preventing the "faiss appended row N but meta still has N-1
rows" race that would otherwise surface as ``IndexError`` on a
concurrent query.
"""
import json
import logging
import threading
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _index_is_hnsw(index) -> bool:
    return "HNSW" in type(index).__name__


def _hnsw_params(namespace: str):
    """(use_hnsw, M, efConstruction, efSearch) for ``namespace``.

    Lazy import so this low-level store module never hard-depends on the
    settings module at import time (avoids any future import cycle).
    """
    from config.settings import (
        EMBEDDING_HNSW_NAMESPACES, EMBEDDING_HNSW_M,
        EMBEDDING_HNSW_EF_CONSTRUCTION, EMBEDDING_HNSW_EF_SEARCH,
    )
    return (
        namespace in EMBEDDING_HNSW_NAMESPACES,
        EMBEDDING_HNSW_M,
        EMBEDDING_HNSW_EF_CONSTRUCTION,
        EMBEDDING_HNSW_EF_SEARCH,
    )


def _make_index(dim: int, namespace: str):
    """IndexHNSWFlat (IP) for HNSW namespaces, else exact IndexFlatIP.
    Both are cosine-correct on the store's L2-normalized vectors."""
    use, m, efc, efs = _hnsw_params(namespace)
    if use:
        ix = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        ix.hnsw.efConstruction = efc
        ix.hnsw.efSearch = efs
        return ix
    return faiss.IndexFlatIP(dim)


def _hash(text: str, namespace: str) -> str:
    return f"{namespace}-{md5(text.encode()).hexdigest()}"


class EmbeddingStore:
    """faiss + parquet store; flexible meta schema.

    `extra_metadata` on writes is a dict of column → list-of-values parallel
    to ``hash_ids``. Columns are added on first write and persisted; rows
    that don't supply a value get NaN — pandas / parquet handles it.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        namespace: str,
        dim: Optional[int] = None,
    ):
        self.dir = Path(directory)
        self.namespace = namespace
        self.dim = dim

        self.dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.Index] = None
        self._meta: pd.DataFrame = pd.DataFrame({"hash_id": [], "text": []})
        self._hash_id_to_idx: Dict[str, int] = {}
        # Generation counter bumped on every _meta/_hash_id_to_idx
        # mutation; the derived-view caches below are keyed by it so any
        # mutation forces a rebuild on next access (provably zero-semantic
        # — see _memo). Covers the append/load/reset paths (the only
        # mutations on the indexing path; a delete path, if added, must
        # also bump _gen).
        self._gen = 0
        self._cache: Dict[str, tuple] = {}

        # Re-entrant lock so an ``add()``-from-tests / ``insert_text()``
        # call that internally walks the same store doesn't deadlock.
        # Held by every method that touches the three mutable fields
        # (``_index`` / ``_meta`` / ``_hash_id_to_idx``).
        self._lock = threading.RLock()

        self._load()

    # ------------------------------------------------------------------ I/O

    def _config_path(self) -> Path:
        return self.dir / "config.json"

    def _index_path(self) -> Path:
        return self.dir / "index.faiss"

    def _meta_path(self) -> Path:
        return self.dir / "meta.parquet"

    def _load(self) -> None:
        config_exists = self._config_path().exists()
        index_exists = self._index_path().exists()
        meta_exists = self._meta_path().exists()

        if config_exists:
            cfg = json.loads(self._config_path().read_text(encoding="utf-8"))
            self.dim = cfg.get("dim", self.dim)
            self.namespace = cfg.get("namespace", self.namespace)
        if index_exists:
            self._index = faiss.read_index(str(self._index_path()))
            if self.dim is None:
                self.dim = self._index.d
            # ANN migration: an on-disk flat index for an HNSW-configured
            # namespace is rebuilt to HNSW in memory (one-time per
            # process; the next persist writes HNSW). reconstruct_n works
            # on IndexFlat. Shadow-A/B-proven to leave the post-admission
            # accepted alias set identical, so this is transparent.
            use, m, efc, efs = _hnsw_params(self.namespace)
            if use and not _index_is_hnsw(self._index) and self._index.ntotal > 0:
                vecs = self._index.reconstruct_n(0, self._index.ntotal)
                ix = faiss.IndexHNSWFlat(
                    self._index.d, m, faiss.METRIC_INNER_PRODUCT
                )
                ix.hnsw.efConstruction = efc
                ix.hnsw.efSearch = efs
                ix.add(vecs)
                self._index = ix
                logger.info(
                    "EmbeddingStore[%s]: migrated flat→HNSW in memory "
                    "(%d vecs, M=%d efSearch=%d)",
                    self.namespace, ix.ntotal, m, efs,
                )
            elif use and _index_is_hnsw(self._index):
                # honor a possibly-changed configured efSearch on reload
                self._index.hnsw.efSearch = efs
        if meta_exists:
            self._meta = pd.read_parquet(self._meta_path())
            self._hash_id_to_idx = {
                h: i for i, h in enumerate(self._meta["hash_id"].tolist())
            }
            self._gen += 1

        # Either nothing exists (cold start, fine), or all three exist and
        # row counts match. ANY other combination is a partial-write
        # corruption — fail loud rather than silently rebuild and confuse
        # downstream alignment of meta-row → faiss-row.
        artifacts_present = sum((config_exists, index_exists, meta_exists))
        if artifacts_present == 0:
            return
        if not (config_exists and index_exists and meta_exists):
            raise RuntimeError(
                f"EmbeddingStore at {self.dir} is in a partial-write state: "
                f"config={config_exists} index={index_exists} meta={meta_exists}. "
                f"Delete the directory and re-ingest, or restore from backup."
            )
        if len(self._meta) != self._index.ntotal:
            raise RuntimeError(
                f"EmbeddingStore at {self.dir} is inconsistent: "
                f"meta has {len(self._meta)} rows but faiss index has "
                f"{self._index.ntotal}. Likely a partial-write crash; "
                f"rebuild from upstream sources."
            )

    def save(self) -> None:
        # Atomic writes: stage to ``.tmp`` then replace, so a crash mid-save
        # never leaves index / meta / config out of sync. Held under
        # ``self._lock`` because we're snapshotting ``_index.ntotal`` /
        # ``_meta`` / ``dim`` together; a concurrent ``add()`` between
        # writing the faiss index and writing meta would emit a
        # config.size that doesn't match.
        with self._lock:
            if self._index is not None:
                tmp = self._index_path().with_suffix(".faiss.tmp")
                faiss.write_index(self._index, str(tmp))
                tmp.replace(self._index_path())
            meta_tmp = self._meta_path().with_suffix(".parquet.tmp")
            self._meta.to_parquet(meta_tmp, index=False)
            meta_tmp.replace(self._meta_path())
            cfg = {
                "dim": int(self.dim) if self.dim is not None else None,
                "namespace": self.namespace,
                "metric": "ip",
                "size": int(self._index.ntotal) if self._index is not None else 0,
            }
            cfg_tmp = self._config_path().with_suffix(".json.tmp")
            cfg_tmp.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cfg_tmp.replace(self._config_path())

    def reload_from_disk(self) -> None:
        """Re-read faiss / parquet / config from disk **into the same object**.

        Process-wide caching (``shared_embedding_store``) means the
        lifespan-built channel and any per-ingest LinearRAG share the
        SAME ``EmbeddingStore`` instance — for in-process ingest the
        live mutation already shows up to query, no reload needed. But
        ``/admin/refresh-indexes`` is documented as the out-of-band
        recovery path (someone hand-edited ``faiss/`` or restored a
        backup); the cached store would otherwise stay frozen at the
        pre-edit snapshot. This method drops the in-memory state and
        re-runs ``_load`` against the (possibly modified) on-disk
        artifacts, all under the per-store lock so a concurrent query
        can't observe a half-loaded mid-state.
        """
        with self._lock:
            self._index = None
            self._meta = pd.DataFrame({"hash_id": [], "text": []})
            self._hash_id_to_idx = {}
            self._gen += 1
            self._load()

    # ------------------------------------------------------------------ misc

    # All these helpers read either ``_index`` or ``_hash_id_to_idx``
    # and must run under the per-instance lock; otherwise a concurrent
    # ``add()`` between e.g. ``_hash_id_to_idx[hash_id]`` and the
    # ``_meta.iloc[...]`` lookup would give an out-of-range row.

    def __len__(self) -> int:
        with self._lock:
            return int(self._index.ntotal) if self._index is not None else 0

    def has(self, hash_id: str) -> bool:
        with self._lock:
            return hash_id in self._hash_id_to_idx

    def get_text(self, hash_id: str) -> str:
        with self._lock:
            return self._meta.iloc[self._hash_id_to_idx[hash_id]]["text"]

    def get_index(self, hash_id: str) -> int:
        with self._lock:
            return self._hash_id_to_idx[hash_id]

    def get_meta_row(self, hash_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._meta.iloc[self._hash_id_to_idx[hash_id]].to_dict()

    def get_embedding(self, hash_id: str) -> np.ndarray:
        with self._lock:
            idx = self._hash_id_to_idx[hash_id]
            return self._index.reconstruct(idx)

    # ------------------------------------------------------------------ writes

    def hash_for(self, text: str) -> str:
        return _hash(text, self.namespace)

    def add(
        self,
        hash_ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Union[np.ndarray, Sequence[Sequence[float]]],
        extra_metadata: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> List[str]:
        """Append items, deduping on hash_id. Returns the list of hash_ids actually added.

        Critical: this method must be atomic with respect to readers
        (``topk`` / ``hash_id_to_text`` etc). The faiss index growth
        and the meta DataFrame append are two operations; a reader
        that sliced between them would see ``ntotal == N+1`` but
        ``len(_meta) == N`` and either misalign labels or raise
        IndexError. ``self._lock`` covers the whole sequence.
        """
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings, dtype=np.float32)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (N, D); got shape {embeddings.shape}"
            )
        if len(hash_ids) != embeddings.shape[0] or len(texts) != embeddings.shape[0]:
            raise ValueError("hash_ids / texts / embeddings length mismatch")
        # Validate ``extra_metadata`` shapes BEFORE any mutation so a
        # length-mismatch raise can't leave behind a half-written index.
        if extra_metadata:
            for col, values in extra_metadata.items():
                if len(values) != len(hash_ids):
                    raise ValueError(
                        f"extra_metadata['{col}'] length {len(values)} ≠ hash_ids length {len(hash_ids)}"
                    )

        # IndexFlatIP cosine semantics require unit-norm vectors. Our
        # EmbeddingClient already L2-normalizes, but assert defensively so a
        # caller that builds vectors by hand can't silently break similarity.
        # Run normalisation OUTSIDE the lock — purely on the caller's
        # arrays, no shared state touched.
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            zero = norms == 0
            norms[zero] = 1.0
            embeddings = (embeddings / norms[:, None]).astype(np.float32, copy=False)

        with self._lock:
            if self.dim is None:
                self.dim = int(embeddings.shape[1])
            if self._index is None:
                self._index = _make_index(self.dim, self.namespace)
            if embeddings.shape[1] != self.dim:
                raise ValueError(
                    f"Embedding dim {embeddings.shape[1]} ≠ store dim {self.dim}"
                )

            keep_local_idx: List[int] = []
            seen_in_batch: set = set()
            for i, h in enumerate(hash_ids):
                if h in self._hash_id_to_idx or h in seen_in_batch:
                    continue
                keep_local_idx.append(i)
                seen_in_batch.add(h)

            if not keep_local_idx:
                return []

            kept_h = [hash_ids[i] for i in keep_local_idx]
            kept_t = [texts[i] for i in keep_local_idx]
            kept_e = embeddings[keep_local_idx].astype(np.float32, copy=False)

            # Build the new meta row BEFORE touching faiss / map so a
            # pandas-side error (e.g. mixed dtypes) raises before any
            # mutation.
            new_row: Dict[str, List[Any]] = {"hash_id": kept_h, "text": kept_t}
            if extra_metadata:
                for col, values in extra_metadata.items():
                    new_row[col] = [values[i] for i in keep_local_idx]
            new_df = pd.DataFrame(new_row)

            start = self._index.ntotal
            self._index.add(kept_e)
            for offset, h in enumerate(kept_h):
                self._hash_id_to_idx[h] = start + offset
                self._gen += 1

            if self._meta.empty:
                self._meta = new_df
            else:
                # Pad missing columns on either side so concat doesn't drop info.
                self._meta = pd.concat([self._meta, new_df], ignore_index=True, sort=False)

            # ``save()`` re-acquires the same RLock — fine because it's
            # re-entrant; we keep both inside one critical section so a
            # concurrent reader can't observe half-written state.
            self.save()
            return kept_h

    def insert_text(
        self,
        texts: Sequence[str],
        embedding_client,
        extra_metadata: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> List[str]:
        """Compute hashes, embed only the new ones, append. Returns all hash_ids (new+seen)."""
        all_hash_ids = [_hash(t, self.namespace) for t in texts]

        new_local_idx: List[int] = []
        seen_in_batch: set = set()
        # Snapshot ``_hash_id_to_idx`` membership under the lock so a
        # concurrent ``add()`` between two iterations can't flip the
        # dedup answer mid-loop. We don't need to hold the lock while
        # we then call ``embedding_client.encode`` (a slow network IO);
        # ``add()`` below re-checks dedup atomically anyway.
        with self._lock:
            existing_hash_ids = set(self._hash_id_to_idx)
        for i, h in enumerate(all_hash_ids):
            if h in existing_hash_ids or h in seen_in_batch:
                continue
            new_local_idx.append(i)
            seen_in_batch.add(h)

        if not new_local_idx:
            return all_hash_ids

        new_texts = [texts[i] for i in new_local_idx]
        new_hash_ids = [all_hash_ids[i] for i in new_local_idx]

        embeddings = embedding_client.encode(new_texts)
        if isinstance(embeddings, list):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        sliced_extra: Optional[Dict[str, List[Any]]] = None
        if extra_metadata:
            sliced_extra = {
                col: [values[i] for i in new_local_idx] for col, values in extra_metadata.items()
            }
            # Re-key as full-length so the .add() length check passes.
            sliced_extra_full = {
                col: [None] * len(new_hash_ids) for col in extra_metadata
            }
            for col in extra_metadata:
                sliced_extra_full[col] = sliced_extra[col]
            sliced_extra = sliced_extra_full

        self.add(new_hash_ids, new_texts, embeddings, extra_metadata=sliced_extra)
        return all_hash_ids

    # ----------------------------------------------------------------- search

    def topk(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[List[Tuple[str, float]]] | List[Tuple[str, float]]:
        """faiss top-k. Single-query input returns a flat list; batch returns list-of-lists.

        Locked because we read ``_index.ntotal`` and ``_meta['hash_id']``
        as a paired snapshot — a concurrent ``add()`` between those two
        reads would emit a faiss row index past the end of meta and
        crash with ``IndexError``.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return [] if query_embedding.ndim == 1 else [[] for _ in range(query_embedding.shape[0])]

            single = query_embedding.ndim == 1
            q = query_embedding.reshape(1, -1) if single else query_embedding
            q = np.ascontiguousarray(q.astype(np.float32))

            scores, indices = self._index.search(q, min(k, self._index.ntotal))
            hash_ids = self._meta["hash_id"].tolist()

        # faiss search arrays are local; build the result outside the
        # lock so concurrent writers don't wait on Python list construction.
        out: List[List[Tuple[str, float]]] = []
        for row in range(scores.shape[0]):
            row_out: List[Tuple[str, float]] = []
            for rank in range(scores.shape[1]):
                idx = int(indices[row, rank])
                if idx < 0:
                    continue
                row_out.append((hash_ids[idx], float(scores[row, rank])))
            out.append(row_out)
        return out[0] if single else out

    # ------------------------------------------------------ collection views

    def _memo(self, key, builder):
        """Generation-keyed memo for the O(N) derived views below.

        Holds the RLock itself, so ``builder`` runs under the same lock
        the original ``with self._lock:`` body held (RLock is
        re-entrant) — identical concurrency semantics. A cached value is
        reused only while ``_gen`` is unchanged, so any mutation
        (append/load/reset bumps ``_gen``) forces a rebuild.
        """
        with self._lock:
            c = self._cache.get(key)
            if c is not None and c[0] == self._gen:
                return c[1]
            v = builder()
            self._cache[key] = (self._gen, v)
            return v

    @property
    def hash_ids(self) -> List[str]:
        return self._memo("hash_ids", lambda: self._meta["hash_id"].tolist())

    @property
    def texts(self) -> List[str]:
        return self._memo("texts", lambda: self._meta["text"].tolist())

    @property
    def embeddings(self) -> np.ndarray:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return np.zeros((0, self.dim or 0), dtype=np.float32)
            return self._index.reconstruct_n(0, self._index.ntotal)

    def all_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Inner product of ``query_embedding`` against every stored vector,
        aligned to ``self.hash_ids`` row order.

        Equivalent to ``self.embeddings @ query`` but routes through
        ``IndexFlatIP.search`` so we never materialize the full (N, D)
        embedding matrix in user code — for the PPR hot path that meant
        a per-request ~60 MB temporary on a 10K-passage corpus, lifted
        to GB-scale on bigger ones. faiss does the same arithmetic in
        C++ with BLAS without the round-trip allocation.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return np.zeros((0,), dtype=np.float32)
            q = np.ascontiguousarray(query_embedding.reshape(1, -1).astype(np.float32))
            n = self._index.ntotal
            if _index_is_hnsw(self._index):
                # HNSW.search(q, ntotal) is a pathological full-graph
                # walk and not row-aligned. all_similarities means an
                # exact full scan — reconstruct + BLAS matmul, exact and
                # row-aligned. (Not hit by HNSW namespaces in practice;
                # defensive so a misconfig can't silently corrupt PPR.)
                mat = self._index.reconstruct_n(0, n)
                return (mat @ q.reshape(-1)).astype(np.float32)
            scores, indices = self._index.search(q, n)
        aligned = np.zeros(n, dtype=np.float32)
        # IndexFlatIP returns one score per stored row — every position is
        # filled, no -1 sentinels possible at k == ntotal.
        aligned[indices[0]] = scores[0]
        return aligned

    @property
    def hash_id_to_text(self) -> Dict[str, str]:
        return self._memo("hash_id_to_text", lambda: dict(
            zip(self._meta["hash_id"].tolist(), self._meta["text"].tolist())))

    @property
    def text_to_hash_id(self) -> Dict[str, str]:
        # Last write wins on duplicate text — same surface across files
        # collapses to one hash via md5(text), so this is well-defined.
        return self._memo("text_to_hash_id", lambda: {
            t: h for h, t in zip(
                self._meta["hash_id"].tolist(), self._meta["text"].tolist())})

    @property
    def hash_id_to_idx(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._hash_id_to_idx)

    def get_hash_id_to_text(self) -> Dict[str, str]:
        return self.hash_id_to_text

    def get_embeddings(self, hash_ids: Iterable[str]) -> np.ndarray:
        ids = list(hash_ids)
        if not ids:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return np.zeros((0, self.dim or 0), dtype=np.float32)
            # Per-row reconstruct beats a full ``reconstruct_n`` + advanced
            # indexing whenever ``len(ids) << ntotal`` (which is the common
            # case — agent / RAG callers typically pass a handful of seeds).
            out = np.empty((len(ids), self._index.d), dtype=np.float32)
            for i, h in enumerate(ids):
                out[i] = self._index.reconstruct(int(self._hash_id_to_idx[h]))
            return out

    # ----------------------------------------------------- meta column access

    def meta_column(self, name: str) -> List[Any]:
        with self._lock:
            if name not in self._meta.columns:
                return [None] * len(self._meta)
            return self._meta[name].tolist()

    def filter_hash_ids(self, **conditions: Any) -> List[str]:
        """Return hash_ids whose meta row matches every (col, value) condition."""
        if not conditions:
            return self.hash_ids
        with self._lock:
            mask = pd.Series(True, index=self._meta.index)
            for col, value in conditions.items():
                if col not in self._meta.columns:
                    return []
                mask &= self._meta[col] == value
            return self._meta.loc[mask, "hash_id"].tolist()


# --------------------------------------------------------------- factory

def get_or_create_store(
    directory: Union[str, Path],
    namespace: str,
    dim: Optional[int] = None,
) -> "EmbeddingStore":
    """Return the process-cached :class:`EmbeddingStore` for ``(directory, namespace)``.

    Construction calls ``faiss.read_index`` + ``pd.read_parquet`` which
    are expensive (100 MB - 1 GB resident depending on corpus). Before
    centralising this cache, the lifespan PPR channel and each per-
    ingest LinearRAG built independent stores for the same on-disk
    artifact, doubling the RSS — a major contributor to OOM on 8 GB
    hosts.

    Cache contract:

    * Keyed by ``(canonical-string-path, namespace)``. Two callers
      passing semantically-equal paths share the cached instance.
    * ``dim`` only matters on first construction (when the on-disk
      ``config.json`` doesn't exist yet). Subsequent calls ignore it
      because the cached store's ``dim`` is already set.
    * Mutability: ``add()`` mutates in-memory state; concurrent readers
      see new vectors immediately. Writes are serialised by the upstream
      ``INGEST_LOCK`` (see :mod:`api.services.files`).
    """
    from config.shared import shared_embedding_store_for

    return shared_embedding_store_for(Path(directory), namespace)


# Concise alias kept for tab-completion / readability at callsites.
shared_store = get_or_create_store
