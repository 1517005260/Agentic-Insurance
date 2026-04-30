"""Unified faiss-backed embedding store.

One store = one directory containing:

    index.faiss     IndexFlatIP, vectors L2-normalized → IP == cos sim
    meta.parquet    row_idx aligned to faiss; cols: hash_id, text, [+caller-supplied]
    config.json     {dim, namespace, metric, size}

Hash-keyed dedup via md5 of normalized text. Stores are designed to be
**global** — new files append into the same store; ``file_id`` is just a
column on meta.parquet, used for filtering, not for partitioning.
"""
import json
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np
import pandas as pd


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
        if meta_exists:
            self._meta = pd.read_parquet(self._meta_path())
            self._hash_id_to_idx = {
                h: i for i, h in enumerate(self._meta["hash_id"].tolist())
            }

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
        # never leaves index / meta / config out of sync.
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

    # ------------------------------------------------------------------ misc

    def __len__(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def has(self, hash_id: str) -> bool:
        return hash_id in self._hash_id_to_idx

    def get_text(self, hash_id: str) -> str:
        return self._meta.iloc[self._hash_id_to_idx[hash_id]]["text"]

    def get_index(self, hash_id: str) -> int:
        return self._hash_id_to_idx[hash_id]

    def get_meta_row(self, hash_id: str) -> Dict[str, Any]:
        return self._meta.iloc[self._hash_id_to_idx[hash_id]].to_dict()

    def get_embedding(self, hash_id: str) -> np.ndarray:
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
        """Append items, deduping on hash_id. Returns the list of hash_ids actually added."""
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

        if self.dim is None:
            self.dim = int(embeddings.shape[1])
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dim)
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} ≠ store dim {self.dim}"
            )

        # IndexFlatIP cosine semantics require unit-norm vectors. Our
        # EmbeddingClient already L2-normalizes, but assert defensively so a
        # caller that builds vectors by hand can't silently break similarity.
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            # Re-normalize rather than raising — keeps callers tolerant.
            zero = norms == 0
            norms[zero] = 1.0
            embeddings = (embeddings / norms[:, None]).astype(np.float32, copy=False)

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

        start = self._index.ntotal
        self._index.add(kept_e)
        for offset, h in enumerate(kept_h):
            self._hash_id_to_idx[h] = start + offset

        new_row: Dict[str, List[Any]] = {"hash_id": kept_h, "text": kept_t}
        if extra_metadata:
            for col, values in extra_metadata.items():
                if len(values) != len(hash_ids):
                    raise ValueError(
                        f"extra_metadata['{col}'] length {len(values)} ≠ hash_ids length {len(hash_ids)}"
                    )
                new_row[col] = [values[i] for i in keep_local_idx]
        new_df = pd.DataFrame(new_row)

        if self._meta.empty:
            self._meta = new_df
        else:
            # Pad missing columns on either side so concat doesn't drop info.
            self._meta = pd.concat([self._meta, new_df], ignore_index=True, sort=False)

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
        for i, h in enumerate(all_hash_ids):
            if h in self._hash_id_to_idx or h in seen_in_batch:
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
        """faiss top-k. Single-query input returns a flat list; batch returns list-of-lists."""
        if self._index is None or self._index.ntotal == 0:
            return [] if query_embedding.ndim == 1 else [[] for _ in range(query_embedding.shape[0])]

        single = query_embedding.ndim == 1
        q = query_embedding.reshape(1, -1) if single else query_embedding
        q = np.ascontiguousarray(q.astype(np.float32))

        scores, indices = self._index.search(q, min(k, self._index.ntotal))
        out: List[List[Tuple[str, float]]] = []
        hash_ids = self._meta["hash_id"].tolist()
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

    @property
    def hash_ids(self) -> List[str]:
        return self._meta["hash_id"].tolist()

    @property
    def texts(self) -> List[str]:
        return self._meta["text"].tolist()

    @property
    def embeddings(self) -> np.ndarray:
        if self._index is None or self._index.ntotal == 0:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        return self._index.reconstruct_n(0, self._index.ntotal)

    @property
    def hash_id_to_text(self) -> Dict[str, str]:
        return dict(zip(self._meta["hash_id"].tolist(), self._meta["text"].tolist()))

    @property
    def text_to_hash_id(self) -> Dict[str, str]:
        # Last write wins on duplicate text — same surface across files
        # collapses to one hash via md5(text), so this is well-defined.
        return {t: h for h, t in zip(self._meta["hash_id"].tolist(), self._meta["text"].tolist())}

    @property
    def hash_id_to_idx(self) -> Dict[str, int]:
        return dict(self._hash_id_to_idx)

    def get_hash_id_to_text(self) -> Dict[str, str]:
        return self.hash_id_to_text

    def get_embeddings(self, hash_ids: Iterable[str]) -> np.ndarray:
        ids = list(hash_ids)
        if not ids:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        if self._index is None or self._index.ntotal == 0:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        rows = np.asarray([self._hash_id_to_idx[h] for h in ids], dtype=np.int64)
        all_emb = self._index.reconstruct_n(0, self._index.ntotal)
        return all_emb[rows]

    # ----------------------------------------------------- meta column access

    def meta_column(self, name: str) -> List[Any]:
        if name not in self._meta.columns:
            return [None] * len(self._meta)
        return self._meta[name].tolist()

    def filter_hash_ids(self, **conditions: Any) -> List[str]:
        """Return hash_ids whose meta row matches every (col, value) condition."""
        if not conditions:
            return self.hash_ids
        mask = pd.Series(True, index=self._meta.index)
        for col, value in conditions.items():
            if col not in self._meta.columns:
                return []
            mask &= self._meta[col] == value
        return self._meta.loc[mask, "hash_id"].tolist()
