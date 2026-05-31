"""Four retrieval indexes; each backed by a global cross-file store.

* ``text_dense``    — sentence text embeddings  → faiss store at ``STORAGE_PATH/faiss/dense/``
* ``vision_dense``  — page-image embeddings     → faiss store at ``STORAGE_PATH/faiss/visual/``
* ``bm25``          — tantivy index             → ``STORAGE_PATH/bm25/``
* ``graph``         — LinearRAG relation-free entity graph → ``STORAGE_PATH/faiss/graph/``

Every builder consumes a ``PageAsset`` list and a ``file_id``. Stores are
**global**: each call appends, ``file_id`` is a meta column for filtering
and per-file removal. Build-time the four builders are independent.

This package intentionally re-exports nothing: each submodule pulls in
heavyweight backends (torch + transformers for ``graph_linearrag`` /
``text_dense``, tantivy for ``bm25_tantivy``) and importing the
package would chain-load all of them. Workers that need only one
builder — most importantly the spawn-mode graph subprocess — would
otherwise pay 600 MB+ of resident memory at child boot. Callers
import the specific submodule they need
(``from ingestion.index.maintenance import purge_file_artifacts``).
"""

__all__: list[str] = []
