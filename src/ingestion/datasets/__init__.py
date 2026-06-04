"""Dataset-specific page-asset adapters.

Each module here converts one external dataset's native format into the
project's ``PageAsset`` contract (see :mod:`storage.page_store`) so the
generic ingest pipeline (NER → entity/sentence graph → faiss stores) and
the agent's read / graph tools work unchanged.

Currently:

* :mod:`ingestion.datasets.multihop_corpus` — HippoRAG-format cross-document
  multi-hop corpora (2WikiMultihopQA / MuSiQue / HotpotQA).

The Double-Bench adapter still lives at :mod:`ingestion.dbocr_pageassets`
(kept in place to avoid churning its import path); it can be moved here
later for symmetry.
"""

from ingestion.datasets.multihop_corpus import (
    WIKI_GLINER_LABELS,
    build_pageasset_from_corpus_item,
    corpus_file_id,
    iter_question_paragraphs,
    load_corpus,
    load_qa,
)

__all__ = [
    "WIKI_GLINER_LABELS",
    "build_pageasset_from_corpus_item",
    "corpus_file_id",
    "iter_question_paragraphs",
    "load_corpus",
    "load_qa",
]
