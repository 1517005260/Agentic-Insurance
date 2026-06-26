# semantic tier (ranked retrieval — needs embeddings + the graph algorithm)

These programs reuse the real retrieval channels (genuine PPR / dense / entity
embeddings — zero degradation), which need the embedding API (network), GLiNER,
faiss + igraph, and the KG artifacts under STORAGE_PATH. Those all live in a
host-side **sidecar** (`agentic.tools.evfs_sidecar`) that loads the channels
once and serves the ops over a unix socket bound into the sandbox at
`$EVFS_SIDECAR_SOCK`; the scripts here are thin stdlib clients, so the sandbox
itself stays fully hermetic (no venv / network / GPU). They are mounted only on
the graph agent's PATH and kept separate from `lexical/` so the agent can be
ablated: lexical-only vs lexical+semantic.

All print `rank … score …` and are candidate generation — READ the source to
confirm, don't answer from the preview.

- `rank_passages --query "..." [--top-k N]` — graph/PPR page ranking.
- `search_dense   --query "..." [--top-k N]` — dense (embedding) page ranking.
- `seed_surfaces  --query "..." [--top-k N]` — surfaces nearest the query
  (embedding counterpart of `find_surface`).
- `candidate_aliases sf_* [--top-k N]` — surfaces nearest a given surface
  (possible synonyms; EvidenceFS does NO merge — you decide).
- `semantic_bridge sf_* --query "..." [--top-k N]` — surfaces co-mentioned with
  this one in the query-most-relevant sentences (the embedding counterpart of
  `bridge_surfaces`), each with the bridging `s_*`. Read the sentence to recover
  the relation.
