"""LinearRAG build-time configuration.

Lives under ``config/`` so all project configs sit in one place. Storage
paths come from ``config.settings`` — this struct only carries runtime
knobs (embedding client, GLiNER model id / labels / threshold, NER worker
count, alias-edge quality thresholds).

The ``EmbeddingClient`` import is guarded by ``TYPE_CHECKING`` so importing
``config`` doesn't pull ``model_client`` (which itself imports ``config``).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from model_client import EmbeddingClient


# Default open-set NER label list (see ingestion.index.linear_rag.ner.DEFAULT_LABELS).
# Defined as a module-level list literal so the dataclass default_factory
# returns a fresh copy per instance and admin overrides don't leak across
# requests.
_DEFAULT_GLINER_LABELS: List[str] = [
    "product",
    "term",
    "concept",
    "organization",
    "code",
    "law",
    "regulation",
    "person role",
    # Note: ``currency`` is intentionally OFF by default — currency
    # surfaces like "加元" / "英镑" enter the entity universe as spurious
    # nodes, which pollute PPR neighbourhoods. Admins running
    # an FX-heavy domain can add it via the config-store override.
]


@dataclass
class LinearRAGConfig:
    embedding_client: Optional["EmbeddingClient"] = None

    # GLiNER NER configuration. ``gliner_model_id`` is a HuggingFace repo
    # id; weights live in the standard HF cache (``~/.cache/huggingface/hub/``)
    # so swapping the model id picks up a different checkpoint without
    # touching ``STORAGE_PATH``.
    #
    # ``gliner_labels`` is the open-set label prompt — change this list to
    # adapt to a new domain (e.g. ``["disease", "drug", "procedure"]`` for
    # medical) without touching the codebase. English wording is intentional:
    # GLiNER's mT5 backbone tokenises English label tokens more stably than
    # Chinese ones in the multilingual variant.
    #
    # ``gliner_threshold`` controls the score floor for emitted spans;
    # 0.3 balances recall against noise.
    gliner_model_id: str = "urchade/gliner_multiv2.1"
    gliner_labels: List[str] = field(
        default_factory=lambda: list(_DEFAULT_GLINER_LABELS)
    )
    # Decoy / noise-sink subset of ``gliner_labels``. GLiNER is asked to
    # score these (so junk surfaces — pronouns, bare dates, numbers —
    # attach to them) and the pipeline then discards spans tagged with
    # them. Model-native noise control via the open-set label prompt,
    # not a hand-rolled surface filter. Empty by default (inert);
    # listed members MUST also appear in ``gliner_labels``.
    gliner_noise_labels: List[str] = field(default_factory=list)
    gliner_threshold: float = 0.3
    gliner_batch_size: int = 16

    # Structural span-shape filter applied to every GLiNER output BEFORE
    # surface normalization (see ``ner.is_misbound_span``). Rejects spans
    # with >``ner_max_span_chars`` raw characters AND no bracket (real
    # product / clause names rarely exceed 80 chars; bracketed surfaces
    # are kept regardless of length) OR with interior hard sentence-end
    # punctuation. 80 is a defensive cap; legitimate legal/insurance
    # surfaces rarely exceed ~50 chars.
    ner_max_span_chars: int = 80

    max_workers: int = 4

    # How often LinearRAG.index() persists LinearRAG.graphml, in
    # index() calls. Default 1 = write every doc; the per-file API
    # builder makes a fresh instance per file so its counter is always
    # 1. A persistent bulk driver (one LinearRAG over a whole corpus,
    # e.g. GraphIndexBuilder(reuse_graph=True)) sets this >1 so the
    # O(V+E) graphml (de)serialisation is amortised across docs instead
    # of paid every doc, which is an O(N²) wall-time blow-up. Such a
    # driver must force a final flush_graphml() at the end and before
    # any checkpoint that reads the on-disk graphml.
    graphml_flush_every: int = 1

    # How often LinearRAG.index() recomputes the expensive Leiden
    # (compute_clusters) partition for the returned ``cluster_shape``,
    # in index() calls. Default 1 = compute every doc; the per-file API
    # builder makes a fresh instance per file so its counter is always
    # 1. Leiden
    # is O(E_alias) and E_alias grows with the corpus, so paying it
    # every doc makes a persistent bulk build O(N²) in wall time. A
    # bulk driver (one LinearRAG over a whole corpus, e.g.
    # GraphIndexBuilder(reuse_graph=True)) sets this >1 so the
    # partition is recomputed on a cadence; on skipped docs index()
    # still returns a well-formed cluster_shape carrying the cheap O(V)
    # largest_cc_ratio percolation tripwire (the expensive
    # diameter / pairwise fields are already cadence-only via the P1
    # cheap=True path). The resulting graph is unaffected — clusters
    # are a recomputable derived view over the immutable alias edges.
    cluster_shape_every: int = 1

    # Build-stage toggles (default True = current behavior preserved). A
    # dataset whose entity surfaces are clean (e.g. Wikipedia multi-hop) can
    # turn alias ER off — the per-new-entity dual-query + reranker veto is both
    # the build's dominant cost and a source of false bridges there. A corpus
    # of independent single-page documents (one file_id per passage) can turn
    # adjacent-passage edges off — they would only link arbitrary corpus
    # neighbours. Set per-build via LinearRAGConfig from the dataset driver;
    # the algorithm layer never hard-codes a domain default.
    alias_edges_enabled: bool = True
    adjacent_passage_edges_enabled: bool = True

    # Alias-edge RECALL stage (blocking) — dual-query top-k (bare-surface +
    # mention centroid). Precision is NO LONGER decided here: it moves to the
    # Stage-C gate below (IDF lexical overlap + co-occurrence veto). The recall
    # floor is loosened from the old 0.85 so more true variants survive for the
    # gate to judge; the gradient cutoff still trims the unrelated long tail.
    alias_top_k: int = 20
    alias_gradient: float = 0.3
    alias_min_sim: float = 0.80  # = er_recall_floor

    # Mention-context centroid: dedup mention sentences and cap per entity
    # so high-frequency entities don't get pulled toward boilerplate noise.
    centroid_max_mentions: int = 8

    # Whether to fold Traditional Chinese to Simplified at canonicalization
    # time (OpenCC). Disable when the corpus is intentionally bilingual and
    # script distinctions carry meaning.
    fold_traditional: bool = True

    # Maximum length (in Han characters) for an entity surface that
    # contains no bracket. Surfaces above this are rejected as
    # sentence-fragment leakage from open-set NER at low threshold.
    # Domain-tuned: insurance product names top out at ~10, legal
    # clause titles can reach 18-25 ("中华人民共和国证券法第一百四十二条"),
    # patent technique names similar. Bracketed surfaces (product codes,
    # SKU markers) are always kept regardless of length. Default 12;
    # legal / patent admins should raise it to 18-25.
    junk_max_han_chars: int = 12

    # Literal-substring backfill (KAG-style "domain mount"). NER is
    # contextual, so the same surface gets tagged on its introduction page
    # but missed on later reference pages. This pass sweeps every page
    # against the union of NER-discovered entity surfaces and adds the
    # missing entity↔passage edges. See ingestion.index.linear_rag.backfill.
    literal_backfill_enabled: bool = True
    literal_backfill_min_chars: int = 4          # drops "us", "irs"
    literal_backfill_multi_word_only: bool = True  # drops "axa", "company"

    # ===== 0-LLM ER precision gate + de-percolation (replaces reranker veto) =====
    # Recall (ANN embedding) and precision now use DIFFERENT signal classes
    # (record linkage / Fellegi-Sunter, xCoRe): a single semantic threshold
    # cannot separate template collisions ("3rd Baron Acton" vs "3rd Baroness
    # Herbert") from true variants, so the gate below is a distinct-class
    # precision matcher run as a flush-time batch. All knobs kwarg-injectable.
    # TODO admin panel: expose er_* via config_store/schema.py once tuned.

    # Mutual/reciprocal-kNN symmetrization: keep a candidate pair only if each
    # entity is in the other's top-k. Breaks single-linkage hub chaining
    # (Maier/Hein/von Luxburg). Applied as a bulk symmetrization at flush.
    er_mutual_knn: bool = True

    # IDF-weighted lexical token-overlap threshold (surface-arm regime). Tokens
    # are corpus-IDF weighted so frequent template tokens (baron/war/3rd) carry
    # little weight and rare head tokens (acton/herbert) dominate: a
    # surface-similar pair sharing only template tokens scores low → rejected.
    # Weighted Jaccard in [0,1]; tune on the intrinsic alias dev set.
    er_idf_lex_threshold: float = 0.35
    # Han surfaces have no whitespace; tokenize them as character n-grams.
    er_han_token_ngram: int = 2

    # Centroid-arm-only regime (cross-surface synonyms like US / United States):
    # surfaces differ so the lexical gate is skipped; require a high
    # mention-centroid cosine instead.
    er_ctx_synonym_threshold: float = 0.88

    # Relational must-not-link veto (Bhattacharya-Getoor collective ER): two
    # entities co-occurring in the same passage cannot be aliases. Skipped for
    # entities in more than ``er_max_df_for_veto`` passages (hubs — an ABSOLUTE
    # cap, not a percentile, keeps the veto O(N*k); hubs are handled by
    # mutual-kNN + the outdegree cap + Leiden de-percolation instead).
    er_cooccur_veto: bool = True
    er_max_df_for_veto: int = 50

    # Per-entity alias outdegree cap (SPRIG-PRUNE): keep only the top-L accepted
    # edges by score. Bounds percolation regardless of the clustering step.
    er_max_alias_degree: int = 8

    # Acceptance handler. ``overlay`` is the default reversible path
    # (alias edges only, never collapses); ``collapse_basic`` /
    # ``collapse_provenance`` collapse the canonical (the latter
    # persisting a reverse_map). Collapse modes break native
    # surface-path attribution (P4) and have non-zero rollback
    # locality (P2).
    acceptance_handler: str = "overlay"

    # Logical-entity partitioner over the (immutable) alias subgraph.
    # ``connected_components`` = raw transitive closure: single-linkage,
    # percolates to a giant component at open-domain scale (phase
    # transition in N, not a tunable tail; available as an
    # alternative). ``leiden_cpm`` = Leiden on the Constant-Potts-Model
    # objective (igraph, no new dep): the chaining-resistant principled
    # partition the ER / cross-doc-coref literature converges on.
    # Clusters are a recomputable derived view over immutable alias
    # edges, so reversibility / P1 / P4 are unchanged either way.
    #
    # Default is ``leiden_cpm`` @ resolution 0.01 (weighted): versus raw
    # connected_components it de-percolates the giant component without
    # retrieval regression. 0.01 is the least-aggressive resolution that
    # de-percolates with margin, minimising the risk of fragmenting
    # genuine multi-surface entities. ``cluster_leiden_weighted`` uses
    # the alias edge propagation weight so stronger aliases resist being
    # cut.
    cluster_algorithm: str = "leiden_cpm"
    cluster_leiden_resolution: float = 0.01
    cluster_leiden_weighted: bool = True

    # Propagation policy. Decouples per-edge audit features
    # (cos_sim / reranker_score) from the PPR-propagation weight.
    # ``cos`` uses the per-edge ``cos_sim`` directly as the propagation
    # weight (the default).
    alias_propagation_policy: str = "cos"
    alias_prop_const: float = 1.0
    alias_prop_lo: float = 0.7
    alias_prop_hi: float = 1.0
    alias_prop_tau_cos: float = 0.85
    alias_prop_tau_rerank: float = 0.7
    # Calibrated policy — z-score normalisation params + linear weights
    # for the sigmoid. Defaults are an unfitted "best-guess" prior; in
    # production they're meant to come from a fit on a held-out
    # alias-judgement set.
    # TODO admin panel: expose alias_prop_calib_* via config_store/schema.py
    # once the held-out fit pipeline lands. Kwarg-injectable only for now.
    alias_prop_calib_a: float = 1.0
    alias_prop_calib_b: float = 1.0
    alias_prop_calib_c: float = 0.0
    alias_prop_calib_cos_mean: float = 0.9
    alias_prop_calib_cos_std: float = 0.05
    alias_prop_calib_rerank_mean: float = 0.8
    alias_prop_calib_rerank_std: float = 0.1

    # GLiNER confidence calibration.
    # When ``gliner_calibration_enabled=True`` the raw GLiNER span scores
    # are temperature-scaled by ``score / gliner_temperature`` before the
    # threshold gate: effectively tightening the threshold (T>1) or
    # softening it (T<1). The temperature is fitted offline on a silver
    # span dev set (see ``experiments/ner_calibration.py``); the default
    # 1.0 = no-op (identical to current threshold-only behaviour).
    #
    # Empirically global temperature fitting on a silver dev set lands
    # at T≈1, with ECE slightly worsening — the miscalibration is
    # label-stratified, not global. Prefer ``gliner_label_thresholds``
    # for per-label correction; this knob stays OFF by default.
    #
    # TODO admin panel: expose once the per-label thresholds stabilise.
    # Kwarg-injectable only for now.
    gliner_calibration_enabled: bool = False
    gliner_temperature: float = 1.0

    # Label-specific score thresholds (label-conditional calibration).
    # Overrides ``gliner_threshold`` for the named labels. Empty dict = inert
    # (all labels use ``gliner_threshold``). ``concept`` is the noisiest label
    # in the open-set prompt — a typical insurance/legal corpus sweep shows
    # 0.5 trims ~18 pp of over-generation fuel with sub-1 pp recall loss
    # vs the global 0.3 floor. Tighten further per domain via the admin
    # panel (config_store/schema.py: ``linear_rag.gliner_label_thresholds``).
    gliner_label_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"concept": 0.5}
    )
    # Stopword-based admission filter (multilingual). GLiNER routinely
    # routes closed-class function words ('we'/'he'/'you'/'they' and the
    # Chinese equivalents) into ``person``/``organization`` because its
    # training distribution treats first-person plurals as paper authors;
    # adding a 'function word' decoy label does not score-compete on these
    # surfaces (measured empirically). This filter consults the
    # stopwords-iso multilingual lexicon and drops any NER surface whose
    # lowercased form is a stopword in any of the configured languages
    # AND whose GLiNER score is below ``gliner_stopword_confidence_floor``
    # (so high-confidence proper-noun collisions like "May" / "Will"
    # scoring >= 0.95 are preserved). Empty list disables the filter
    # (back to the per-label-noise behaviour above).
    gliner_stopword_languages: List[str] = field(default_factory=lambda: ["en", "zh"])
    gliner_stopword_confidence_floor: float = 0.95
