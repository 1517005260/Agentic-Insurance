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
    # Note: ``currency`` is intentionally OFF by default — the empirical
    # benchmark surfaced "加元" / "英镑" entering the entity universe as
    # spurious nodes, which polluted PPR neighbourhoods. Admins running
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
    # 0.3 was empirically the best-recall/lowest-noise setting on the
    # 4-document insurance corpus we benchmarked against.
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
    # punctuation. 80 is a defensive cap — measured longest legitimate
    # legal/insurance surface in the benchmark sits at ~50 chars.
    ner_max_span_chars: int = 80

    max_workers: int = 4

    # How often LinearRAG.index() persists LinearRAG.graphml, in
    # index() calls. Default 1 = write every doc (bit-identical to the
    # pre-cadence behaviour; the per-file API builder makes a fresh
    # instance per file so its counter is always 1 → unchanged). A
    # persistent bulk driver (one LinearRAG over a whole corpus, e.g.
    # GraphIndexBuilder(reuse_graph=True)) sets this >1 so the O(V+E)
    # graphml (de)serialisation is amortised across docs instead of
    # paid every doc (the 650-build O(N²) wall-time blow-up). Such a
    # driver must force a final flush_graphml() at the end and before
    # any checkpoint that reads the on-disk graphml.
    graphml_flush_every: int = 1

    # How often LinearRAG.index() recomputes the expensive Leiden
    # (compute_clusters) partition for the returned ``cluster_shape``,
    # in index() calls. Default 1 = compute every doc (bit-identical to
    # the pre-cadence behaviour; the per-file API builder makes a fresh
    # instance per file so its counter is always 1 → unchanged). Leiden
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

    # Alias-edge thresholds — see disambig.DEFAULT_MIN_SIM.
    # The dual-query recall path (bare-surface + centroid) needs
    # headroom for sequence-/pluralization-/abbreviation-level variants
    # to make the cut; the gradient cutoff still trims unrelated
    # long-tail candidates.
    alias_top_k: int = 20
    alias_gradient: float = 0.3
    alias_min_sim: float = 0.85

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
    # SKU markers) are always kept regardless of length. 12 was chosen
    # from a 56-sample benchmark (precision 95.5%, recall 52.5%, one
    # false-positive on the boundary clause "保单分拆预设指示权益条款");
    # legal / patent admins should raise it back to 18-25.
    junk_max_han_chars: int = 12

    # Literal-substring backfill (KAG-style "domain mount"). NER is
    # contextual, so the same surface gets tagged on its introduction page
    # but missed on later reference pages. This pass sweeps every page
    # against the union of NER-discovered entity surfaces and adds the
    # missing entity↔passage edges. See ingestion.index.linear_rag.backfill.
    literal_backfill_enabled: bool = True
    literal_backfill_min_chars: int = 4          # drops "us", "irs"
    literal_backfill_multi_word_only: bool = True  # drops "axa", "company"

    # Reranker veto layer — final gate on alias-edge creation.
    # The dual-query gradient_topk recall is generous (top-K=20); a
    # pairwise cross-encoder scores each surviving (anchor, candidate)
    # pair and rejects below ``reranker_threshold``. The score is true
    # pairwise (Qwen3-Reranker yes/no logit), so the threshold is a
    # stable absolute boundary across calls. AUC vs hand-labelled set
    # is ~0.66 — high enough to use as a low-confidence veto, NOT as
    # an identity classifier; high scores do not auto-confirm alias on
    # their own.
    reranker_enabled: bool = True
    reranker_threshold: float = 0.7
    # ER-specific instruction. Spelled out as a hard-negative checklist
    # because the model defaults to retrieval relevance ("are these
    # topically related") and we need identity ("are these the same
    # entity"); the negative-class enumeration is the difference.
    reranker_instruction: str = (
        "Score yes only if the two strings refer to the exact same real-world "
        "entity or accepted alias (synonym, abbreviation, pluralization, "
        "traditional/simplified Chinese variant). Reply no for any of: "
        "related-but-distinct concepts; broader/narrower scope; ordered "
        "options or tiers; negation or quantifier flips; modifier-introduced "
        "variants; action vs entity phrases; sentence fragments."
    )

    # Acceptance handler. ``overlay`` is the default reversible path
    # (alias edges only, never collapses); ``collapse_basic`` /
    # ``collapse_provenance`` are the B7a / B7b baselines (canonical
    # absorbs members, reverse_map persisted). Collapse modes break
    # native surface-path attribution (P4) and have non-zero rollback
    # locality (P2) — only flip in for ablation experiments.
    acceptance_handler: str = "overlay"

    # Logical-entity partitioner over the (immutable) alias subgraph.
    # ``connected_components`` = raw transitive closure: single-linkage,
    # percolates to a giant component at open-domain scale (phase
    # transition in N, not a tunable tail; kept for ablation /
    # bit-compat). ``leiden_cpm`` = Leiden on the Constant-Potts-Model
    # objective (igraph, no new dep): the chaining-resistant principled
    # partition the ER / cross-doc-coref literature converges on.
    # Clusters are a recomputable derived view over immutable alias
    # edges, so reversibility / P1 / P4 are unchanged either way.
    #
    # Default is ``leiden_cpm`` @ resolution 0.01 (weighted), chosen by
    # a controlled retrieval A/B on a 154-doc open-domain stock: vs raw
    # connected_components it cut largest_cc_ratio 0.335→0.0021 (G5 PASS,
    # 10× margin; giant component 24717→153 entities) with **no
    # retrieval regression** (ranked Page Recall@10 identical 0.828,
    # Recall@5 within run noise). 0.01 is the least-aggressive
    # resolution that clears G5 with margin → minimal risk of
    # fragmenting genuine multi-surface entities. ``cluster_leiden_
    # weighted`` uses the alias edge propagation weight so stronger
    # aliases resist being cut.
    cluster_algorithm: str = "leiden_cpm"
    cluster_leiden_resolution: float = 0.01
    cluster_leiden_weighted: bool = True

    # Propagation policy. Decouples per-edge audit features
    # (cos_sim / reranker_score) from the PPR-propagation weight.
    # ``cos`` matches the historical ``weight = cos_sim`` behaviour,
    # so the default is bit-stable with pre-v0.5 ingest.
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
    # once the held-out fit pipeline lands. Currently kwarg-injectable
    # only (memory: feedback_admin_panel_all_tunables.md).
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
    # A/B result (2026-05-18, 154-doc stock, 385-span silver dev):
    # T=1.035≈1.0; ECE 0.052→0.057 (worse); over-gen 50.3%→50.3%;
    # gates FAIL → default stays OFF. Root cause: miscalibration is
    # label-stratified, not global; use gliner_label_thresholds instead.
    #
    # TODO admin panel: expose once label-threshold A/B is done.
    # Currently kwarg-injectable only (memory: feedback_admin_panel_all_tunables.md).
    gliner_calibration_enabled: bool = False
    gliner_temperature: float = 1.0

    # Label-specific score thresholds (label-conditional calibration).
    # Overrides ``gliner_threshold`` for the named labels. Empty dict = inert
    # (all labels use ``gliner_threshold``). Data-driven from 2026-05-18
    # concept-threshold sweep on 154-doc stock (ranked Page Recall@10 guardrail,
    # ≤1 pp tolerance vs C2 frozen 0.914 reference):
    #
    #   concept@0.3 baseline: R@10=0.8284, overgen=50.3%
    #   concept@0.5:          R@10=0.8206 (−0.78pp PASS), overgen=32.1% (−18.2pp)
    #   concept@0.6:          R@10=0.8179 (−1.05pp FAIL)
    #
    # concept@0.5 is the best passing threshold: −18pp over-generation-fuel cut,
    # −0.78pp recall (within tolerance), guardrail PASS.
    # Data: /root/autodl-tmp/_exp/ner_label_thr_sweep.json (2026-05-18).
    # Admin panel: exposed via config_store/schema.py (linear_rag.gliner_label_thresholds).
    gliner_label_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"concept": 0.5}
    )
