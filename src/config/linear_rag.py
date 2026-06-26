"""LinearRAG build-time configuration.

Lives under ``config/`` so all project configs sit in one place. Storage
paths come from ``config.settings`` — this struct only carries runtime
knobs (embedding client, GLiNER model id / labels / threshold, NER worker
count, surface-quality thresholds).

The ``EmbeddingClient`` import is guarded by ``TYPE_CHECKING`` so importing
``config`` doesn't pull ``model_client`` (which itself imports ``config``).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from config import settings

if TYPE_CHECKING:
    from model_client import EmbeddingClient


# Generic, env-overridable open-set NER label list. The authoritative source
# is ``config.settings.GLINER_LABELS`` (default = a generic universal palette;
# override per-domain via the ``GLINER_LABELS`` env var). Snapshotted into the
# dataclass default_factory below so each instance gets a fresh copy and admin
# overrides don't leak across requests. ``config.settings`` must NOT import this
# module — the dependency is one-way (config → settings) to avoid a cycle.
_DEFAULT_GLINER_LABELS: List[str] = list(settings.GLINER_LABELS)


@dataclass
class LinearRAGConfig:
    embedding_client: Optional["EmbeddingClient"] = None

    # GLiNER NER configuration. ``gliner_model_id`` is a HuggingFace repo
    # id; weights live in the standard HF cache (``~/.cache/huggingface/hub/``)
    # so swapping the model id picks up a different checkpoint without
    # touching ``STORAGE_PATH``.
    #
    # ``gliner_labels`` is the open-set label prompt. The default is a
    # GENERIC universal palette (sourced from ``settings.GLINER_LABELS``);
    # override it per-domain via the ``GLINER_LABELS`` env var or this field
    # (e.g. ``["disease", "drug", "procedure"]`` for medical) without
    # touching the codebase. Lowercase, single words or short phrases.
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
    # not a hand-rolled surface filter. Default = ``settings.GLINER_NOISE_LABELS``
    # (env-overridable via ``GLINER_NOISE_LABELS``); listed members MUST
    # also appear in ``gliner_labels``.
    gliner_noise_labels: List[str] = field(
        default_factory=lambda: list(settings.GLINER_NOISE_LABELS)
    )
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

    # Adjacent-passage edges. A corpus of independent single-page
    # documents (one file_id per passage) can turn these off — they would
    # only link arbitrary corpus neighbours. Set per-build via
    # LinearRAGConfig from the dataset driver; the algorithm layer never
    # hard-codes a domain default.
    adjacent_passage_edges_enabled: bool = True

    # Whether to fold Traditional Chinese to Simplified at canonicalization
    # time (OpenCC). Disable when the corpus is intentionally bilingual and
    # script distinctions carry meaning.
    fold_traditional: bool = True

    # Emit EvidenceFS — the shell-operable evidence filesystem — at the end of
    # every flush_all(). On by default so a production build always materializes
    # the agent-readable evidence FS; turn off for the text-benchmark / bulk
    # paths that don't have a combined.md corpus to anchor against.
    evidence_fs_enabled: bool = True

    # Maximum length (in Han characters) for an entity surface that
    # contains no bracket. Surfaces above this are rejected as
    # sentence-fragment leakage from open-set NER at low threshold.
    # Domain-tuned: insurance product names top out at ~10, legal
    # clause titles can reach 18-25 ("中华人民共和国证券法第一百四十二条"),
    # patent technique names similar. Bracketed surfaces (product codes,
    # SKU markers) are always kept regardless of length. Default 12;
    # legal / patent admins should raise it to 18-25.
    junk_max_han_chars: int = 12

    # Query-time PPR gazetteer surface filters (GraphPPRChannel). NER is
    # contextual, so the same surface gets tagged on its introduction page
    # but missed on later reference pages; the channel sweeps the question's
    # candidate passages against the union of NER-discovered entity surfaces
    # to rescue those literal misses. These two knobs bound the gazetteer
    # surface set. See ingestion.index.linear_rag.backfill.
    literal_backfill_min_chars: int = 4          # drops "us", "irs"
    literal_backfill_multi_word_only: bool = True  # drops "axa", "company"

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
