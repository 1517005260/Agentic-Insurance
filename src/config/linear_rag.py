"""LinearRAG build-time configuration.

Lives under ``config/`` so all project configs sit in one place. Storage
paths come from ``config.settings`` — this struct only carries runtime
knobs (embedding client, GLiNER model id / labels / threshold, NER worker
count, alias-edge quality thresholds).

The ``EmbeddingClient`` import is guarded by ``TYPE_CHECKING`` so importing
``config`` doesn't pull ``model_client`` (which itself imports ``config``).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

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
    gliner_threshold: float = 0.3
    gliner_batch_size: int = 16

    max_workers: int = 4

    # Alias-edge thresholds — see disambig.DEFAULT_MIN_SIM.
    alias_top_k: int = 5
    alias_gradient: float = 0.3
    alias_min_sim: float = 0.85
    alias_min_sim_low_context: float = 0.90

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
    # SKU markers) are always kept regardless of length.
    junk_max_han_chars: int = 15

    # Literal-substring backfill (KAG-style "domain mount"). NER is
    # contextual, so the same surface gets tagged on its introduction page
    # but missed on later reference pages. This pass sweeps every page
    # against the union of NER-discovered entity surfaces and adds the
    # missing entity↔passage edges. See ingestion.index.linear_rag.backfill.
    literal_backfill_enabled: bool = True
    literal_backfill_min_chars: int = 4          # drops "us", "irs"
    literal_backfill_multi_word_only: bool = True  # drops "axa", "company"
