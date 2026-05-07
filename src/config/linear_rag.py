"""LinearRAG build-time configuration.

Lives under ``config/`` so all project configs sit in one place. Storage
paths come from ``config.settings`` — this struct only carries runtime
knobs (embedding client, spaCy model path, NER worker count, alias-edge
quality thresholds).

The ``EmbeddingClient`` import is guarded by ``TYPE_CHECKING`` so importing
``config`` doesn't pull ``model_client`` (which itself imports ``config``).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from model_client import EmbeddingClient


@dataclass
class LinearRAGConfig:
    embedding_client: Optional["EmbeddingClient"] = None
    # spaCy model paths — ``spacy_model`` is the default / English pipeline;
    # ``zh_spacy_model`` is loaded when provided and is selected per passage
    # via langdetect (Han ideograph → zh, else en).
    spacy_model: str = "en_core_web_trf"
    zh_spacy_model: Optional[str] = None
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

    # Literal-substring backfill (KAG-style "domain mount"). spaCy NER is
    # contextual, so the same surface gets tagged on its introduction page
    # but missed on later reference pages. This pass sweeps every page
    # against the union of NER-discovered entity surfaces and adds the
    # missing entity↔passage edges. See ingestion.index.linear_rag.backfill.
    # TODO(config-center): expose these to the admin tunables panel.
    literal_backfill_enabled: bool = True
    literal_backfill_min_chars: int = 4          # drops "us", "irs"
    literal_backfill_multi_word_only: bool = True  # drops "axa", "company"
