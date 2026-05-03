"""Evidence-ingestion subsystem.

* :mod:`store`           — append-only ``EvidenceStore`` (observations,
                           claims, bindings).
* :mod:`extractors`      — auto-extract ScanClaim from
                           PAGE_HITS_EXHAUSTIVE observations.
* :mod:`claim_validator` — validate LLM-proposed claim_candidates
                           against citation snapshots and inventory.
* :mod:`observation`     — payload → :class:`NormalisedObservation`
                           chokepoint with typed :class:`ScanMeta` and
                           ``narrowing_sources`` detection.
"""
from agentic.proof.evidence.claim_validator import validate_claim_candidate
from agentic.proof.evidence.extractors import (
    auto_extract,
    _aggregate_page_hits_to_units,
)
from agentic.proof.evidence.observation import (
    EntryPayload,
    NormalisedObservation,
    ScanMeta,
    citation_in,
    fetch_span_text,
    normalise,
    snapshots_for,
)
from agentic.proof.evidence.store import EvidenceStore

__all__ = [
    "EntryPayload",
    "EvidenceStore",
    "NormalisedObservation",
    "ScanMeta",
    "_aggregate_page_hits_to_units",
    "auto_extract",
    "citation_in",
    "fetch_span_text",
    "normalise",
    "snapshots_for",
    "validate_claim_candidate",
]
