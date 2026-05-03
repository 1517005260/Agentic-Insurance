"""Observation normalisation package.

A single chokepoint, ``normalise(observation)``, converts every
acquisition-tool payload into one canonical
:class:`NormalisedObservation` shape that downstream proof code reads.
This eliminates the schema drift that previously had each plant
helper testing for ``page_global_id`` vs ``global_id`` vs
``file_id+page_id`` and ``text`` vs ``text_markdown`` vs ``snippet``
vs ``body``.

Public API:

* :class:`NormalisedObservation` — the canonical view.
* :class:`EntryPayload`         — per-page text + structured fields.
* :class:`ScanMeta`             — typed pattern_search metadata,
                                  including ``narrowing_sources`` so
                                  any narrowing axis disqualifies a
                                  file-level / section-level
                                  ScanClaim that would otherwise certify.
* :func:`normalise`             — payload → NormalisedObservation.
* :func:`citation_in`           — does ``citation`` reference a page
                                  the observation actually saw?
* :func:`snapshots_for`         — every text snapshot for a cited page.
* :func:`fetch_span_text`       — citation.span | first snapshot | "".
"""
from agentic.proof.evidence.observation.normaliser import (
    EntryPayload,
    NormalisedObservation,
    ScanMeta,
    citation_in,
    fetch_span_text,
    normalise,
    snapshots_for,
)

__all__ = [
    "EntryPayload",
    "NormalisedObservation",
    "ScanMeta",
    "citation_in",
    "fetch_span_text",
    "normalise",
    "snapshots_for",
]
