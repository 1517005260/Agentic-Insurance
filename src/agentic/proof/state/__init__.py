"""State-machine subsystem for the proof gate.

* :mod:`obligation_store` — append-only obligation store + ``apply_event``.
* :mod:`transitions`      — declarative ``Event`` table + closure-cone /
                            ``can_close`` graph queries.
* :mod:`challenge_store`  — pending / discharged / rejected challenges.
* :mod:`domain_map`       — virtualised ``map_over_domain`` bookkeeping
                            (canonical_keys + materialised_children).
"""
from agentic.proof.state.challenge_store import ChallengeStore
from agentic.proof.state.domain_map import (
    DomainMap,
    DomainMapStore,
    canonical_key,
)
from agentic.proof.state.obligation_store import (
    ObligationStateError,
    ObligationStore,
)
from agentic.proof.state.transitions import (
    Event,
    TransitionRule,
    ancestor_challenged,
    can_close,
    closure_cone,
    lookup,
)

__all__ = [
    "ChallengeStore",
    "DomainMap",
    "DomainMapStore",
    "Event",
    "ObligationStateError",
    "ObligationStore",
    "TransitionRule",
    "ancestor_challenged",
    "can_close",
    "canonical_key",
    "closure_cone",
    "lookup",
]
