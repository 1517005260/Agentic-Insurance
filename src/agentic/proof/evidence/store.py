"""EvidenceStore — observations, claims, and bindings.

All three are append-only. The store is a flat container with simple
filters; sophisticated indexing is left to the plant since it knows
which lookups it actually performs (auto_bind walks claims by
scope+predicate, gate.diagnose walks by recency).

Observations are the raw tool outputs registered by ProofAgent after
each acquisition tool call. Claims are validated views of those
observations; the plant emits them via auto_extract or the LLM proposes
them via evidence_ingest. Bindings link a claim to the obligations it
helps close — auto-bound at ingest time, never rewritten.
"""
import time
import threading
import uuid
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from agentic.proof.types import (
    Binding,
    Claim,
    ClaimType,
    Observation,
    ObservationType,
    PredicateSpec,
    ScopeRef,
)


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class EvidenceStore:
    """Single-process append-only store for observations and claims."""

    def __init__(self) -> None:
        self._observations: Dict[str, Observation] = {}
        self._claims: Dict[str, Claim] = {}
        self._bindings: List[Binding] = []
        self._observations_order: List[str] = []
        self._claims_order: List[str] = []
        self._lock = threading.Lock()

    # ----------------------------------------------------------- observations

    def add_observation(
        self,
        *,
        tool_name: str,
        observation_type: ObservationType,
        payload: Dict,
        citations: List,
        timestamp: Optional[float] = None,
    ) -> Observation:
        with self._lock:
            obs_id = _short_id("obs")
            obs = Observation(
                id=obs_id,
                tool_name=tool_name,
                observation_type=observation_type,
                payload=payload,
                citations=citations,
                timestamp=timestamp if timestamp is not None else time.time(),
            )
            self._observations[obs_id] = obs
            self._observations_order.append(obs_id)
            return obs

    def get_observation(self, observation_id: str) -> Optional[Observation]:
        return self._observations.get(observation_id)

    def observations(self, *, observation_type: Optional[ObservationType] = None) -> List[Observation]:
        if observation_type is None:
            return [self._observations[i] for i in self._observations_order]
        return [
            self._observations[i] for i in self._observations_order
            if self._observations[i].observation_type == observation_type
        ]

    # ----------------------------------------------------------- claims

    def add_claim(self, claim: Claim) -> Claim:
        with self._lock:
            if not claim.id:
                claim.id = _short_id("c")
            if claim.id in self._claims:
                raise ValueError(f"duplicate claim id={claim.id}")
            self._claims[claim.id] = claim
            self._claims_order.append(claim.id)
            return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)

    def claims(
        self,
        *,
        claim_type: Optional[ClaimType] = None,
        unit_type: Optional[str] = None,
    ) -> List[Claim]:
        out: List[Claim] = []
        for cid in self._claims_order:
            c = self._claims[cid]
            if claim_type is not None and c.claim_type != claim_type:
                continue
            if unit_type is not None and c.unit_type != unit_type:
                continue
            out.append(c)
        return out

    def recent_claims(self, limit: int = 10) -> List[Claim]:
        if limit <= 0:
            return []
        return [self._claims[cid] for cid in self._claims_order[-limit:]]

    # ----------------------------------------------------------- bindings

    def add_binding(self, binding: Binding) -> Binding:
        with self._lock:
            self._bindings.append(binding)
            return binding

    def bindings_for_obligation(self, obligation_id: str) -> List[Binding]:
        return [b for b in self._bindings if b.obligation_id == obligation_id]

    def bindings_for_claim(self, claim_id: str) -> List[Binding]:
        return [b for b in self._bindings if b.claim_id == claim_id]

    def all_bindings(self) -> List[Binding]:
        return list(self._bindings)
