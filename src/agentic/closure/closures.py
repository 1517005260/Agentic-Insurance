"""Per-kind closure rules.

Pure functions of ``(obligation, claims, inventory)``. None of them
mutate state — only ``Plant.run_closure`` writes the ``CLOSED`` status.
Every rule returns either ``Closed`` (with the certified value and the
claim ids that supported it) or ``Open`` (with a short diagnostic code
and structured detail for the next acquisition turn).

Soundness boundary — what these rules certify, and what they do NOT:

* Set / count / forall / negation certify a complete partition over
  ``inventory.units(scope, unit_type)``. The atom granularity comes
  from the corpus-side InventoryStore (file / section / page /
  passage / table_row). If a question's gold answer is finer-grained
  than the corpus' atoms (e.g. each bullet inside one passage is a
  semantic item), the kernel cannot distinguish them — a ScanClaim
  is still complete *over the atom domain*. Pushing per-bullet
  enumeration into the kernel re-grows the trusted base; the right
  fix is corpus-side: split the atom in InventoryStore.

* Argmax operates on the full domain; predicate is NOT used to
  filter candidates. Candidate selection is encoded by the scope.
  The predicate slot is a sentinel (``argmax_domain``) emitted by
  the planner.

* close_negation / close_forall use the predicate the planner chose.
  A loose pattern certifies a loose statement — the gate cannot
  detect "this regex over-matches the user's intent." This is the
  attested NL→obligation faithfulness boundary; predicates are
  recorded as canonical_id in every obligation summary so they can be
  audited after the fact. For stricter control, supply pre-computed
  obligations.
"""
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union  # noqa: F401

from agentic.closure.claims import (
    Claim,
    ScanClaim,
    ValueClaim,
    WitnessClaim,
)
from agentic.closure.complete_scan import complete_scan, scan_coverage_diff
from agentic.closure.inventory import Inventory
from agentic.closure.obligation import Obligation


@dataclass(frozen=True)
class Closed:
    value: Any
    by: list[str]
    diagnostic_data: Optional[dict] = None
    closed: bool = field(default=True, init=False)


@dataclass(frozen=True)
class Open:
    reason: str
    diagnostic_data: Optional[dict] = None
    closed: bool = field(default=False, init=False)


ClosureResult = Union[Closed, Open]


# ---------------------------------------------------------------- helpers


def _witness_claims(claims: Iterable[Claim]) -> list[WitnessClaim]:
    return [c for c in claims if isinstance(c, WitnessClaim)]


def _scan_claims(claims: Iterable[Claim]) -> list[ScanClaim]:
    return [c for c in claims if isinstance(c, ScanClaim)]


def _value_claims(claims: Iterable[Claim]) -> list[ValueClaim]:
    return [c for c in claims if isinstance(c, ValueClaim)]


def _matches_predicate(claim: WitnessClaim | ValueClaim, obligation: Obligation) -> bool:
    if isinstance(claim, WitnessClaim):
        return claim.predicate.canonical_id == obligation.predicate.canonical_id
    return True


# ---------------------------------------------------------------- exists / lookup


def close_exists(o: Obligation, claims: Sequence[Claim], inventory: Inventory) -> ClosureResult:
    domain = inventory.units(o.scope, o.unit_type)
    for c in _witness_claims(claims):
        if (
            c.unit_id in domain
            and c.polarity == "positive"
            and c.predicate.canonical_id == o.predicate.canonical_id
        ):
            return Closed(value=True, by=[c.id])
    return Open("missing_witness", {"domain_size": len(domain)})


def close_lookup(o: Obligation, claims: Sequence[Claim], inventory: Inventory) -> ClosureResult:
    """Close on a ValueClaim with matching ``field`` whose evidence is
    in ``o.scope``. Two evidence shapes:

    * extracted: ``v.unit_id ∈ inventory.units(scope, unit_type)``.
    * derived:   every input claim's ``unit_id`` is itself in the
      domain (recursive — derivations can chain). The kernel already
      verified the arithmetic at ingest, so closure only re-checks
      domain residency of the underlying anchors.

    Multiple disagreeing values → ambiguous_lookup (PCN's fail-closed
    "absence of mark = uncertainty"; here, conflict = uncertainty).
    """

    if not o.field:
        return Open("missing_field", {"obligation_id": o.id})
    domain = inventory.units(o.scope, o.unit_type)
    claim_index = {c.id: c for c in _value_claims(claims)}

    candidates: list[ValueClaim] = []
    for v in claim_index.values():
        if v.field != o.field:
            continue
        if not _value_claim_in_domain(v, domain, claim_index):
            continue
        candidates.append(v)

    if not candidates:
        return Open("missing_value", {"field": o.field})

    distinct: dict[Any, list[ValueClaim]] = {}
    for v in candidates:
        distinct.setdefault(v.value, []).append(v)

    if len(distinct) == 1:
        ((value, claim_set),) = distinct.items()
        return Closed(value=value, by=[c.id for c in claim_set])
    return Open(
        "ambiguous_lookup",
        {
            "values_seen": list(distinct.keys()),
            "claim_ids": sorted({c.id for c in candidates}),
        },
    )


def _value_claim_in_domain(
    v: ValueClaim,
    domain: frozenset[str],
    claim_index: dict[str, ValueClaim],
    *,
    seen: Optional[set[str]] = None,
) -> bool:
    """Extracted: unit_id in domain. Derived: every input recursively
    grounded in ``domain``. ``seen`` guards against derivation cycles
    (defence in depth — DerivedProvenance.__post_init__ should prevent
    them, but a kernel invariant checker never trusts upstream)."""

    if not v.is_derived:
        return v.unit_id is not None and v.unit_id in domain
    if seen is None:
        seen = set()
    if v.id in seen:
        return False
    seen.add(v.id)
    for input_id in v.derived.input_claim_ids:                # type: ignore[union-attr]
        upstream = claim_index.get(input_id)
        if upstream is None:
            return False
        if not _value_claim_in_domain(upstream, domain, claim_index, seen=seen):
            return False
    return True


# ---------------------------------------------------------------- count / set


def _first_complete_scan(
    o: Obligation,
    claims: Sequence[Claim],
    inventory: Inventory,
) -> Optional[ScanClaim]:
    for c in _scan_claims(claims):
        if complete_scan(c, o, inventory):
            return c
    return None


def _scan_open_diagnostic(
    o: Obligation,
    claims: Sequence[Claim],
    inventory: Inventory,
    *,
    reason: str,
) -> Open:
    related = [
        c for c in _scan_claims(claims)
        if c.scope.canonical_scope_id == o.scope.canonical_scope_id
        and c.unit_type == o.unit_type
        and c.predicate.canonical_id == o.predicate.canonical_id
    ]
    if related:
        diff = scan_coverage_diff(related[-1], o, inventory)
        diff["candidate_scan_id"] = related[-1].id
        return Open(reason, diff)
    return Open(reason, {"domain_size": len(inventory.units(o.scope, o.unit_type))})


def close_count(o: Obligation, claims: Sequence[Claim], inventory: Inventory) -> ClosureResult:
    scan = _first_complete_scan(o, claims, inventory)
    if scan is None:
        return _scan_open_diagnostic(o, claims, inventory, reason="missing_complete_scan")
    return Closed(value=len(scan.positive_units), by=[scan.id])


def close_set(o: Obligation, claims: Sequence[Claim], inventory: Inventory) -> ClosureResult:
    scan = _first_complete_scan(o, claims, inventory)
    if scan is None:
        return _scan_open_diagnostic(o, claims, inventory, reason="missing_complete_scan")
    return Closed(value=sorted(scan.positive_units), by=[scan.id])


# ---------------------------------------------------------------- forall / negation


def close_forall(o: Obligation, claims: Sequence[Claim], inventory: Inventory) -> ClosureResult:
    domain = inventory.units(o.scope, o.unit_type)
    for c in _witness_claims(claims):
        if (
            c.unit_id in domain
            and c.polarity == "negative"
            and c.predicate.canonical_id == o.predicate.canonical_id
        ):
            return Closed(value=False, by=[c.id])
    scan = _first_complete_scan(o, claims, inventory)
    if scan is None:
        return _scan_open_diagnostic(
            o, claims, inventory,
            reason="missing_complete_scan_or_counterexample",
        )
    if not scan.negative_units:
        return Closed(value=True, by=[scan.id])
    return Closed(value=False, by=[scan.id], diagnostic_data={"counterexamples": sorted(scan.negative_units)})


def close_negation(o: Obligation, claims: Sequence[Claim], inventory: Inventory) -> ClosureResult:
    domain = inventory.units(o.scope, o.unit_type)
    for c in _witness_claims(claims):
        if (
            c.unit_id in domain
            and c.polarity == "positive"
            and c.predicate.canonical_id == o.predicate.canonical_id
        ):
            return Closed(value=False, by=[c.id])
    scan = _first_complete_scan(o, claims, inventory)
    if scan is None:
        return _scan_open_diagnostic(
            o, claims, inventory,
            reason="missing_complete_scan_or_witness",
        )
    if not scan.positive_units:
        return Closed(value=True, by=[scan.id])
    return Closed(value=False, by=[scan.id], diagnostic_data={"witnesses": sorted(scan.positive_units)})


# ---------------------------------------------------------------- argmax


def close_argmax_exact(
    o: Obligation,
    claims: Sequence[Claim],
    inventory: Inventory,
) -> ClosureResult:
    # Argmax operates on the full ``inventory.units(scope, unit_type)``;
    # the predicate does NOT filter candidates. The planner must encode
    # candidate selection in the scope itself. The predicate slot on
    # argmax obligations is a sentinel ("argmax_domain") — any value is
    # ignored at closure.
    if not o.score_field:
        return Open("missing_score_field", {"obligation_id": o.id})
    domain = inventory.units(o.scope, o.unit_type)
    if not domain:
        return Open("empty_domain", {"obligation_id": o.id})

    best_value_per_unit: dict[str, ValueClaim] = {}
    for v in _value_claims(claims):
        if v.unit_id not in domain or v.field != o.score_field:
            continue
        existing = best_value_per_unit.get(v.unit_id)
        if existing is None:
            best_value_per_unit[v.unit_id] = v
            continue
        if existing.value != v.value:
            return Open(
                "ambiguous_lookup",
                {"unit_id": v.unit_id, "values_seen": [existing.value, v.value]},
            )

    missing = sorted(domain - best_value_per_unit.keys())
    if missing:
        return Open("missing_value", {"missing": missing, "field": o.score_field})

    items = list(best_value_per_unit.items())
    try:
        max_value = max(v.value for _, v in items)
    except TypeError as exc:
        return Open("invalid_score", {"error": str(exc), "field": o.score_field})
    winners = [unit_id for unit_id, v in items if v.value == max_value]
    if len(winners) > 1:
        return Open("argmax_tie", {"tied_units": sorted(winners), "value": max_value})

    winner = winners[0]
    used = [v.id for _, v in items]
    return Closed(
        value={"unit_id": winner, "score": max_value, "field": o.score_field},
        by=used,
    )


# ---------------------------------------------------------------- dispatch


CLOSURE_RULES = {
    "exists": close_exists,
    "lookup": close_lookup,
    "count": close_count,
    "set": close_set,
    "forall": close_forall,
    "negation": close_negation,
    "argmax": close_argmax_exact,
}


def try_close(
    obligation: Obligation,
    claims: Sequence[Claim],
    inventory: Inventory,
) -> ClosureResult:
    rule = CLOSURE_RULES.get(obligation.kind)
    if rule is None:
        return Open("unknown_obligation_kind", {"kind": obligation.kind})
    return rule(obligation, claims, inventory)
