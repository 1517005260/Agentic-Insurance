"""Shared helpers used by the per-kind Γ closures.

These utilities answer questions every Γ_kind needs to ask about the
obligation's domain: which units are in scope, whether section-level
scans are sound for that scope, whether a claim's scope is broad
enough to count, which scans tile the universe, and whether the scope
is sealed against silent additions.

Section-level soundness depends on segmentation quality. Sound scans
require every section in scope to be ``confidence >= medium`` AND
``is_page_exclusive``. ΓNegation on the active root tightens this to
``confidence == high`` because false absence is the most damaging
failure mode. v1's ``heading_extracted`` provenance caps confidence at
``"medium"``, so root-negation queries against fresh inventories will
refuse — by design, not by accident.
"""
from typing import List, Optional, Sequence

from agentic.proof import predicate as pr
from agentic.proof.types import (
    Claim,
    ClaimType,
    Obligation,
    ScopeRef,
)
from storage.inventory_store import InventoryStore, Section


def _scope_units(obligation: Obligation, inventory: InventoryStore) -> List[str]:
    """Domain over which closure must hold."""
    return inventory.units(
        obligation.spec.unit_type,
        file_ids=list(obligation.spec.scope.file_ids) or None,
        section_ids=list(obligation.spec.scope.section_ids) if obligation.spec.scope.section_ids else None,
    )


def _section_objects(scope: ScopeRef, inventory: InventoryStore) -> List[Section]:
    out: List[Section] = []
    if scope.section_ids is not None:
        for sid in scope.section_ids:
            sec = inventory.get(sid)
            if sec is not None:
                out.append(sec)
        return out
    # File-level scope: collect every section under each file.
    for fid in scope.file_ids:
        out.extend(inventory.sections_for_file(fid))
    return out


def _section_scope_ok(scope: ScopeRef, inventory: InventoryStore, *, require_high: bool = False
                      ) -> Optional[str]:
    """Validate the section confidence + exclusivity guard.

    Returns ``None`` if the scope can sustain a section-level scan
    certificate; otherwise returns a diagnostic string the plant can
    surface in gate.diagnose. When ``require_high`` is True (negation
    on the active root) we require ``confidence == "high"``; otherwise
    ``>= "medium"`` is enough.
    """
    sections = _section_objects(scope, inventory)
    if not sections:
        return "empty_section_universe"
    rank = {"low": 0, "medium": 1, "high": 2}
    threshold = 2 if require_high else 1
    for sec in sections:
        if rank.get(sec.confidence, 0) < threshold:
            return "low_confidence_segmentation"
        if not sec.is_page_exclusive:
            return "section_level_scan_unsupported"
    return None


def _scope_compatible(claim_scope: ScopeRef, obligation_scope: ScopeRef) -> bool:
    """A claim's scope is compatible with an obligation's scope iff the
    claim covers a superset (file-level claim covers all sections in
    those files; section-level claim must list the obligation's
    sections).

    For ScanClaim used to close completeness obligations the scope must
    match the obligation's domain exactly. For witness claims the
    scope only has to contain the witnessed units.
    """
    if not obligation_scope.file_ids.issubset(claim_scope.file_ids):
        return False
    if claim_scope.section_ids is None and obligation_scope.section_ids is None:
        return True
    if claim_scope.section_ids is None:
        return True   # file-level claim subsumes section-level obligation
    if obligation_scope.section_ids is None:
        return False
    return obligation_scope.section_ids.issubset(claim_scope.section_ids)


def _aligned_complete_scans(
    obligation: Obligation,
    claims: Sequence[Claim],
    inventory: InventoryStore,
) -> List[Claim]:
    """Return ScanClaims that exactly cover the obligation's domain.

    ``positive ∪ negative`` must equal the inventory units; intersection
    must be empty. The predicate_entails check uses claim ⊨ obligation
    so a stronger claim predicate is fine.
    """
    out: List[Claim] = []
    domain = set(_scope_units(obligation, inventory))
    for claim in claims:
        if claim.claim_type != ClaimType.SCAN:
            continue
        if claim.unit_type != obligation.spec.unit_type:
            continue
        if claim.predicate is None:
            continue
        if not pr.predicate_entails(claim.predicate, obligation.spec.predicate):
            continue
        if not _scope_compatible(claim.scope, obligation.spec.scope):
            continue
        positive = set(claim.positive_units)
        negative = set(claim.negative_units)
        if positive & negative:
            continue
        if positive | negative != domain:
            continue
        out.append(claim)
    return out


def _is_sealed(obligation: Obligation) -> bool:
    """Sealed scope is the soundness anchor for ΓNegation. Either the
    scope was created sealed, or the agent explicitly accepted the
    weaker ``sealed_scope_override`` knob."""
    return obligation.spec.scope.sealed or obligation.spec.sealed_scope_override
