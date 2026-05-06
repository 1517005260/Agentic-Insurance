"""``complete_scan`` — the single completeness predicate.

Count, set, forall-true, and negation-true all reduce to a complete
finite-domain classification check. Keeping the check in one place is
the reason the per-kind closure surface stays small.

Returns a plain bool; the caller decides what diagnostic to emit on
False (which invariant failed, which units remain uncovered).
"""
from agentic.closure.claims import ScanClaim
from agentic.closure.inventory import Inventory
from agentic.closure.obligation import Obligation


def complete_scan(claim: ScanClaim, obligation: Obligation, inventory: Inventory) -> bool:
    if claim.scope.canonical_scope_id != obligation.scope.canonical_scope_id:
        return False
    if claim.unit_type != obligation.unit_type:
        return False
    if claim.predicate.canonical_id != obligation.predicate.canonical_id:
        return False
    if not claim.exhaustive:
        return False
    domain = inventory.units(obligation.scope, obligation.unit_type)
    if claim.scanned_units != domain:
        return False
    if (claim.positive_units | claim.negative_units) != domain:
        return False
    if claim.positive_units & claim.negative_units:
        return False
    return True


def scan_coverage_diff(
    claim: ScanClaim,
    obligation: Obligation,
    inventory: Inventory,
) -> dict:
    """Diagnostic helper for ``Open`` reasons — never used inside ``complete_scan``."""

    domain = inventory.units(obligation.scope, obligation.unit_type)
    return {
        "missing_from_scan": sorted(domain - claim.scanned_units),
        "extra_in_scan": sorted(claim.scanned_units - domain),
        "double_labeled": sorted(claim.positive_units & claim.negative_units),
        "uncovered_in_scan": sorted(claim.scanned_units - (claim.positive_units | claim.negative_units)),
    }
