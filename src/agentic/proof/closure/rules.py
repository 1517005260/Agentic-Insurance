"""Γ_kind dispatch + plant-facing diagnostic translation.

Each Γ_kind closure lives in :mod:`agentic.proof.closure.per_kind`;
this module only picks the right one for an obligation and turns a
failure_kind string into the structured ``ToolDiagnostic`` the plant
hands back to the agent.

The plant continues to import ``_scope_compatible`` from this module
(``closure_rules._scope_compatible``); the real definition lives in
:mod:`agentic.proof.closure.helpers` and is re-exported here so that
internal call site keeps working without touching plant.py.
"""
from typing import Sequence

from agentic.proof.closure.helpers import _scope_compatible
from agentic.proof.closure.per_kind import (
    gamma_argmax,
    gamma_count,
    gamma_exists,
    gamma_forall,
    gamma_negation,
    gamma_set,
)
from agentic.proof.types import (
    Claim,
    ClosureResult,
    Obligation,
    ObligationKind,
    ToolDiagnostic,
)
from storage.inventory_store import InventoryStore


_GAMMA = {
    ObligationKind.EXISTS: gamma_exists,
    ObligationKind.COUNT: gamma_count,
    ObligationKind.SET: gamma_set,
    ObligationKind.FORALL: gamma_forall,
    ObligationKind.NEGATION: gamma_negation,
    ObligationKind.ARGMAX: gamma_argmax,
}


def evaluate_closure(
    obligation: Obligation,
    claims: Sequence[Claim],
    inventory: InventoryStore,
) -> ClosureResult:
    """Pick the right Γ_kind closure rule and run it."""
    rule = _GAMMA.get(obligation.spec.kind)
    if rule is None:
        return ClosureResult(success=False, diagnostic=f"unknown_kind:{obligation.spec.kind}")
    return rule(obligation, claims, inventory)


_SUGGESTED_TOOLS_BY_FAILURE = {
    "missing_witness":           ["semantic_search", "bm25_search", "graph_explore", "read_page"],
    "missing_scan":              ["pattern_search"],
    "unsealed_scope":            ["obligation_challenge"],
    "missing_comparison":        ["read_page", "evidence_ingest"],
    "missing_score_ref":         ["obligation_challenge"],
    "empty_domain":              ["obligation_challenge", "list_files", "toc"],
    "unsupported_score_field":   ["obligation_challenge"],
    "low_confidence_segmentation": ["toc", "obligation_challenge"],
    "section_level_scan_unsupported": ["toc", "obligation_challenge"],
    "argmax_tie_unresolved":     ["obligation_challenge"],
    "positive_witness_found":    ["obligation_challenge"],
    "counterexample_found":      ["obligation_challenge"],
}


# When the failure_kind has a unique, well-defined repair pathway, the
# diagnostic surfaces the exact repair_kind so the LLM doesn't have to
# rederive it. Empty here means "the LLM picks one of the suggested
# tools — no canonical repair_kind".
_SUGGESTED_REPAIR_KIND_BY_FAILURE = {
    "unsealed_scope":            "scope_too_narrow",
    "missing_score_ref":         "wrong_question_kind",
    "unsupported_score_field":   "wrong_question_kind",
    "empty_domain":              "scope_too_narrow",
    "low_confidence_segmentation": "scope_too_broad",
    "section_level_scan_unsupported": "scope_too_broad",
    "positive_witness_found":    "wrong_question_kind",
    "counterexample_found":      "wrong_question_kind",
}


def diagnose(failure_kind: str, obligation_id: str) -> ToolDiagnostic:
    """Translate a Γ failure_kind into a structured hint for gate.diagnose."""
    return ToolDiagnostic(
        obligation_id=obligation_id,
        failure_kind=failure_kind,
        suggested_tools=list(_SUGGESTED_TOOLS_BY_FAILURE.get(failure_kind, [])),
        suggested_repair_kind=_SUGGESTED_REPAIR_KIND_BY_FAILURE.get(failure_kind),
    )


__all__ = [
    "_scope_compatible",
    "diagnose",
    "evaluate_closure",
]
