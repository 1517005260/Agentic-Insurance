"""Typed Γ_kind closure rules.

The package exposes one Γ per :class:`ObligationKind`
(exists / count / set / forall / negation / argmax). Each Γ lives in
:mod:`agentic.proof.closure.per_kind`; :mod:`rules` dispatches on
``obligation.spec.kind`` and translates failure kinds into
:class:`ToolDiagnostic` for the plant.
"""
from agentic.proof.closure.rules import (
    diagnose,
    evaluate_closure,
)

__all__ = [
    "diagnose",
    "evaluate_closure",
]
