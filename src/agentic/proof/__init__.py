"""Proof-obligation gate: stores, registries, closure rules, plant.

Public API surface used by ``src/agentic/tools/proof/`` and the
``ProofAgent``. Tools import dataclasses + Plant; everything else stays
internal.

The package is organised by subsystem:

* ``state/``     — obligation_store, transitions, challenge_store, domain_map
* ``evidence/``  — evidence store, observation normaliser, claim validator,
                   auto-extractor
* ``predicate/`` — primitive registry + algebra (AND-canonicalisation,
                   syntactic entailment)
* ``closure/``   — Γ_kind closure rules (per ObligationKind)
* ``rules/``     — decomposition + repair contract registries
* ``score/``     — score extractors (numeric / percentage / date / etc.)

Module-level re-exports below keep legacy import paths
``from agentic.proof import closure_rules / score_registry`` working
without inventing new shim modules.
"""

from agentic.proof.types import (
    Binding,
    Challenge,
    Citation,
    Claim,
    ClaimType,
    ClosureResult,
    DerivedBy,
    GateView,
    Obligation,
    ObligationKind,
    ObligationSpec,
    ObligationStatus,
    Observation,
    ObservationType,
    PredicateSpec,
    RepairKind,
    ScopeRef,
    ScoreSpec,
    ToolDiagnostic,
    UnitType,
)
from agentic.proof.plant import Plant, PlantResult

# Subsystem aliases — keep existing call sites stable while the inner
# layout evolves. ``from agentic.proof import predicate_registry as pr``
# now resolves to the ``agentic.proof.predicate`` package, which
# re-exports every public predicate symbol regardless of which inner
# module (``registry``, ``algebra``, ``evaluation``) defines it.
from agentic.proof.closure import rules as closure_rules
from agentic.proof import predicate as predicate_registry
from agentic.proof.rules import decomposition as decomposition_rules
from agentic.proof.rules import repair as repair_contracts
from agentic.proof.score import registry as score_registry
from agentic.proof.state import transitions


__all__ = [
    "Plant",
    "PlantResult",
    "Binding",
    "Challenge",
    "Citation",
    "Claim",
    "ClaimType",
    "ClosureResult",
    "DerivedBy",
    "GateView",
    "Obligation",
    "ObligationKind",
    "ObligationSpec",
    "ObligationStatus",
    "Observation",
    "ObservationType",
    "PredicateSpec",
    "RepairKind",
    "ScopeRef",
    "ScoreSpec",
    "ToolDiagnostic",
    "UnitType",
    "closure_rules",
    "decomposition_rules",
    "predicate_registry",
    "repair_contracts",
    "score_registry",
    "transitions",
]
