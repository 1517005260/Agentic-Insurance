"""Declarative rule registries.

* :mod:`decomposition` — ``RuleContract`` per (and_split / scope_partition /
                         case_split / map_over_domain).
* :mod:`repair`        — ``RepairContract`` per (scope_too_narrow /
                         scope_too_broad / predicate_mismatch /
                         missing_subobligation / wrong_question_kind).
"""
from agentic.proof.rules import decomposition, repair
from agentic.proof.rules.decomposition import (
    RuleContract,
    all_rule_ids,
    validate_rule,
)
from agentic.proof.rules.repair import (
    RepairContract,
    all_repair_kinds,
    envelope_check,
    validate_replacement,
)

__all__ = [
    "RepairContract",
    "RuleContract",
    "all_repair_kinds",
    "all_rule_ids",
    "decomposition",
    "envelope_check",
    "repair",
    "validate_replacement",
    "validate_rule",
]
