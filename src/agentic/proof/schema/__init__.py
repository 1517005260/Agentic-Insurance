"""Pydantic schema layer for the proof tool boundary.

The discriminated predicate union lives in :mod:`predicates`; scope/score
shapes are in :mod:`scope`; ObligationSpec / ClaimCandidate composites
are in :mod:`obligations`; per-tool wrappers are in :mod:`tools`.
"""
from copy import deepcopy
from typing import Any, Dict

from agentic.proof.schema.compose import compose_tool_description
from agentic.proof.schema.envelope import to_envelope
from agentic.proof.schema.tools import (
    AnswerFinalizeArgsModel,
    EvidenceIngestArgsModel,
    ObligationChallengeArgsModel,
    ObligationCreateArgsModel,
    ObligationDecomposeArgsModel,
)


def flatten_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline every ``$ref`` lookup against the top-level ``$defs`` table.

    Recursive shapes (``AndPredicate`` references its own union) would
    inline forever; the in-progress set breaks cycles by re-emitting the
    ``$ref`` and keeping ``$defs`` at the root for any unresolved refs.
    """
    schema = deepcopy(schema)
    defs = schema.get("$defs", {}) or {}
    has_residual_ref = False

    def _resolve(node: Any, in_progress: frozenset[str]) -> Any:
        nonlocal has_residual_ref
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                if key in in_progress or key not in defs:
                    has_residual_ref = True
                    return {k: _resolve(v, in_progress) for k, v in node.items()}
                merged = _resolve(deepcopy(defs[key]), in_progress | {key})
                for k, v in node.items():
                    if k != "$ref":
                        merged[k] = _resolve(v, in_progress)
                return merged
            return {k: _resolve(v, in_progress) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(v, in_progress) for v in node]
        return node

    resolved = _resolve(schema, frozenset())
    if not has_residual_ref:
        resolved.pop("$defs", None)
    return resolved


__all__ = [
    "AnswerFinalizeArgsModel",
    "EvidenceIngestArgsModel",
    "ObligationChallengeArgsModel",
    "ObligationCreateArgsModel",
    "ObligationDecomposeArgsModel",
    "compose_tool_description",
    "flatten_refs",
    "to_envelope",
]
