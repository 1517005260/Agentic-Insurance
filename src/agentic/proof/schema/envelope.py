"""Adapter: pydantic ``ValidationError`` → plant-style envelope.

Thin wrapper over :func:`agentic.proof.errors.from_validation_error`
so the pydantic boundary and the plant inline rejections produce the
exact same envelope shape (same ``code``, same ``valid_example``
registry, same ``affected_fields`` roll-up). All registry knowledge
lives in :mod:`agentic.proof.errors`; this module only translates.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError

from agentic.proof.errors import from_validation_error


def _collect_field_keys(model: type[BaseModel], _seen: Optional[set] = None) -> Dict[tuple, list]:
    """Walk a pydantic model and collect the allowed field keys at
    every nested level. Output keys are tuples of model-field names
    (e.g. ``("spec", "scope")``); values are the list of fields the
    pydantic model declares at that level.

    Used to produce ``did_you_mean`` candidates when a payload carries
    an unknown key under ``extra='forbid'``. Recurses into nested
    BaseModel fields and List[BaseModel] item types.
    """
    seen = _seen if _seen is not None else set()
    out: Dict[tuple, list] = {}
    if model in seen:
        return out
    seen.add(model)
    fields = list(model.model_fields.keys())
    out[()] = fields
    for name, info in model.model_fields.items():
        annotation = info.annotation
        nested = _unwrap_basemodel(annotation)
        if nested is None:
            continue
        sub = _collect_field_keys(nested, seen)
        for key, val in sub.items():
            out[(name, *key)] = val
    return out


def _unwrap_basemodel(annotation: Any) -> Optional[type[BaseModel]]:
    """Return the BaseModel class inside ``annotation`` if present."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    args = getattr(annotation, "__args__", None) or ()
    for arg in args:
        nested = _unwrap_basemodel(arg)
        if nested is not None:
            return nested
    return None


def to_envelope(
    exc: ValidationError,
    *,
    gate: Any = None,
    model: Optional[type[BaseModel]] = None,
) -> Dict[str, Any]:
    """Convert ``exc`` to a plant-style envelope.

    ``gate`` (when supplied by the tool wrapper) goes into
    ``context.gate`` so pydantic-path errors carry the same gate
    snapshot plant-path errors do — the LLM reads one envelope shape
    regardless of which channel rejected.

    ``model`` (when supplied) lets the registry surface
    ``did_you_mean`` candidates for ``extra_forbidden`` typos.
    """
    schema_keys = _collect_field_keys(model) if model is not None else None
    return from_validation_error(exc, gate=gate, schema_keys_by_loc=schema_keys)


__all__ = ["to_envelope"]
