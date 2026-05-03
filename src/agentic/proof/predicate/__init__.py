"""Predicate algebra: 10 primitives + AND composite.

The package is laid out as:

* ``registry.py``     — ``PrimitiveSchema`` dataclass, registry dict,
  ``primitives()``, ``build_spec()`` and ``PredicateError``.
* ``algebra.py``      — AND canonicalisation (``build_and_spec``),
  hashing (``serialize_spec``), entailment (``predicate_entails``),
  structural classification (``is_structural`` /
  ``has_content_conjunct``).
* ``evaluation.py``   — runtime dispatcher (``evaluate``) plus
  applicability helpers used by the plant.
* ``helpers.py``      — small utilities (string coercion, dotted-path
  field reads, ISO-date parsing, the regex universality blacklist).
* ``primitives/``     — one module per registered primitive.
"""
from agentic.proof.predicate.algebra import (
    build_and_spec,
    has_content_conjunct,
    is_structural,
    predicate_entails,
    serialize_spec,
)
from agentic.proof.predicate.evaluation import (
    applicable_to_obligation_unit_type,
    evaluate,
    predicate_applicable,
)
from agentic.proof.predicate.registry import (
    PredicateError,
    PrimitiveSchema,
    build_spec,
    primitives,
)

__all__ = [
    "PredicateError",
    "PrimitiveSchema",
    "applicable_to_obligation_unit_type",
    "build_and_spec",
    "build_spec",
    "evaluate",
    "has_content_conjunct",
    "is_structural",
    "predicate_applicable",
    "predicate_entails",
    "primitives",
    "serialize_spec",
]
