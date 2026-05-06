"""Obligation: the certification contract.

States are ``OPEN`` and ``CLOSED`` only — no DECOMPOSED / RETIRED /
CHALLENGED. Decomposition happens in the planner (tool boundary), not
in the runtime kernel.

Two structured references live here, ``ScopeRef`` and ``PredicateRef``,
both with a precomputed ``canonical_*_id``. Equality flows through
those ids — opaque strings would push parsing into every consumer and
break ``complete_scan``'s membership check.
"""
import hashlib
import json
from dataclasses import dataclass, field as _dc_field
from typing import Any, Literal, Optional, Tuple


ObligationKind = Literal[
    "exists",
    "lookup",
    "count",
    "set",
    "forall",
    "negation",
    "argmax",
]

ObligationStatus = Literal["OPEN", "CLOSED"]

UnitType = Literal["file", "section", "page", "passage", "table_row"]


_KINDS_WITH_SCORE: frozenset[str] = frozenset({"argmax"})
_KINDS_WITH_FIELD: frozenset[str] = frozenset({"lookup"})


def _hash(parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True)
class ScopeRef:
    file_ids: Tuple[str, ...]
    section_ids: Optional[Tuple[str, ...]] = None
    canonical_scope_id: str = ""

    @classmethod
    def build(
        cls,
        file_ids: Tuple[str, ...] | list[str],
        section_ids: Optional[Tuple[str, ...] | list[str]] = None,
    ) -> "ScopeRef":
        files = tuple(sorted({str(f) for f in (file_ids or ()) if str(f).strip()}))
        sections: Optional[Tuple[str, ...]] = None
        if section_ids:
            cleaned = tuple(sorted({str(s) for s in section_ids if str(s).strip()}))
            sections = cleaned or None
        canonical = _hash({"files": list(files), "sections": list(sections or ())})
        return cls(file_ids=files, section_ids=sections, canonical_scope_id=canonical)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScopeRef) and self.canonical_scope_id == other.canonical_scope_id

    def __hash__(self) -> int:
        return hash(self.canonical_scope_id)


@dataclass(frozen=True)
class PredicateRef:
    name: str
    args: Tuple[Tuple[str, Any], ...]
    canonical_id: str = ""

    @classmethod
    def build(cls, name: str, args: Optional[dict] = None) -> "PredicateRef":
        if not name or not name.strip():
            raise ValueError("PredicateRef.name must be non-empty.")
        # Normalise per-primitive defaults so equivalent surfaces canonicalise
        # to the same id. regex_match flags default to case-insensitive (matches
        # what pattern_search uses); contains_string defaults to case-insensitive.
        normalised = dict(args or {})
        if name == "regex_match":
            normalised.setdefault("flags", "i")
        elif name == "contains_string":
            normalised.setdefault("case_sensitive", False)
        items = tuple(sorted((str(k), v) for k, v in normalised.items()))
        canonical = _hash({"name": name, "args": [list(kv) for kv in items]})
        return cls(name=name, args=items, canonical_id=canonical)

    def args_dict(self) -> dict:
        return {k: v for k, v in self.args}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PredicateRef) and self.canonical_id == other.canonical_id

    def __hash__(self) -> int:
        return hash(self.canonical_id)


@dataclass
class Obligation:
    id: str
    kind: ObligationKind
    scope: ScopeRef
    unit_type: UnitType
    predicate: PredicateRef
    required: bool = True
    field: Optional[str] = None         # lookup
    score_field: Optional[str] = None   # argmax
    status: ObligationStatus = "OPEN"
    closed_value: Any = None
    closed_by: list[str] = _dc_field(default_factory=list)
    failure_kind: Optional[str] = None
    diagnostic_data: Optional[dict] = None

    def __post_init__(self) -> None:
        # Argmax: score_field is mandatory and must be the only "field" slot used.
        if self.kind in _KINDS_WITH_SCORE:
            if not self.score_field or not str(self.score_field).strip():
                raise ValueError(
                    f"score_field is required for kind={self.kind!r}."
                )
        elif self.score_field is not None:
            raise ValueError(
                f"score_field must be None for kind={self.kind!r}."
            )
        # Lookup: field is mandatory; other kinds must NOT carry a field.
        if self.kind in _KINDS_WITH_FIELD:
            if not self.field or not str(self.field).strip():
                raise ValueError(
                    f"field is required for kind={self.kind!r}."
                )
        elif self.field is not None:
            raise ValueError(
                f"field must be None for kind={self.kind!r}."
            )

    def structural_key(self) -> Tuple[Any, ...]:
        return (
            self.kind,
            self.scope.canonical_scope_id,
            self.unit_type,
            self.predicate.canonical_id,
            self.field,
            self.score_field,
            self.required,
        )

    def is_open(self) -> bool:
        return self.status == "OPEN"

    def is_closed(self) -> bool:
        return self.status == "CLOSED"
