"""Contract — single source of truth for kind ↔ evidence ↔ tools.

Every consumer (planner, Plant, proof_claim_ingest, system prompt,
contract tests) imports from here. Do NOT inline the rules elsewhere.

This contract covers four layers:

* **Kind contract** — per ObligationKind: which predicate names are
  allowed, what claim shape closes it, which observation type the
  Plant trusts as a source.
* **Predicate args** — per primitive name: required/optional argument
  schemas. ``contains_string`` rejects regex specials; ``regex_match``
  requires a compilable, non-trivial pattern.
* **Lookup field naming** — placeholder names like ``value`` /
  ``relevance`` / ``data`` are rejected so multi-value lookups on
  the same scope can disambiguate (avoids the trap where
  ``field="value"`` collapses three distinct totals into one
  ambiguous_lookup).
* **Compute ops** — whitelisted arithmetic operations the kernel
  re-runs over closed ValueClaims when verifying a derived value.
  Aligned with PCN claim-bound numerics (arXiv:2509.06902) and PoT
  external-interpreter compute (arXiv:2211.12588) — but the kernel
  does the arithmetic itself, so no LLM-written code is trusted.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from agentic.closure.obligation import Obligation, ObligationKind, UnitType


# ---------------------------------------------------------------- predicate primitives


_EXECUTABLE_PREDICATES: frozenset[str] = frozenset({"contains_string", "regex_match"})

# Regex special characters that disqualify a contains_string pattern.
# A LLM that writes "First-in-market|market-first" should be forced to
# pick regex_match instead — these are not literal substrings.
_REGEX_SPECIALS: frozenset[str] = frozenset(r"|*+?()[]{}\^$")
_TRIVIAL_REGEXES: frozenset[str] = frozenset({".*", ".+", r"\d+", r"\w+", r"\s+", r".*?", r".+?"})


@dataclass(frozen=True)
class ArgSchema:
    """Per-primitive argument schema. ``required`` keys must be present
    AND must pass the named validator; ``optional`` keys, if present,
    must pass; everything else is rejected as foreign args."""

    required: dict[str, str]    # arg_name → validator id
    optional: dict[str, str]


PREDICATE_ARG_SCHEMAS: dict[str, ArgSchema] = {
    "contains_string": ArgSchema(
        required={"pattern": "literal_string_no_regex"},
        optional={"case_sensitive": "bool"},
    ),
    "regex_match": ArgSchema(
        required={"pattern": "compilable_regex_nontrivial"},
        optional={"flags": "regex_flags"},
    ),
    "argmax_domain": ArgSchema(required={}, optional={}),
}


def _validate_literal_string_no_regex(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return "must be a non-empty string"
    bad = [ch for ch in _REGEX_SPECIALS if ch in value]
    if bad:
        return (
            f"contains regex special chars {sorted(bad)!r}; "
            "use predicate.name='regex_match' for patterns with alternation / wildcards / groups"
        )
    return None


def _validate_compilable_regex_nontrivial(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return "must be a non-empty string"
    if value in _TRIVIAL_REGEXES:
        return f"trivial regex {value!r} matches almost everything; anchor with literal terms"
    try:
        import regex as _regex
        _regex.compile(value)
    except Exception as exc:                              # pragma: no cover
        return f"does not compile: {exc}"
    return None


def _validate_bool(value: object) -> Optional[str]:
    if not isinstance(value, bool):
        return "must be bool"
    return None


def _validate_regex_flags(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return "must be string of flags letters"
    if any(ch not in "imsux" for ch in value):
        return f"unsupported flag chars in {value!r}; use subset of 'imsux'"
    return None


_VALIDATORS: dict[str, Callable[[object], Optional[str]]] = {
    "literal_string_no_regex": _validate_literal_string_no_regex,
    "compilable_regex_nontrivial": _validate_compilable_regex_nontrivial,
    "bool": _validate_bool,
    "regex_flags": _validate_regex_flags,
}


def validate_predicate_args(name: str, args: dict) -> Optional[str]:
    spec = PREDICATE_ARG_SCHEMAS.get(name)
    if spec is None:
        return f"unknown predicate {name!r}"
    for key in spec.required:
        if key not in args:
            return f"predicate {name!r} missing required arg {key!r}"
    for key, value in args.items():
        validator_id = spec.required.get(key) or spec.optional.get(key)
        if validator_id is None:
            return f"predicate {name!r} got unexpected arg {key!r}"
        err = _VALIDATORS[validator_id](value)
        if err is not None:
            return f"predicate {name!r} arg {key!r}: {err}"
    return None


# ---------------------------------------------------------------- lookup field naming


# Reserved placeholder names — rejected because they collide across
# multi-value scopes (three lookups on field="value" → spurious
# ambiguous_lookup). Field MUST be a semantic role like
# "existing_policy_total_interest".
LOOKUP_FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "value", "relevance", "default", "field", "val", "data",
    "amount", "price", "result", "answer", "v", "x",
})


def validate_lookup_field(field: object) -> Optional[str]:
    if not isinstance(field, str) or not field.strip():
        return "field must be a non-empty string"
    cleaned = field.strip().lower()
    if cleaned in LOOKUP_FORBIDDEN_FIELDS:
        return (
            f"field={field!r} is a reserved placeholder; "
            "use a semantic role name like 'existing_policy_total_interest', "
            "'min_notional_amount', 'issue_age_max'"
        )
    if "_" not in cleaned and cleaned == cleaned.lower() and len(cleaned) < 6:
        return (
            f"field={field!r} too generic; use snake_case with semantic content "
            "(e.g. 'segregated_policy_total_interest')"
        )
    return None


# ---------------------------------------------------------------- compute operations


@dataclass(frozen=True)
class OperationSpec:
    name: str
    arity: str   # "binary" | "variadic"
    output_value_type: str
    summary: str


# Whitelisted arithmetic operations the kernel re-runs over closed
# ValueClaims (see plant._run_operation). Aligned with PCN's
# verify-at-renderer principle and PoT's separation of computation,
# but the kernel evaluates the operation itself — code_run output is
# never trusted as the source of truth.
COMPUTE_OPERATIONS: dict[str, OperationSpec] = {
    "sum": OperationSpec("sum", "variadic", "numeric",
                         "Σ inputs (numeric / integer_count)."),
    "product": OperationSpec("product", "variadic", "numeric",
                             "Π inputs (numeric)."),
    "percent_of": OperationSpec("percent_of", "binary", "numeric",
                                "input[0] × input[1] / 100 (input[1] is percentage)."),
    "difference": OperationSpec("difference", "binary", "numeric",
                                "input[0] − input[1] (numeric)."),
    "quotient": OperationSpec("quotient", "binary", "numeric",
                              "input[0] / input[1] (numeric)."),
    "max": OperationSpec("max", "variadic", "numeric", "max of numeric inputs."),
    "min": OperationSpec("min", "variadic", "numeric", "min of numeric inputs."),
}


# ---------------------------------------------------------------- KindContract


@dataclass(frozen=True)
class KindContract:
    kind: ObligationKind
    allowed_predicate_names: frozenset[str]
    requires_field: bool                 # lookup
    requires_score_field: bool           # argmax
    claim_for_evidence: tuple[str, ...]
    source_observation_types: tuple[str, ...]
    default_unit_type: UnitType
    closure_summary: str


KIND_CONTRACTS: dict[ObligationKind, KindContract] = {
    "exists": KindContract(
        kind="exists",
        allowed_predicate_names=_EXECUTABLE_PREDICATES,
        requires_field=False,
        requires_score_field=False,
        claim_for_evidence=("WitnessClaim",),
        source_observation_types=("PatternScanObservation", "PageReadObservation",
                                  "PassageReadObservation", "TableRowReadObservation"),
        default_unit_type="page",
        closure_summary="one positive WitnessClaim from a unit in scope.",
    ),
    "lookup": KindContract(
        kind="lookup",
        allowed_predicate_names=_EXECUTABLE_PREDICATES,
        requires_field=True,
        requires_score_field=False,
        claim_for_evidence=("ValueClaim",),
        source_observation_types=("PageReadObservation", "PassageReadObservation",
                                  "TableRowReadObservation"),
        default_unit_type="page",
        closure_summary="one ValueClaim with field == obligation.field; "
                        "ValueClaim may be either span-extracted OR derived (sum/percent_of/...) "
                        "over previously-closed ValueClaims.",
    ),
    "count": KindContract(
        kind="count",
        allowed_predicate_names=_EXECUTABLE_PREDICATES,
        requires_field=False,
        requires_score_field=False,
        claim_for_evidence=("ScanClaim",),
        source_observation_types=("PatternScanObservation",),
        default_unit_type="page",
        closure_summary="one complete ScanClaim; answer is len(positive_units).",
    ),
    "set": KindContract(
        kind="set",
        allowed_predicate_names=_EXECUTABLE_PREDICATES,
        requires_field=False,
        requires_score_field=False,
        claim_for_evidence=("ScanClaim",),
        source_observation_types=("PatternScanObservation",),
        default_unit_type="passage",
        closure_summary="one complete ScanClaim; answer is sorted(positive_units).",
    ),
    "forall": KindContract(
        kind="forall",
        allowed_predicate_names=_EXECUTABLE_PREDICATES,
        requires_field=False,
        requires_score_field=False,
        claim_for_evidence=("WitnessClaim", "ScanClaim"),
        source_observation_types=("PatternScanObservation", "PageReadObservation",
                                  "PassageReadObservation", "TableRowReadObservation"),
        default_unit_type="page",
        closure_summary="negative WitnessClaim closes false; "
                        "complete ScanClaim with no negative units closes true.",
    ),
    "negation": KindContract(
        kind="negation",
        allowed_predicate_names=_EXECUTABLE_PREDICATES,
        requires_field=False,
        requires_score_field=False,
        claim_for_evidence=("WitnessClaim", "ScanClaim"),
        source_observation_types=("PatternScanObservation", "PageReadObservation",
                                  "PassageReadObservation", "TableRowReadObservation"),
        default_unit_type="page",
        closure_summary="positive WitnessClaim closes false; "
                        "complete ScanClaim with no positive units closes true.",
    ),
    "argmax": KindContract(
        kind="argmax",
        allowed_predicate_names=frozenset({"argmax_domain"}),
        requires_field=False,
        requires_score_field=True,
        claim_for_evidence=("ValueClaim",),
        source_observation_types=("PageReadObservation", "PassageReadObservation",
                                  "TableRowReadObservation"),
        default_unit_type="page",
        closure_summary="one ValueClaim per domain unit on score_field; "
                        "max value wins; tie → argmax_tie.",
    ),
}


# ---------------------------------------------------------------- accessors / validation


def contract_for(kind: ObligationKind) -> KindContract:
    spec = KIND_CONTRACTS.get(kind)
    if spec is None:
        raise KeyError(f"no contract for kind={kind!r}")
    return spec


def allowed_predicates_for(kind: ObligationKind) -> frozenset[str]:
    return contract_for(kind).allowed_predicate_names


def claim_for(kind: ObligationKind) -> tuple[str, ...]:
    return contract_for(kind).claim_for_evidence


def source_observation_types_for(kind: ObligationKind) -> tuple[str, ...]:
    return contract_for(kind).source_observation_types


def default_unit_type_for(kind: ObligationKind) -> UnitType:
    return contract_for(kind).default_unit_type


def validate_obligation(o: Obligation) -> Optional[str]:
    """Return None if ``o`` is contract-valid, else a short error code."""

    spec = KIND_CONTRACTS.get(o.kind)
    if spec is None:
        return "unknown_kind"
    if o.unit_type not in {"page", "passage", "table_row"}:
        return "unsupported_unit_type"
    if o.predicate.name not in spec.allowed_predicate_names:
        return "unsupported_predicate"
    arg_err = validate_predicate_args(o.predicate.name, o.predicate.args_dict())
    if arg_err is not None:
        return f"predicate_args_invalid:{arg_err}"
    if spec.requires_field:
        if not getattr(o, "field", None):
            return "missing_field"
        field_err = validate_lookup_field(o.field)
        if field_err is not None:
            return f"field_invalid:{field_err}"
    elif getattr(o, "field", None):
        return "field_not_allowed"
    if spec.requires_score_field and not o.score_field:
        return "missing_score_field"
    if not spec.requires_score_field and o.score_field:
        return "score_field_not_allowed"
    return None


# ---------------------------------------------------------------- prompt rendering


def render_contract_summary() -> str:
    rows = ["kind      | predicate (args)                         | evidence    | source observation     | summary"]
    for kind, spec in KIND_CONTRACTS.items():
        preds = []
        for name in sorted(spec.allowed_predicate_names):
            schema = PREDICATE_ARG_SCHEMAS.get(name)
            args = ",".join(schema.required) if schema and schema.required else "-"
            preds.append(f"{name}({args})")
        rows.append(
            f"{kind:<9} | "
            f"{'/'.join(preds):<40} | "
            f"{'/'.join(spec.claim_for_evidence):<11} | "
            f"{'/'.join(t.replace('Observation','') for t in spec.source_observation_types):<22} | "
            f"{spec.closure_summary}"
        )
    return "\n".join(rows)


def render_compute_operations() -> str:
    rows = ["operation     | arity     | summary"]
    for name, spec in COMPUTE_OPERATIONS.items():
        rows.append(f"{name:<13} | {spec.arity:<9} | {spec.summary}")
    return "\n".join(rows)


__all__ = [
    "ArgSchema",
    "COMPUTE_OPERATIONS",
    "KIND_CONTRACTS",
    "KindContract",
    "LOOKUP_FORBIDDEN_FIELDS",
    "OperationSpec",
    "PREDICATE_ARG_SCHEMAS",
    "allowed_predicates_for",
    "claim_for",
    "contract_for",
    "default_unit_type_for",
    "render_compute_operations",
    "render_contract_summary",
    "source_observation_types_for",
    "validate_lookup_field",
    "validate_obligation",
    "validate_predicate_args",
]
