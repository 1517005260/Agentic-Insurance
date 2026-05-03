"""Single source of truth for proof-gate error envelopes.

Every code path that rejects an LLM payload produces a uniform
envelope:

    {
        "code":            "<plant-domain code>",
        "message":         "<one short sentence>",
        "remediation":     "<actionable repair instruction>",
        "valid_example":   <example JSON, picked from EXAMPLES by code>,
        "affected_fields": ["<top-level field>", ...],   # multi-error roll-up
        "context":         {...},                          # machine-readable extras
    }

Plant, spec_builder, handlers, domain_map_handler, and the pydantic
schema layer all funnel through ``make_envelope`` (or
``from_validation_error``). Adding a new error code = add one row to
:data:`EXAMPLES` (and optionally :data:`REMEDIATIONS`) here, never
back into a per-module branch.

Why centralise rather than keep four ``_err`` helpers:

* The audit at ``docs/audit_proof_gate_report.md`` showed pydantic
  envelopes lacked ``valid_example`` while plant ones had it — same
  failure mode, two LLM-facing shapes. A registry fixes that on every
  channel without per-channel patches.
* Mid-tier LLMs only repair on what the FIRST error tells them.
  Surfacing ``affected_fields`` at the top level lets them fix every
  typo in one turn.
* Adding a code in one place keeps the example/remediation copy in
  sync across plant inline rejections, pydantic validation, and tool
  wrappers.
"""
import difflib
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --------------------------------------------------------------- registries


# Canonical valid_example per error code. The plant inline path and the
# pydantic path both pull the SAME example so the LLM sees one shape.
EXAMPLES: Dict[str, Any] = {
    "invalid_scope":           {"file_ids": ["<file_id>"], "section_ids": None, "sealed": False},
    "scope_too_narrow":        {"file_ids": ["<file_id>"], "section_ids": None, "sealed": False},
    "unknown_file_id":         {"file_ids": ["<file_id from list_files>"]},
    "unknown_section_id":      {"section_ids": ["<file_id>:sec_001"]},
    "invalid_unit_type":       {"unit_type": "file"},
    "invalid_kind":            {"kind": "exists"},
    "invalid_polarity":        {"polarity": "positive"},
    "invalid_tie_policy":      {"tie_policy": "first"},
    "invalid_derived_by":      {"derived_by": "user_constraint"},
    "invalid_predicate":       {"name": "contains_string", "args": {"pattern": "<term>", "case_sensitive": False}},
    "invalid_score":           {"name": "numeric_amount", "args": {}},
    "missing_score_ref":       {"name": "numeric_amount", "args": {}},
    "invalid_claim_type":      {"claim_type": "WitnessClaim"},
    "invalid_citation":        {"file_id": "<file_id>", "page_id": "p_0001", "span": "<verbatim text>"},
    "missing_citations":       [{"file_id": "<file_id>", "page_id": "p_0001", "span": "<verbatim text>"}],
    "missing_value_map":       {"value_map": {"<unit_id>": 1234}},
    "missing_field":           None,
    "unexpected_field":        None,
    "invalid_argument":        None,
}


# Path-prefix → domain code. Pydantic-derived envelopes use this to
# pick the same code the plant would have emitted inline. Longest
# matching prefix wins; shorter prefixes are fallbacks.
LOC_TO_CODE: Dict[Tuple[str, ...], str] = {
    ("spec", "scope"):       "invalid_scope",
    ("spec", "predicate"):   "invalid_predicate",
    ("spec", "score"):       "invalid_score",
    ("spec", "kind"):        "invalid_kind",
    ("spec", "unit_type"):   "invalid_unit_type",
    ("spec", "tie_policy"):  "invalid_tie_policy",
    ("spec", "polarity"):    "invalid_polarity",
    ("spec", "derived_by"):  "invalid_derived_by",
    ("claim_candidate", "scope"):      "invalid_scope",
    ("claim_candidate", "predicate"):  "invalid_predicate",
    ("claim_candidate", "score"):      "invalid_score",
    ("claim_candidate", "unit_type"):  "invalid_unit_type",
    ("claim_candidate", "claim_type"): "invalid_claim_type",
    ("claim_candidate", "citations"):  "invalid_citation",
    ("scope",):                "invalid_scope",
    ("predicate",):            "invalid_predicate",
    ("score",):                "invalid_score",
    ("draft_text",):           "invalid_argument",
    ("cited_claim_ids",):      "invalid_argument",
}


# Pydantic ``error.type`` slug → plant code, used as a fallback when
# the loc-prefix table doesn't match (e.g. a top-level ``draft_text``
# string-too-short).
PYDANTIC_TYPE_TO_CODE: Dict[str, str] = {
    "missing":             "missing_field",
    "string_type":         "invalid_argument",
    "int_type":            "invalid_argument",
    "float_type":          "invalid_argument",
    "bool_type":           "invalid_argument",
    "list_type":           "invalid_argument",
    "dict_type":           "invalid_argument",
    "string_too_short":    "invalid_argument",
    "too_short":           "invalid_argument",
    "literal_error":       "invalid_argument",
    "enum":                "invalid_argument",
    "union_tag_invalid":   "invalid_argument",
    "union_tag_not_found": "missing_field",
    "value_error":         "invalid_argument",
    "extra_forbidden":     "unexpected_field",
}


def _resolve_code(loc: Tuple[Any, ...], pyd_type: str) -> str:
    """Pick the plant code by longest matching loc prefix; fall back
    to the pydantic-type table; fall back to ``invalid_argument``."""
    str_loc = tuple(p for p in loc if isinstance(p, str))
    for length in range(min(len(str_loc), 3), 0, -1):
        prefix = str_loc[:length]
        code = LOC_TO_CODE.get(prefix)
        if code is not None:
            return code
    return PYDANTIC_TYPE_TO_CODE.get(pyd_type, "invalid_argument")


# --------------------------------------------------------------- envelope builder


def make_envelope(
    code: str,
    message: str,
    *,
    remediation: Optional[str] = None,
    valid_example: Any = None,
    affected_fields: Optional[List[str]] = None,
    **context: Any,
) -> Dict[str, Any]:
    """Build a single envelope. Auto-attaches the canonical
    :data:`EXAMPLES` entry when the caller didn't override and a code
    has one — so the LLM always sees a fixable shape."""
    out: Dict[str, Any] = {
        "code": code,
        "message": message,
        "context": context,
    }
    if remediation is not None:
        out["remediation"] = remediation
    example = valid_example if valid_example is not None else EXAMPLES.get(code)
    if example is not None:
        out["valid_example"] = example
    if affected_fields:
        out["affected_fields"] = list(affected_fields)
    return out


# --------------------------------------------------------------- pydantic adapter


def _format_loc(loc: Tuple[Any, ...]) -> str:
    parts: List[str] = []
    for piece in loc:
        if isinstance(piece, int):
            parts.append(f"[{piece}]")
        else:
            parts.append(f".{piece}" if parts else str(piece))
    return "".join(parts) or "<root>"


def _normalise_errors(raw: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in raw:
        out.append({
            "loc": list(e.get("loc", ())),
            "type": e.get("type", "value_error"),
            "msg": e.get("msg", ""),
        })
    return out


def _affected_top_level(errors: List[Dict[str, Any]]) -> List[str]:
    """Distinct top-level field names across all errors. Mid-tier
    LLMs read this to fix multiple typos in one turn instead of
    looping on the first failure."""
    seen: List[str] = []
    for e in errors:
        for piece in e.get("loc", []):
            if isinstance(piece, str) and piece and piece not in seen:
                seen.append(piece)
                break
    return seen


def _typo_candidates(field: str, allowed: Iterable[str]) -> List[str]:
    """Closest-match candidates for an unexpected key (e.g. ``unit``
    → suggest ``unit_type``). Used for ``extra_forbidden`` errors."""
    return difflib.get_close_matches(field, list(allowed), n=3, cutoff=0.6)


def _remediation_for(code: str, loc_str: str, msg: str) -> str:
    if code == "missing_field":
        return f"Add the required field {loc_str}; consult the tool's schema for its exact shape."
    if code == "unexpected_field":
        return f"Drop the unknown key {loc_str}; the schema accepts only the documented fields."
    if code == "invalid_scope":
        return "Use list_files / toc to discover valid ids; scope.file_ids must be a non-empty list."
    if code == "invalid_predicate":
        return (
            "Use a registered primitive — contains_string / regex_match / field_equals / "
            "numeric_compare / date_compare / type_is / table_cell_contains / section_title_contains / "
            "range_in / list_contains / and. Each requires its own args shape (see tool description)."
        )
    if code == "invalid_score":
        return "argmax accepts numeric_amount / percentage / integer_count / date_iso. text_field is not orderable."
    if code == "invalid_unit_type":
        return "Set unit_type to 'file' or 'section'; 'page' is never accepted."
    if code == "invalid_kind":
        return "Pick one of: exists / count / set / forall / negation / argmax."
    if code == "invalid_polarity":
        return "polarity is 'positive' (default). Use kind='negation' to express absence; 'negative' is rejected."
    if code == "invalid_tie_policy":
        return "tie_policy is 'first' (default), 'all', or 'error' — argmax-only."
    if code == "invalid_derived_by":
        return "Drop `derived_by`; the plant sets it from parent_id / discharges_challenge / decomposition rule."
    if code == "invalid_claim_type":
        return "Set claim_type to 'WitnessClaim' (per-unit citation) or 'ScanClaim' (full-coverage partition)."
    if code == "invalid_citation":
        return "Each citation is {file_id, page_id, span?}; both file_id and page_id are required."
    return f"Fix {loc_str} to match the tool's schema (validator reported: {msg})."


def from_validation_error(
    exc: Any,
    *,
    gate: Any = None,
    schema_keys_by_loc: Optional[Dict[Tuple[str, ...], List[str]]] = None,
) -> Dict[str, Any]:
    """Convert a pydantic ``ValidationError`` to an envelope.

    * Picks ``code`` from the FIRST error's loc, but lists every
      affected top-level field in ``affected_fields`` so the LLM
      can see ALL issues at once.
    * Adds ``valid_example`` from :data:`EXAMPLES`.
    * For ``extra_forbidden`` errors, attaches typo candidates via
      ``schema_keys_by_loc`` (if supplied) so the LLM sees
      "you wrote 'unit', closest match is 'unit_type'".
    * Caller may pass ``gate`` (a serialisable snapshot) so the
      envelope shape matches plant rejections regardless of which
      validation channel rejected the call.
    """
    raw = list(exc.errors())
    if not raw:
        return make_envelope("invalid_argument", "validation failed (no details)")
    norm = _normalise_errors(raw)
    primary = raw[0]
    loc = tuple(primary.get("loc", ()))
    pyd_type = str(primary.get("type", "value_error"))
    code = _resolve_code(loc, pyd_type)
    loc_str = _format_loc(loc)
    msg = str(primary.get("msg", "validation failed"))
    affected = _affected_top_level(norm)
    # Compose a message that reveals every affected top-level field.
    if len(affected) > 1:
        message = f"validation failed on {affected}: first issue at {loc_str}: {msg}"
    else:
        message = f"{loc_str}: {msg}"
    context: Dict[str, Any] = {
        "loc": list(loc),
        "errors": norm,
    }
    # Typo-candidates: walk EVERY error (not just primary) for
    # extra_forbidden so the common "missing X + typo'd X" pattern still
    # surfaces a candidate even when pydantic reports the missing-field
    # error first. Output keyed by the bad key so the LLM can match
    # candidates back to its specific typo.
    if schema_keys_by_loc:
        suggestions: Dict[str, List[str]] = {}
        allowed_by_parent: Dict[str, List[str]] = {}
        for e in raw:
            if e.get("type") != "extra_forbidden":
                continue
            e_loc = tuple(e.get("loc", ()))
            if not e_loc:
                continue
            parent_loc = tuple(p for p in e_loc[:-1] if isinstance(p, str))
            bad_key = e_loc[-1] if isinstance(e_loc[-1], str) else None
            allowed = schema_keys_by_loc.get(parent_loc) or []
            if not bad_key or not allowed:
                continue
            cands = _typo_candidates(bad_key, allowed)
            if cands:
                suggestions[bad_key] = cands
            parent_key = ".".join(parent_loc) or "<root>"
            allowed_by_parent.setdefault(parent_key, sorted(allowed))
        if suggestions:
            context["did_you_mean"] = suggestions
        if allowed_by_parent:
            context["allowed_fields"] = allowed_by_parent
    if gate is not None:
        context["gate"] = gate
    return make_envelope(
        code,
        message,
        remediation=_remediation_for(code, loc_str, msg),
        affected_fields=affected,
        **context,
    )


__all__ = [
    "EXAMPLES",
    "LOC_TO_CODE",
    "PYDANTIC_TYPE_TO_CODE",
    "from_validation_error",
    "make_envelope",
]
