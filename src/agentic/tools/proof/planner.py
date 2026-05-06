"""NL → flat required obligations.

Trusted-tool boundary. The planner uses the chat model to translate
the question into typed obligations, then **drops** any obligation
that fails ``closure.contract.validate_obligation``. There is no
``required=false`` fallback: if an obligation cannot be made
contract-valid (e.g. predicate not in the executable whitelist,
field missing for lookup, scope unresolvable), the planner emits a
``discovery_diagnostic`` instead — the agent uses search/toc/read to
resolve missing pieces, then proposes a contract-valid obligation
through ``proof_gap_propose``.

Predicate-tightness limitation: the kernel cannot tell whether the
planner's regex captures the user's nuanced intent. Reproducibility
comes from the cache (same question + corpus_hint → same obligations)
and from the canonical_id recorded in every obligation.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

from agentic.closure.contract import (
    KIND_CONTRACTS,
    contract_for,
    render_compute_operations,
    render_contract_summary,
    validate_obligation,
)
from agentic.closure.obligation import (
    Obligation,
    PredicateRef,
    ScopeRef,
)


logger = logging.getLogger(__name__)


@dataclass
class DiscoveryDiagnostic:
    code: str       # missing_scope | unsupported_predicate | missing_field | …
    detail: str
    hint: str       # short next-tool suggestion


@dataclass
class PlannerResult:
    obligations: list[Obligation]
    diagnostics: list[DiscoveryDiagnostic]


_PLANNER_PROMPT = (
    "Translate the user's question into a JSON array of typed obligations.\n"
    "Output ONLY the JSON array; no prose.\n\n"
    "Allowed kinds and their predicate/args (single source of truth):\n"
    "{contract_summary}\n\n"
    "Whitelisted compute operations (used by DerivedValueClaim for compound\n"
    "lookups — e.g. Q3-style 90,000 × 27% = 24,300):\n"
    "{compute_operations}\n\n"
    "Each obligation object MUST include:\n"
    '  kind         one of the kinds above\n'
    '  scope        {{"file_ids": [...], "section_ids": [...] | null}}\n'
    '  unit_type    one of: page | passage | table_row\n'
    '  predicate    {{"name": <whitelisted name>, "args": <per-name schema above>}}\n'
    '  field        REQUIRED for lookup; semantic role name like "min_notional_amount"\n'
    '                 or "existing_policy_total_interest"; NEVER "value" / "amount" / "relevance"\n'
    '  score_field  REQUIRED for argmax; the score field name\n\n'
    "Hard rules:\n"
    "* Predicate args must satisfy the per-name schema:\n"
    "    contains_string(pattern=<LITERAL substring; no regex chars>, case_sensitive?=bool)\n"
    "    regex_match(pattern=<COMPILABLE regex, non-trivial>, flags?=<imsux subset>)\n"
    "    argmax_domain() — no args\n"
    "* For list-all questions over feature-level items (rebate tiers, feature\n"
    "  list entries, table rows), prefer unit_type=passage or table_row over page.\n"
    "* For compound numeric answers (sum/product/percent_of/...), emit MULTIPLE\n"
    "  lookup obligations — one per source value with a distinct semantic field —\n"
    "  AND one final lookup obligation for the derived combined value with its\n"
    "  own semantic field (e.g. existing_policy_total_interest +\n"
    "  segregated_policy_total_interest + combined_total_interest). The agent\n"
    "  will close the derived obligation via DerivedValueClaim arithmetic.\n"
    "* If the question references a specific section by name and you have file_ids\n"
    "  but no section_ids: still emit a file-scope obligation. The agent's\n"
    "  proof_scan / read flow will resolve from the file domain.\n"
    "* Always produce at least one contract-valid obligation when the question\n"
    "  is answerable. Returning an empty array means you genuinely cannot map\n"
    "  the question to any kind — usually you can.\n\n"
    "Question: {question}\n"
    "Files available (use as scope.file_ids): {corpus_hint}\n\n"
    "Output a JSON array."
)


_LOCK = threading.Lock()
_CACHE: dict[tuple[str, tuple], list[dict]] = {}


def propose_initial_obligations(
    question: str,
    *,
    corpus_hint: Optional[list[str]] = None,
) -> PlannerResult:
    fallback_files = tuple(corpus_hint or ())
    raw_specs = _call_llm(question, fallback_files)
    if not raw_specs:
        return PlannerResult(
            obligations=[],
            diagnostics=[DiscoveryDiagnostic(
                code="planner_no_specs",
                detail="LLM returned no parseable obligations.",
                hint="Use list_files / toc / semantic_search to explore the corpus, "
                     "then call proof_gap_propose with a contract-valid obligation.",
            )],
        )

    counter = [0]
    seen: set[tuple] = set()
    materialised: list[Obligation] = []
    diagnostics: list[DiscoveryDiagnostic] = []

    for spec in raw_specs:
        if not isinstance(spec, dict):
            diagnostics.append(DiscoveryDiagnostic(
                code="malformed_spec",
                detail=f"spec is not an object: {spec!r}",
                hint="Ignore.",
            ))
            continue
        out = _materialise(spec, fallback_files=fallback_files, counter=counter)
        if isinstance(out, DiscoveryDiagnostic):
            diagnostics.append(out)
            continue
        key = out.structural_key()
        if key in seen:
            continue
        seen.add(key)
        materialised.append(out)
    return PlannerResult(obligations=materialised, diagnostics=diagnostics)


def _call_llm(question: str, corpus_hint: tuple[str, ...]) -> list[dict]:
    cache_key = (question.strip(), corpus_hint)
    with _LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    from model_client import LLMClient

    try:
        client = LLMClient()
    except ValueError as exc:
        logger.warning("planner: chat model not configured: %s", exc)
        return []

    prompt = _PLANNER_PROMPT.format(
        contract_summary=render_contract_summary(),
        compute_operations=render_compute_operations(),
        question=question.strip(),
        corpus_hint=list(corpus_hint),
    )
    try:
        response = client.chat(
            messages=[
                {"role": "system", "content": "Reply with a JSON array only."},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("planner LLM call failed: %s", exc)
        return []

    content = (response.get("message") or {}).get("content") or ""
    parsed = _parse_json_array(content)
    with _LOCK:
        _CACHE[cache_key] = parsed
    return parsed


def _parse_json_array(content: str) -> list[dict]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def _materialise(
    spec: dict,
    *,
    fallback_files: tuple[str, ...],
    counter: list[int],
) -> Obligation | DiscoveryDiagnostic:
    kind = str(spec.get("kind", "")).strip()
    if kind not in KIND_CONTRACTS:
        return DiscoveryDiagnostic(code="unknown_kind", detail=f"kind={kind!r}", hint="Drop spec.")
    spec_contract = contract_for(kind)  # type: ignore[arg-type]

    unit_type = str(spec.get("unit_type", "")).strip()
    if unit_type not in {"page", "passage", "table_row"}:
        # Closure rules + read tool only support these atom granularities.
        return DiscoveryDiagnostic(
            code="unsupported_unit_type",
            detail=f"unit_type={unit_type!r}",
            hint="Use page / passage / table_row.",
        )

    scope_raw = spec.get("scope") or {}
    file_ids = tuple(scope_raw.get("file_ids") or fallback_files)
    section_ids = scope_raw.get("section_ids") or None
    if not file_ids:
        return DiscoveryDiagnostic(
            code="missing_scope",
            detail="no file_ids resolved from question or corpus_hint",
            hint="Call list_files first, then proof_gap_propose.",
        )
    scope = ScopeRef.build(
        file_ids=file_ids,
        section_ids=tuple(section_ids) if section_ids else None,
    )

    predicate_raw = spec.get("predicate") or {}
    predicate_name = str(predicate_raw.get("name", "")).strip()
    if kind == "argmax":
        predicate = PredicateRef.build("argmax_domain", {})
    else:
        if predicate_name not in spec_contract.allowed_predicate_names:
            return DiscoveryDiagnostic(
                code="unsupported_predicate",
                detail=f"predicate.name={predicate_name!r} for kind={kind!r}",
                hint=f"Use one of {sorted(spec_contract.allowed_predicate_names)}; refine via "
                     "semantic_search to find the right wording, then proof_gap_propose.",
            )
        try:
            predicate = PredicateRef.build(predicate_name, predicate_raw.get("args") or {})
        except ValueError as exc:
            return DiscoveryDiagnostic(code="invalid_predicate", detail=str(exc), hint="Drop.")

    field = spec.get("field")
    if field is not None:
        field = str(field).strip() or None
    score_field = spec.get("score_field")
    if score_field is not None:
        score_field = str(score_field).strip() or None

    counter[0] += 1
    try:
        obligation = Obligation(
            id=f"o_{counter[0]:03d}",
            kind=kind,  # type: ignore[arg-type]
            scope=scope,
            unit_type=unit_type,  # type: ignore[arg-type]
            predicate=predicate,
            required=True,
            field=field,
            score_field=score_field,
        )
    except ValueError as exc:
        return DiscoveryDiagnostic(code="invalid_obligation", detail=str(exc), hint="Drop.")

    contract_err = validate_obligation(obligation)
    if contract_err is not None:
        return DiscoveryDiagnostic(
            code=contract_err,
            detail=f"obligation rejected by contract: {contract_err}",
            hint="Refine via discovery + proof_gap_propose.",
        )
    return obligation
