"""End-to-end contract tests with synthetic observations.

Each test exercises one obligation kind by:
  1. handing the session an obligation produced by the planner shape,
  2. injecting a fake *ReadObservation / PatternScanObservation,
  3. calling the proof_claim_ingest tool with obligation-centric args,
  4. asserting the gate decision via proof_finalize.

Stub Inventory + raw observation JSON keeps each case <1s and free of
LLM / corpus dependencies.
"""
import json

import pytest

from agentic.closure import (
    Budget,
    CandidateGap,
    EvidenceHint,
    Observation,
    Obligation,
    PredicateRef,
    ProofSession,
    ScopeRef,
)
from agentic.closure.contract import validate_obligation
from agentic.core.context import AgentContext
from agentic.tools.proof.proof_claim_ingest import ProofClaimIngestTool
from agentic.tools.proof.proof_finalize import ProofFinalizeTool
from agentic.tools.proof.proof_gap_propose import ProofGapProposeTool


# ---------------------------------------------------------------- fixtures


class _StubInventory:
    """Inventory that returns a fixed unit set per (scope, unit_type) key."""

    def __init__(self, mapping: dict[tuple[str, str], frozenset[str]]):
        self._mapping = mapping

    def units(self, scope, unit_type):
        return self._mapping.get((scope.canonical_scope_id, unit_type), frozenset())


def _session(inv) -> ProofSession:
    return ProofSession.build(inventory=inv, budget=Budget(remaining_steps=8))


def _scope(file_ids=("fA",), section_ids=None) -> ScopeRef:
    return ScopeRef.build(file_ids=file_ids, section_ids=section_ids)


def _pred(name="regex_match", **args) -> PredicateRef:
    return PredicateRef.build(name=name, args=args)


def _push_observation(session, obs_id: str, payload: dict, tool_name: str):
    session.append_observation(
        Observation(id=obs_id, tool_name=tool_name, text=json.dumps(payload, ensure_ascii=False)),
    )


def _ctx() -> AgentContext:
    return AgentContext()


def _ingest(session, **kwargs):
    return ProofClaimIngestTool(session).execute(_ctx(), **kwargs)


def _finalize(session, **kwargs):
    return ProofFinalizeTool(session).execute(_ctx(), **kwargs)


# ---------------------------------------------------------------- exists


def test_exists_witness_from_read_certifies():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="exists", scope=scope, unit_type="page",
                   predicate=_pred(pattern="rebate"))
    session.obligations.append(o)
    _push_observation(session, "obs_1", {
        "observation_type": "PageReadObservation",
        "unit_type": "page",
        "units": [
            {"unit_id": "fA/p_0001", "file_id": "fA", "page_id": "p_0001",
             "page_number": 1, "text": "Rebate program details follow."}
        ],
    }, tool_name="read")

    _, log = _ingest(
        session,
        obligation_id="o1", observation_id="obs_1",
        unit_id="fA/p_0001", polarity="positive",
        span="Rebate program",
    )
    assert log["error"] is None
    _, fl = _finalize(session)
    assert fl["decision"] == "CERTIFIED"


# ---------------------------------------------------------------- lookup


def test_lookup_value_claim_certifies():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0015"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="lookup", scope=scope, unit_type="page",
                   predicate=_pred(pattern="USD"), field="min_notional")
    session.obligations.append(o)
    _push_observation(session, "obs_1", {
        "observation_type": "PageReadObservation", "unit_type": "page",
        "units": [{"unit_id": "fA/p_0015", "file_id": "fA", "page_id": "p_0015",
                   "page_number": 15, "text": "Minimum notional USD 10,000 / HKD 80,000"}],
    }, tool_name="read")

    _, log = _ingest(
        session,
        obligation_id="o1", observation_id="obs_1",
        unit_id="fA/p_0015",
        value="10000", value_type="numeric", span="USD 10,000",
    )
    assert log["error"] is None
    _, fl = _finalize(session, draft_text="Minimum notional is USD 10,000.")
    assert fl["decision"] == "CERTIFIED"


def test_lookup_disagreeing_values_ambiguous():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001", "fA/p_0002"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="lookup", scope=scope, unit_type="page",
                   predicate=_pred(pattern="USD"), field="min_notional")
    session.obligations.append(o)
    for pid, value, span in [
        ("fA/p_0001", "10000", "USD 10,000"),
        ("fA/p_0002", "12000", "USD 12,000"),
    ]:
        _push_observation(session, f"obs_{pid}", {
            "observation_type": "PageReadObservation", "unit_type": "page",
            "units": [{"unit_id": pid, "file_id": "fA", "page_id": pid.split("/")[1],
                       "page_number": int(pid.split("_")[-1]), "text": span}],
        }, tool_name="read")
        _ingest(session, obligation_id="o1", observation_id=f"obs_{pid}",
                unit_id=pid, value=value, value_type="numeric", span=span)

    _, fl = _finalize(session)
    assert fl["decision"] == "ABSTAIN"
    payload = json.loads(_finalize(session)[0])
    assert payload["reason"] == "ambiguous_lookup"


# ---------------------------------------------------------------- count / set


def _scan_observation(scope: ScopeRef, unit_type: str, *, pattern: str,
                      scanned: list[str], positive: list[str]) -> dict:
    return {
        "observation_type": "PatternScanObservation",
        "pattern": pattern,
        "scope": {"file_ids": list(scope.file_ids),
                  "section_ids": list(scope.section_ids) if scope.section_ids else None},
        "unit_type": unit_type,
        "scanned_units": scanned,
        "positive_units": positive,
        "negative_units": [u for u in scanned if u not in set(positive)],
    }


def test_count_complete_scan_certifies():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"):
                          frozenset({"fA/p_0001", "fA/p_0002", "fA/p_0003"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="count", scope=scope, unit_type="page",
                   predicate=_pred(pattern="X"))
    session.obligations.append(o)
    _push_observation(session, "obs_1",
                      _scan_observation(scope, "page", pattern="X",
                                        scanned=["fA/p_0001", "fA/p_0002", "fA/p_0003"],
                                        positive=["fA/p_0001", "fA/p_0003"]),
                      tool_name="pattern_search")
    _, log = _ingest(session, obligation_id="o1", observation_id="obs_1")
    assert log["error"] is None
    _, fl = _finalize(session, draft_text="There are 2 such pages.")
    assert fl["decision"] == "CERTIFIED"


def test_set_complete_scan_certifies():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "passage"):
                          frozenset({"fA/p_0001:p_0001", "fA/p_0001:p_0002"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="set", scope=scope, unit_type="passage",
                   predicate=_pred(pattern="first-in-market"))
    session.obligations.append(o)
    _push_observation(session, "obs_1",
                      _scan_observation(scope, "passage", pattern="first-in-market",
                                        scanned=["fA/p_0001:p_0001", "fA/p_0001:p_0002"],
                                        positive=["fA/p_0001:p_0001"]),
                      tool_name="pattern_search")
    _, log = _ingest(session, obligation_id="o1", observation_id="obs_1")
    assert log["error"] is None
    _, fl = _finalize(session)
    assert fl["decision"] == "CERTIFIED"


# ---------------------------------------------------------------- forall / negation


def test_forall_counterexample_witness_closes_false():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="forall", scope=scope, unit_type="page",
                   predicate=_pred(pattern="MUST"))
    session.obligations.append(o)
    # pattern_search reports the unit as negative — agent ingests as counterexample.
    _push_observation(session, "obs_1",
                      _scan_observation(scope, "page", pattern="MUST",
                                        scanned=["fA/p_0001"],
                                        positive=[]),
                      tool_name="pattern_search")
    _, log = _ingest(session, obligation_id="o1", observation_id="obs_1",
                     unit_id="fA/p_0001", polarity="negative")
    assert log["error"] is None
    payload = json.loads(_finalize(session)[0])
    assert payload["decision"] == "CERTIFIED"
    assert payload["closed_obligations"][0]["closed_value"] is False


def test_negation_complete_scan_zero_positive_closes_true():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001", "fA/p_0002"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="negation", scope=scope, unit_type="page",
                   predicate=_pred(pattern="forbidden"))
    session.obligations.append(o)
    _push_observation(session, "obs_1",
                      _scan_observation(scope, "page", pattern="forbidden",
                                        scanned=["fA/p_0001", "fA/p_0002"],
                                        positive=[]),
                      tool_name="pattern_search")
    _, log = _ingest(session, obligation_id="o1", observation_id="obs_1")
    assert log["error"] is None
    payload = json.loads(_finalize(session)[0])
    assert payload["decision"] == "CERTIFIED"
    assert payload["closed_obligations"][0]["closed_value"] is True


# ---------------------------------------------------------------- argmax


def test_argmax_per_unit_values_certify():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001", "fA/p_0002"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="argmax", scope=scope, unit_type="page",
                   predicate=PredicateRef.build("argmax_domain", {}),
                   score_field="price")
    session.obligations.append(o)
    for pid, val in [("fA/p_0001", 100), ("fA/p_0002", 250)]:
        _push_observation(session, f"obs_{pid}", {
            "observation_type": "PageReadObservation", "unit_type": "page",
            "units": [{"unit_id": pid, "file_id": "fA", "page_id": pid.split("/")[1],
                       "page_number": int(pid.split("_")[-1]), "text": f"Price USD {val}"}],
        }, tool_name="read")
        _ingest(session, obligation_id="o1", observation_id=f"obs_{pid}",
                unit_id=pid, value=val, value_type="numeric", span=f"USD {val}")
    payload = json.loads(_finalize(session, draft_text="Winner has 250.")[0])
    assert payload["decision"] == "CERTIFIED"
    assert payload["closed_obligations"][0]["closed_value"]["unit_id"] == "fA/p_0002"


def test_argmax_missing_value_continues_then_abstains_when_budget_runs_out():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001", "fA/p_0002"})})
    session = ProofSession.build(inventory=inv, budget=Budget(remaining_steps=0))
    o = Obligation(id="o1", kind="argmax", scope=scope, unit_type="page",
                   predicate=PredicateRef.build("argmax_domain", {}),
                   score_field="price")
    session.obligations.append(o)
    _push_observation(session, "obs_1", {
        "observation_type": "PageReadObservation", "unit_type": "page",
        "units": [{"unit_id": "fA/p_0001", "file_id": "fA", "page_id": "p_0001",
                   "page_number": 1, "text": "Price USD 100"}],
    }, tool_name="read")
    _ingest(session, obligation_id="o1", observation_id="obs_1",
            unit_id="fA/p_0001", value=100, value_type="numeric", span="USD 100")
    payload = json.loads(_finalize(session)[0])
    assert payload["decision"] == "ABSTAIN"


# ---------------------------------------------------------------- contract guards


def test_no_required_obligations_abstains():
    inv = _StubInventory({})
    session = _session(inv)
    payload = json.loads(_finalize(session)[0])
    assert payload["decision"] == "ABSTAIN"
    assert payload["reason"] == "no_required_obligations"


def test_unit_type_mismatch_blocks_value_ingest():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "table_row"): frozenset({"fA/p_0001:t_00:r_00"})})
    session = _session(inv)
    o = Obligation(id="o1", kind="lookup", scope=scope, unit_type="table_row",
                   predicate=_pred(pattern="USD"), field="amount")
    session.obligations.append(o)
    # Page-level read observation — wrong granularity.
    _push_observation(session, "obs_1", {
        "observation_type": "PageReadObservation", "unit_type": "page",
        "units": [{"unit_id": "fA/p_0001", "file_id": "fA", "page_id": "p_0001",
                   "page_number": 1, "text": "USD 10,000"}],
    }, tool_name="read")
    _, log = _ingest(session, obligation_id="o1", observation_id="obs_1",
                     unit_id="fA/p_0001", value="10000", value_type="numeric", span="USD 10,000")
    assert log["error"] == "unit_type_mismatch"


def test_promotion_rejects_semantic_predicate():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001"})})
    session = _session(inv)
    bad = Obligation(
        id="o_gap_001", kind="lookup", scope=scope, unit_type="page",
        predicate=PredicateRef.build("is_designated_plan", {}),
        field="something",
    )
    gap = CandidateGap(
        id="gap_001", kind="missing_scope",
        proposed_obligation=bad,
        evidence_hint=EvidenceHint(rationale="agent thought this was a thing"),
    )
    promoted = session.plant.validate_obligation_update(
        gap, session.obligations, session.budget, promoted_so_far=0,
    )
    assert promoted is None  # contract validator rejects the semantic name


def test_promotion_admits_contract_valid():
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001"})})
    session = _session(inv)
    good = Obligation(
        id="o_gap_001", kind="lookup", scope=scope, unit_type="page",
        predicate=PredicateRef.build("contains_string", {"pattern": "minimum"}),
        field="min_notional",
    )
    assert validate_obligation(good) is None
    gap = CandidateGap(
        id="gap_001", kind="missing_scope",
        proposed_obligation=good,
        evidence_hint=EvidenceHint(rationale="found via toc"),
    )
    promoted = session.plant.validate_obligation_update(
        gap, session.obligations, session.budget, promoted_so_far=0,
    )
    assert promoted is good


# ---------------------------------------------------------------- contract args + naming


def test_contains_string_rejects_regex_chars():
    """Q4 trap: contains_string with regex alternation must be rejected
    (force the LLM to switch to regex_match)."""

    scope = _scope()
    pred = PredicateRef.build("contains_string", {"pattern": "First-in-market|market-first"})
    o = Obligation(id="o", kind="set", scope=scope, unit_type="passage", predicate=pred)
    err = validate_obligation(o)
    assert err is not None
    assert "predicate_args_invalid" in err
    assert "contains regex special" in err


def test_regex_match_rejects_trivial():
    pred = PredicateRef.build("regex_match", {"pattern": ".*"})
    o = Obligation(
        id="o", kind="count", scope=_scope(), unit_type="page", predicate=pred,
    )
    err = validate_obligation(o)
    assert err is not None and "trivial regex" in err


def test_lookup_field_placeholder_rejected():
    pred = PredicateRef.build("contains_string", {"pattern": "USD"})
    o = Obligation(
        id="o", kind="lookup", scope=_scope(), unit_type="page",
        predicate=pred, field="value",     # reserved placeholder
    )
    err = validate_obligation(o)
    assert err is not None and "field_invalid" in err


# ---------------------------------------------------------------- replan


def test_replan_replaces_obligations_and_clears_gaps():
    from agentic.tools.proof.proof_plan_init import ProofPlanInitTool
    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001"})})
    session = _session(inv)
    # Seed an obligation as if the first plan_init returned something.
    o1 = Obligation(id="old", kind="lookup", scope=scope, unit_type="page",
                    predicate=_pred(pattern="A"), field="min_notional")
    session.obligations.append(o1)
    session.candidate_gaps.append(
        CandidateGap(id="gap_pre", kind="missing_scope",
                     proposed_obligation=None,
                     evidence_hint=EvidenceHint(rationale="prior")),
    )
    session.promoted_count = 1

    # Re-plan via the tool. Since we don't want to call the LLM, monkey-
    # patch the import the tool actually uses (`proof_plan_init`'s
    # local `propose_initial_obligations` reference).
    from agentic.tools.proof import proof_plan_init as _ppi
    from agentic.tools.proof.planner import PlannerResult
    new_o = Obligation(id="new", kind="lookup", scope=scope, unit_type="page",
                       predicate=_pred(pattern="B"), field="issue_age_max")
    orig = _ppi.propose_initial_obligations
    _ppi.propose_initial_obligations = lambda question, *, corpus_hint=None: PlannerResult(
        obligations=[new_o], diagnostics=[],
    )
    try:
        result, log = ProofPlanInitTool(session).execute(_ctx(), question="q", corpus_hint=["fA"])
    finally:
        _ppi.propose_initial_obligations = orig

    assert log["replan"] is True
    assert len(session.obligations) == 1 and session.obligations[0].id == "new"
    assert session.candidate_gaps == []
    assert session.promoted_count == 0


# ---------------------------------------------------------------- derived value claim


def test_derived_value_claim_percent_of_certifies():
    """Q3 path: 90,000 × 27% = 24,300, both inputs cited, derivation
    closes the lookup."""

    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001"})})
    session = _session(inv)

    src1 = Obligation(id="src_afyp", kind="lookup", scope=scope, unit_type="page",
                      predicate=_pred(pattern="AFYP"), field="afyp_amount")
    src2 = Obligation(id="src_pct", kind="lookup", scope=scope, unit_type="page",
                      predicate=_pred(pattern="refund"), field="combined_refund_pct")
    derived_o = Obligation(id="derived", kind="lookup", scope=scope, unit_type="page",
                           predicate=_pred(pattern="USD"), field="combined_refund_amount")
    session.obligations.extend([src1, src2, derived_o])

    _push_observation(session, "obs_1", {
        "observation_type": "PageReadObservation", "unit_type": "page",
        "units": [{"unit_id": "fA/p_0001", "file_id": "fA", "page_id": "p_0001",
                   "page_number": 1, "text": "AFYP USD 90,000 with refund 27%"}],
    }, tool_name="read")

    _, log1 = _ingest(session, obligation_id="src_afyp", observation_id="obs_1",
                      unit_id="fA/p_0001", value=90000, value_type="numeric",
                      span="USD 90,000")
    assert log1["error"] is None
    _, log2 = _ingest(session, obligation_id="src_pct", observation_id="obs_1",
                      unit_id="fA/p_0001", value="27%", value_type="percentage",
                      span="27%")
    assert log2["error"] is None

    afyp_id = session.claims[0].id
    pct_id = session.claims[1].id
    _, log3 = _ingest(session, obligation_id="derived",
                      operation="percent_of",
                      input_claim_ids=[afyp_id, pct_id],
                      value=24300, value_type="numeric")
    assert log3["error"] is None
    assert log3["must_finalize_next"] is True

    payload = json.loads(_finalize(session, draft_text="AFYP USD 90,000 × 27% = combined refund USD 24,300.")[0])
    assert payload["decision"] == "CERTIFIED"


def test_derived_value_claim_sum_certifies():
    """Q6 path: 8,659 + 39,351 = 48,010, sum operation."""

    scope = _scope()
    inv = _StubInventory({
        (scope.canonical_scope_id, "page"): frozenset({"fA/p_0001", "fA/p_0002"}),
    })
    session = _session(inv)

    o_existing = Obligation(id="existing", kind="lookup", scope=scope, unit_type="page",
                            predicate=_pred(pattern="existing"),
                            field="existing_policy_total_interest")
    o_segregated = Obligation(id="segregated", kind="lookup", scope=scope, unit_type="page",
                              predicate=_pred(pattern="segregated"),
                              field="segregated_policy_total_interest")
    o_combined = Obligation(id="combined", kind="lookup", scope=scope, unit_type="page",
                            predicate=_pred(pattern="combined"),
                            field="combined_total_interest")
    session.obligations.extend([o_existing, o_segregated, o_combined])

    for pid, val, span_text, tag in [
        ("fA/p_0001", 8659, "USD 8,659", "existing total interest"),
        ("fA/p_0002", 39351, "USD 39,351", "segregated total interest"),
    ]:
        _push_observation(session, f"obs_{pid}", {
            "observation_type": "PageReadObservation", "unit_type": "page",
            "units": [{"unit_id": pid, "file_id": "fA", "page_id": pid.split("/")[1],
                       "page_number": int(pid.split("_")[-1]),
                       "text": f"{tag}: {span_text}"}],
        }, tool_name="read")

    _, log_a = _ingest(session, obligation_id="existing", observation_id="obs_fA/p_0001",
                       unit_id="fA/p_0001", value=8659, value_type="numeric", span="USD 8,659")
    assert log_a["error"] is None
    _, log_b = _ingest(session, obligation_id="segregated", observation_id="obs_fA/p_0002",
                       unit_id="fA/p_0002", value=39351, value_type="numeric", span="USD 39,351")
    assert log_b["error"] is None

    a_id, b_id = session.claims[0].id, session.claims[1].id
    _, log_c = _ingest(session, obligation_id="combined",
                       operation="sum",
                       input_claim_ids=[a_id, b_id],
                       value=48010, value_type="numeric")
    assert log_c["error"] is None
    payload = json.loads(_finalize(session, draft_text="Existing 8,659 + segregated 39,351 = combined 48,010.")[0])
    assert payload["decision"] == "CERTIFIED"


def test_derived_arithmetic_mismatch_rejected():
    """Kernel re-runs the math; LLM cannot smuggle a wrong sum."""

    scope = _scope()
    inv = _StubInventory({(scope.canonical_scope_id, "page"): frozenset({"fA/p_0001"})})
    session = _session(inv)
    src = Obligation(id="src_a", kind="lookup", scope=scope, unit_type="page",
                     predicate=_pred(pattern="A"), field="value_a")
    other = Obligation(id="src_b", kind="lookup", scope=scope, unit_type="page",
                       predicate=_pred(pattern="B"), field="value_b")
    derived = Obligation(id="derived", kind="lookup", scope=scope, unit_type="page",
                         predicate=_pred(pattern="C"), field="combined_value")
    session.obligations.extend([src, other, derived])
    _push_observation(session, "obs_1", {
        "observation_type": "PageReadObservation", "unit_type": "page",
        "units": [{"unit_id": "fA/p_0001", "file_id": "fA", "page_id": "p_0001",
                   "page_number": 1, "text": "A=10 B=20"}],
    }, tool_name="read")
    _ingest(session, obligation_id="src_a", observation_id="obs_1",
            unit_id="fA/p_0001", value=10, value_type="numeric", span="10")
    _ingest(session, obligation_id="src_b", observation_id="obs_1",
            unit_id="fA/p_0001", value=20, value_type="numeric", span="20")
    a, b = session.claims[0].id, session.claims[1].id
    _, log = _ingest(session, obligation_id="derived",
                     operation="sum", input_claim_ids=[a, b],
                     value=99, value_type="numeric")          # LIES — real sum is 30
    assert log["error"] == "arithmetic_mismatch"


# ---------------------------------------------------------------- proof_scan + must_finalize


def test_proof_scan_closes_count_with_canonical_alignment():
    """proof_scan reads the obligation and runs the right matcher;
    canonical_id is guaranteed to match → no scan_coverage_mismatch."""

    from agentic.tools.proof.proof_scan import ProofScanTool

    class _StubPageStore:
        def __init__(self, pages):
            self._p = pages

        def get(self, gid):
            return self._p.get(gid)

    class _Page:
        def __init__(self, gid, text):
            self.global_id = gid
            self.file_id = gid.split("/")[0]
            self.page_id = gid.split("/")[1]
            self.page_number = int(gid.split("_")[-1])
            self.text_markdown = text

    class _StubInventoryStore:
        def __init__(self, units_map):
            self._units = units_map
            self.passage_store = None
            self.table_row_store = None

        def units(self, scope, unit_type):
            return self._units.get((scope.canonical_scope_id, unit_type), frozenset())

    scope = _scope()
    domain = frozenset({"fA/p_0001", "fA/p_0002", "fA/p_0003"})
    page_store = _StubPageStore({
        "fA/p_0001": _Page("fA/p_0001", "alpha contains rebate"),
        "fA/p_0002": _Page("fA/p_0002", "beta no match here"),
        "fA/p_0003": _Page("fA/p_0003", "gamma rebate again"),
    })

    class _StubInventoryProtocol:
        def units(self, scope_, ut):
            return domain if ut == "page" else frozenset()

    inv_proto = _StubInventoryProtocol()
    inv_store = _StubInventoryStore({(scope.canonical_scope_id, "page"): domain})

    session = ProofSession.build(inventory=inv_proto, budget=Budget(remaining_steps=4))
    o = Obligation(id="o1", kind="count", scope=scope, unit_type="page",
                   predicate=PredicateRef.build("contains_string", {"pattern": "rebate"}))
    session.obligations.append(o)

    tool = ProofScanTool(session, page_store, inv_store)
    result, log = tool.execute(_ctx(), obligation_id="o1")
    assert log["error"] is None
    assert log["must_finalize_next"] is True
    payload = json.loads(_finalize(session, draft_text="There are 2 such pages.")[0])
    assert payload["decision"] == "CERTIFIED"
    assert payload["closed_obligations"][0]["closed_value"] == 2
