"""Plant — code-only proof-state controller.

The plant is the single mediator between LLM-facing tools and the proof
state. Its public surface:

* ``add_observation`` — register a tool's raw output and run any
  matching auto-extractor. The LLM's tool result will include the
  emerging gate.diagnose snapshot when state changed.
* ``ingest_claim``    — validate an LLM-proposed claim candidate; on
  accept, run reconcile.
* ``handle_*``        — entry points for the five proof tools
  (create / decompose / challenge / finalize). Each runs full
  validation, executes the side effect, and runs reconcile.
* ``reconcile``       — fixed-point loop that drives auto-bind, auto-
  close, and challenge discharge until no state changes.
* ``gate_view``       — read-only state snapshot used as a tool_result
  appendix for state-changing tool calls.

Soundness invariants live in this file. Every state mutation must go
through one of the ``handle_*`` entry points or through ``reconcile``;
LLM-emitted tool calls must not bypass them.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentic.proof import finalize as _finalize
from agentic.proof import gate_view as _gate_view
from agentic.proof import reconcile as _reconcile
from agentic.proof.state import transitions
from agentic.proof.state.challenge_store import ChallengeStore
from agentic.proof.state.domain_map import DomainMapStore
from agentic.proof.state import domain_map_handler as _dm
from agentic.proof.state.obligation_store import ObligationStore
from agentic.proof.evidence.store import EvidenceStore
from agentic.proof.evidence.extractors import auto_extract
from agentic.proof.types import (
    Citation,
    Claim,
    GateView,
    ObligationSpec,
    ObligationStatus,
    Observation,
    ObservationType,
    RepairKind,
    ScopeRef,
)
from storage.inventory_store import InventoryStore


logger = logging.getLogger(__name__)


@dataclass
class PlantResult:
    """Standard return shape for every plant entry point."""

    ok: bool
    payload: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    gate: Optional[GateView] = None


# --------------------------------------------------------------- helper


from agentic.proof.errors import make_envelope as _err  # canonical envelope builder


def _scope_to_dict(scope: ScopeRef) -> Dict[str, Any]:
    return scope.to_dict()


from agentic.proof.spec_builder import (
    build_claim_candidate as _spec_build_claim_candidate,
    build_obligation_spec as _spec_build_obligation_spec,
    check_scope_resolves as _check_scope_resolves,
)
from agentic.proof.handlers import (
    execute_decompose as _h_execute_decompose,
    validate_challenge_path as _h_validate_challenge_path,
    validate_create_path as _h_validate_create_path,
    validate_decompose_path as _h_validate_decompose_path,
    validate_decomposition_rule as _h_validate_decomposition_rule,
    validate_replacement as _h_validate_replacement,
)


# Phase 2 module-level helpers are now thin shims over the
# observation normaliser. They preserve the previous signatures so
# call sites continue to work unchanged; the body of each delegates
# to ``agentic.proof.observation.normalise`` which is the single
# source of truth for payload shape.
from agentic.proof.evidence.observation import (
    citation_in as _ob_citation_in,
    fetch_span_text as _ob_fetch_span_text,
    normalise as _ob_normalise,
    snapshots_for as _ob_snapshots_for,
)


def _entry_gid(entry: Dict[str, Any]) -> Optional[str]:
    """Backwards-compatible global-id picker for raw payload entries.
    Prefer reading ``NormalisedObservation.entries`` instead."""
    if not isinstance(entry, dict):
        return None
    gid = entry.get("page_global_id") or entry.get("global_id")
    if isinstance(gid, str) and gid:
        return gid
    fid = entry.get("file_id")
    pid = entry.get("page_id")
    if isinstance(fid, str) and isinstance(pid, str):
        return f"{fid}/{pid}"
    return None


def _citation_in_payload(
    citation: Citation,
    observation: Observation,
    page_store,
) -> bool:
    """Backwards-compatible citation validator. Delegates to the
    observation normaliser so all payload-shape knowledge lives in
    one module."""
    return _ob_citation_in(citation, _ob_normalise(observation), page_store)


def _observation_snapshots(citation: Citation, observation: Observation) -> List[str]:
    """Backwards-compatible snapshot collector. Delegates to
    :func:`agentic.proof.observation.snapshots_for`."""
    return _ob_snapshots_for(citation, _ob_normalise(observation))


# --------------------------------------------------------------- plant


class Plant:
    """Single point of authority for proof-state mutations."""

    def __init__(
        self,
        inventory: InventoryStore,
        *,
        obligations: Optional[ObligationStore] = None,
        evidence: Optional[EvidenceStore] = None,
        challenges: Optional[ChallengeStore] = None,
        domain_maps: Optional[DomainMapStore] = None,
    ):
        self.inventory = inventory
        self.obligations = obligations or ObligationStore()
        self.evidence = evidence or EvidenceStore()
        self.challenges = challenges or ChallengeStore()
        self.domain_maps = domain_maps or DomainMapStore()
        # observation_id → NormalisedObservation. Populated by
        # record_observation; read by claim validation, citation checks
        # and ScanClaim coverage analysis. Cleared via snapshot/restore
        # is unnecessary because observations are append-only.
        self._normalised_cache: Dict[str, Any] = {}

    # ------------------------------------------------------ observations

    def record_observation(
        self,
        *,
        tool_name: str,
        observation_type: ObservationType,
        payload: Dict[str, Any],
        citations: Optional[List[Citation]] = None,
    ) -> Observation:
        obs = self.evidence.add_observation(
            tool_name=tool_name,
            observation_type=observation_type,
            payload=payload,
            citations=citations or [],
        )
        # Eagerly cache the canonical normalised view so downstream
        # validators don't pay re-normalisation cost on every call.
        self._normalised_cache[obs.id] = _ob_normalise(obs)
        return obs

    def normalised(self, observation: Observation):
        """Return the cached :class:`NormalisedObservation` for
        ``observation``, computing it on demand for observations created
        outside ``record_observation`` (test fixtures, replay)."""
        cached = self._normalised_cache.get(observation.id)
        if cached is not None:
            return cached
        view = _ob_normalise(observation)
        self._normalised_cache[observation.id] = view
        return view

    def auto_extract_and_ingest(self, observation: Observation) -> List[Claim]:
        """Run auto-extractor, ingest emitted claim candidates, and
        attach per-claim bindings to any obligations already in scope.

        Without the per-claim _auto_bind, an obligation that already
        has a (non-closing) binding would be skipped by reconcile's
        step-1 auto_bind, and the freshly-staged ScanClaim would never
        attach to it.
        """
        candidates = auto_extract(observation, self.inventory)
        accepted: List[Claim] = []
        for candidate in candidates:
            stored = self.evidence.add_claim(candidate)
            self._auto_bind(stored)
            accepted.append(stored)
        return accepted

    # ------------------------------------------------------ obligation_create

    def handle_obligation_create(
        self,
        *,
        spec_payload: Dict[str, Any],
        discharges_challenge: Optional[str] = None,
    ) -> PlantResult:
        spec, err = self._build_obligation_spec(spec_payload, discharges_challenge=discharges_challenge)
        if err is None:
            err = _check_scope_resolves(spec.scope, self.inventory)
        replacement_meta = is_root = None
        if err is None:
            replacement_meta, is_root, err = _h_validate_create_path(
                self, spec, discharges_challenge=discharges_challenge,
            )
        if err is not None:
            return PlantResult(ok=False, error=err, gate=self.gate_view())

        spec.derived_by = (
            "challenge_replacement" if discharges_challenge is not None
            else ("root" if is_root else spec.derived_by)
        )
        if discharges_challenge is not None and replacement_meta is not None:
            # Cross-store changes run inside a snapshot/restore frame so
            # a partial failure (DomainMap slot conflict, retire error)
            # cannot leave half-committed state across the three stores.
            # Atomicity stays at plant level — patched in tests.
            snap_ch = self.challenges.snapshot()
            snap_ob = self.obligations.snapshot()
            snap_dm = self.domain_maps.snapshot()
            try:
                obligation = self.obligations.insert(spec, is_root=is_root)
                target_id = replacement_meta["target_obligation_id"]
                target = self.obligations.get(target_id)
                if (
                    target is not None
                    and target.spec.parent_id is not None
                    and target.spec.parent_id in self.domain_maps
                ):
                    # The register helper performs the atomic
                    # materialised_children + canonical_keys move under
                    # the store lock.
                    self._register_with_domain_map_if_applicable(
                        obligation, replacing_id=target_id,
                    )
                self._discharge_challenge_with_replacement(
                    challenge_id=discharges_challenge,
                    old_obligation_id=target_id,
                    new_obligation_id=obligation.id,
                    repair_kind=replacement_meta["repair_kind"],
                )
            except Exception:
                self.challenges.restore(snap_ch)
                self.obligations.restore(snap_ob)
                self.domain_maps.restore(snap_dm)
                raise
        else:
            obligation = self.obligations.insert(spec, is_root=is_root)
            if spec.parent_id is not None and spec.parent_id in self.domain_maps:
                self._register_with_domain_map_if_applicable(obligation)

        self.reconcile()
        return PlantResult(
            ok=True,
            payload={
                "obligation_id": obligation.id,
                "is_root": obligation.is_root,
                "status": obligation.status.value,
            },
            gate=self.gate_view(),
        )

    # ------------------------------------------------------ obligation_decompose

    def handle_obligation_decompose(
        self,
        *,
        parent_id: str,
        rule_id: str,
        child_specs: List[Dict[str, Any]],
        discharges_challenge: Optional[str] = None,
    ) -> PlantResult:
        parent, prepay_discharge, err = _h_validate_decompose_path(
            self,
            parent_id=parent_id,
            rule_id=rule_id,
            discharges_challenge=discharges_challenge,
        )
        if err is not None:
            return PlantResult(ok=False, error=err, gate=self.gate_view())

        if rule_id == "map_over_domain":
            return self._handle_map_over_domain(
                parent=parent,
                discharges_challenge=discharges_challenge,
                prepay_discharge=prepay_discharge,
            )

        # Build & validate child specs before executing — a malformed
        # spec here aborts cleanly with no state mutation.
        built_children: List[ObligationSpec] = []
        for spec_payload in child_specs:
            child_spec, err = self._build_obligation_spec(spec_payload, parent_id_override=parent_id)
            if err is not None:
                return PlantResult(ok=False, error=err, gate=self.gate_view())
            built_children.append(child_spec)

        return _h_execute_decompose(
            self,
            parent_id=parent_id,
            rule_id=rule_id,
            parent=parent,
            built_children=built_children,
            prepay_discharge=prepay_discharge,
            discharges_challenge=discharges_challenge,
        )

    # ------------------------------------------------------ obligation_challenge

    def handle_obligation_challenge(
        self,
        *,
        obligation_id: str,
        repair_kind: RepairKind,
        evidence_ids: List[str],
        reason: str,
    ) -> PlantResult:
        _, wrong_kind_eligible, err = _h_validate_challenge_path(
            self,
            obligation_id=obligation_id,
            repair_kind=repair_kind,
            evidence_ids=evidence_ids,
        )
        if err is not None:
            return PlantResult(ok=False, error=err, gate=self.gate_view())

        # Cap-bearing repair_kinds (wrong_question_kind) consume their
        # cap LAST — only after every validation passes and the
        # challenge is about to be inserted. Otherwise a malformed
        # retry burns the single repair attempt.
        if wrong_kind_eligible and not self.obligations.consume_wrong_kind_attempt():
            return PlantResult(
                ok=False,
                error=_err(
                    "wrong_kind_cap_exhausted",
                    "wrong_question_kind already used once this session",
                    remediation="The session-wide cap of one wrong_question_kind repair is exhausted. Proceed with the current root kind or finalize/abstain.",
                ),
                gate=self.gate_view(),
            )

        challenge = self.challenges.insert(
            obligation_id=obligation_id,
            repair_kind=repair_kind,
            evidence_ids=evidence_ids,
            reason=reason,
        )
        self.obligations.record_challenge_open(obligation_id, challenge.id)
        self.reconcile()
        return PlantResult(
            ok=True,
            payload={"challenge_id": challenge.id, "obligation_status": "CHALLENGED"},
            gate=self.gate_view(),
        )

    # ------------------------------------------------------ evidence_ingest

    def handle_evidence_ingest(
        self,
        *,
        observation_id: str,
        claim_candidate: Dict[str, Any],
    ) -> PlantResult:
        # Claim ingest is disabled until ≥1 required obligation exists
        # — the universe O_active_required must be non-empty for
        # certification semantics to be defined.
        if not self.obligations.has_active_required():
            return PlantResult(ok=False, gate=self.gate_view(), error=_err(
                "no_root_obligation",
                "evidence_ingest is disabled until at least one required obligation exists",
                remediation="Call obligation_create first to register a root obligation that frames the question, then re-issue evidence_ingest.",
            ))
        observation = self.evidence.get_observation(observation_id)
        if observation is None:
            return PlantResult(ok=False, gate=self.gate_view(), error=_err(
                "unknown_observation", f"observation_id={observation_id!r} unknown",
                remediation="Pass an observation_id returned by a recent acquisition tool (read_page / pattern_search / semantic_search / bm25_search etc.); the observation must already exist in the plant.",
                observation_id=observation_id,
            ))
        claim, err = self._build_claim_candidate(observation, claim_candidate)
        if err is not None:
            return PlantResult(ok=False, error=err, gate=self.gate_view())
        stored = self.evidence.add_claim(claim)
        bindings = self._auto_bind(stored)
        closures = self.reconcile()
        return PlantResult(
            ok=True,
            payload={
                "claim_id": stored.id,
                "auto_bindings": [b.obligation_id for b in bindings],
                "closures_triggered": closures,
            },
            gate=self.gate_view(),
        )

    # ------------------------------------------------------ answer_finalize

    def handle_answer_finalize(
        self,
        *,
        draft_text: str,
        cited_claim_ids: List[str],
        budget_exhausted: bool = False,
    ) -> PlantResult:
        return _finalize.run_answer_finalize(
            self,
            draft_text=draft_text,
            cited_claim_ids=cited_claim_ids,
            budget_exhausted=budget_exhausted,
        )

    # ------------------------------------------------------ reconcile

    def reconcile(self) -> List[Dict[str, Any]]:
        return _reconcile.reconcile(self)

    # ------------------------------------------------------ gate.diagnose

    def gate_view(self) -> GateView:
        return _gate_view.build_gate_view(self)

    # ------------------------------------------------------ helpers

    # Forwarders into the extracted helper modules. Each method's
    # signature is preserved so external code (tools/, agent/, tests)
    # that call ``plant._foo(...)`` resolve unchanged.

    def _build_obligation_spec(self, payload, *, parent_id_override=None, discharges_challenge=None):
        return _spec_build_obligation_spec(payload, parent_id_override=parent_id_override, discharges_challenge=discharges_challenge)

    def _build_claim_candidate(self, observation, payload):
        return _spec_build_claim_candidate(self, observation, payload)

    def _citation_for_unit(self, unit_id, unit_type, index):
        """Find the citation that points at ``unit_id``. File-level
        accepts any citation in that file; section-level walks the
        section's pages."""
        if unit_type == "file":
            for (fid, _), citation in index.items():
                if fid == unit_id:
                    return citation
            return None
        sec = self.inventory.get(unit_id)
        if sec is None:
            return None
        for (fid, pid), citation in index.items():
            if fid != sec.file_id:
                continue
            page = self.inventory.page_store.get(f"{fid}/{pid}")
            if page is None or page.page_number is None:
                continue
            if sec.page_start <= page.page_number <= sec.page_end:
                return citation
        return None

    def _fetch_span_text(self, citation, observation):
        return _ob_fetch_span_text(citation, _ob_normalise(observation))

    def _validate_decomposition_rule(self, rule_id, parent, children):
        return _h_validate_decomposition_rule(self, rule_id, parent, children)

    def _handle_map_over_domain(self, *, parent, discharges_challenge, prepay_discharge=None):
        return _dm.handle_map_over_domain(self, parent=parent, discharges_challenge=discharges_challenge, prepay_discharge=prepay_discharge)

    def _validate_domain_map_child(self, parent, spec, *, replacing_obligation_id=None):
        return _dm.validate_domain_map_child(self, parent, spec, replacing_obligation_id=replacing_obligation_id)

    def _register_with_domain_map_if_applicable(self, obligation, *, replacing_id=None):
        _dm.register_with_domain_map_if_applicable(self, obligation, replacing_id=replacing_id)

    def _synthesise_scope_partition_claim(self, parent):
        return _dm.synthesise_scope_partition_claim(self, parent)

    def _propagate_close_to_domain_map(self, obligation, result):
        _dm.propagate_close_to_domain_map(self, obligation, result)

    def _validate_replacement(self, spec, challenge_id):
        return _h_validate_replacement(self, spec, challenge_id)

    def _discharge_challenge_with_replacement(
        self, *, challenge_id, old_obligation_id, new_obligation_id, repair_kind,
    ):
        # The single cross-store frame: discharge → CHALLENGED→OPEN →
        # retire → root transfer. Stays on Plant because the
        # green_cross_store_replacement_atomicity test patches this
        # method to inject a failure mid-frame and verify rollback.
        self.challenges.discharge(
            challenge_id,
            meta={"resolved_via": "replacement", "new_obligation_id": new_obligation_id},
        )
        old = self.obligations.get(old_obligation_id)
        if old is None:
            return
        self.obligations.record_challenge_discharged(old_obligation_id, challenge_id)
        self.obligations.record_retire(
            old_obligation_id,
            reason="challenge_resolved_with_replacement",
            replacement_id=new_obligation_id,
        )
        # Whenever the retired obligation was the root, transfer root
        # identity to the replacement. Repair_kind doesn't matter —
        # ObligationStore only ever holds one active root.
        if old.is_root:
            self.obligations.replace_root(new_obligation_id)

    def _is_pre_proof_window(self):
        for o in self.obligations.all_obligations():
            if not o.spec.required:
                continue
            if o.status not in (ObligationStatus.OPEN, ObligationStatus.CHALLENGED):
                return False
        return True

    def _has_bindings(self, obligation_id):
        return _reconcile.has_bindings(self, obligation_id)

    def _auto_bind(self, claim):
        return _reconcile.auto_bind(self, claim)

    def _auto_bind_obligation(self, obligation):
        return _reconcile.auto_bind_obligation(self, obligation)

    def _claim_matches_obligation(self, claim, obligation):
        return _reconcile.claim_matches_obligation(claim, obligation)

    def _ancestor_challenged(self, obligation_id):
        return transitions.ancestor_challenged(self.obligations, obligation_id)

    def _challenged_in_closure_cone(self):
        return _gate_view.challenged_in_closure_cone(self)

    def _all_children_closed(self, parent):
        return _reconcile.all_children_closed(self, parent)

    def _postcondition_met(self, challenge):
        return _reconcile.postcondition_met(self, challenge)

    def _diagnose_open(self, obligation):
        return _gate_view.diagnose_open(self, obligation)

    def _cursor_for(self, obligation):
        return _gate_view.cursor_for(self, obligation)

    def _compose_final_answer(self, draft_text, cited_claim_ids):
        return _finalize.compose_final_answer(self, draft_text, cited_claim_ids)
