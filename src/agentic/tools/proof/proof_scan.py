"""``proof_scan`` — obligation-centric scan + ingest in one shot.

The agent passes ``obligation_id``; the tool reads the obligation's
scope / unit_type / predicate from the session, runs the scan with
that EXACT predicate (re.escape for contains_string, raw regex for
regex_match), and emits a PatternScanObservation whose canonical
predicate id equals the obligation's. The Plant then ingests the
ScanClaim immediately and run_closure runs.

Why a separate tool: pattern_search stays general-purpose (discovery
mode); the agent doesn't have to align scope/unit_type/predicate
canonical id by hand. Removes the entire ``predicate_mismatch_with_scan``
+ ``scan_coverage_mismatch`` failure class that the agent kept hitting.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import regex as ureg

from agentic.closure.contract import contract_for
from agentic.closure.obligation import Obligation
from agentic.closure.session import Observation
from agentic.tools.acquisition._common import Scope, all_pages, err, filter_pages, ok
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext
    from agentic.closure.session import ProofSession


logger = logging.getLogger(__name__)


_MAX_CITATIONS = 200


class ProofScanTool(BaseTool):
    def __init__(self, session: "ProofSession", page_store, inventory) -> None:
        self._session = session
        self._page_store = page_store
        self._inventory = inventory

    @property
    def name(self) -> str:
        return "proof_scan"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "proof_scan",
                "description": (
                    "Run the exhaustive scan that closes a "
                    "set/count/forall/negation obligation. Reads the "
                    "obligation's scope / unit_type / predicate, runs "
                    "the matcher (literal substring for contains_string, "
                    "regex for regex_match) over inventory.units(scope, "
                    "unit_type), emits a ScanClaim whose canonical "
                    "predicate id equals the obligation's, and ingests "
                    "it. One call closes the obligation if the partition "
                    "is non-empty for the obligation's polarity needs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "obligation_id": {"type": "string"},
                    },
                    "required": ["obligation_id"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        obligation_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if not obligation_id:
            return _err("invalid_argument", "obligation_id is required.")
        obligation = self._session.find_obligation(str(obligation_id))
        if obligation is None:
            return _err("unknown_obligation", f"no obligation with id={obligation_id!r}")

        spec = contract_for(obligation.kind)
        if "ScanClaim" not in spec.claim_for_evidence:
            return _err(
                "scan_not_applicable",
                f"obligation kind={obligation.kind!r} does not close on a ScanClaim. "
                "Use proof_claim_ingest with read observations instead.",
            )

        scope = obligation.scope
        unit_type = obligation.unit_type
        predicate = obligation.predicate

        try:
            compiled = self._compile_predicate(predicate.name, predicate.args_dict())
        except ValueError as exc:
            return _err("predicate_compile_failed", str(exc))

        # Run the scan over inventory.units(scope, unit_type) — guaranteed
        # to match the obligation's domain.
        domain = self._inventory.units(scope, unit_type)
        scanned, positive, negative, citations = self._scan(domain, unit_type, compiled)

        observation_payload = {
            "observation_type": "PatternScanObservation",
            "pattern": predicate.args_dict().get("pattern", ""),
            "predicate_name": predicate.name,
            "scope": {
                "file_ids": list(scope.file_ids),
                "section_ids": list(scope.section_ids) if scope.section_ids else None,
            },
            "unit_type": unit_type,
            "scanned_units": sorted(scanned),
            "positive_units": sorted(positive),
            "negative_units": sorted(negative),
            "citations": citations,
            "exhaustive": True,
        }
        obs_id = f"obs_proof_scan_{len(self._session.observations) + 1:04d}"
        self._session.append_observation(
            Observation(id=obs_id, tool_name="pattern_search", text=json.dumps(observation_payload, ensure_ascii=False)),
        )

        # Ingest immediately — canonical_id is guaranteed to match.
        plant = self._session.plant
        from agentic.closure.obligation import PredicateRef
        # Plant's ingest_scan_claim will canonicalise the observation's
        # `pattern` via predicate_canonical_id_for_pattern_search (regex_match,
        # flags="i"). If the obligation's predicate is contains_string, we
        # need the observation pattern to also canonicalise to the same id —
        # so we re-shape the observation predicate name into the obligation's.
        # Simpler: bypass plant.ingest_scan_claim's canonical recomputation
        # by minting a ScanClaim directly here.
        from agentic.closure.claims import ScanClaim, ScanProvenance
        try:
            scan_claim = ScanClaim(
                id="",
                scope=scope,
                unit_type=unit_type,  # type: ignore[arg-type]
                predicate=predicate,  # use obligation's predicate verbatim
                scanned_units=frozenset(scanned),
                positive_units=frozenset(positive),
                negative_units=frozenset(negative),
                exhaustive=True,
                provenance=ScanProvenance(observation_id=obs_id),
            )
        except ValueError as exc:
            return _err("invalid_scan", str(exc))

        import dataclasses
        scan_claim = dataclasses.replace(scan_claim, id=plant.mint_id("claim"))
        self._session.claims.append(scan_claim)
        plant.run_closure(self._session.obligations, self._session.claims)

        status = _obligation_status(self._session.obligations)
        return (
            ok(
                "ProofScanResult",
                obligation_id=obligation.id,
                claim_id=scan_claim.id,
                scanned_count=len(scanned),
                positive_count=len(positive),
                negative_count=len(negative),
                obligation_status=status,
                must_finalize_next=status["must_finalize_next"],
            ),
            {"error": None, "must_finalize_next": status["must_finalize_next"]},
        )

    # ---------------------------------------------------------- scan internals

    def _compile_predicate(self, name: str, args: dict):
        pattern = args.get("pattern")
        if not pattern:
            raise ValueError(f"predicate {name!r} has no pattern arg.")
        if name == "contains_string":
            # Literal substring → regex.escape so we can reuse the same
            # iterator infrastructure. Case-insensitive default per the
            # contract args schema.
            case_sensitive = bool(args.get("case_sensitive", False))
            flags = 0 if case_sensitive else ureg.IGNORECASE
            return ureg.compile(ureg.escape(pattern), flags)
        if name == "regex_match":
            flags_str = args.get("flags", "i")
            flags = 0
            if "i" in flags_str:
                flags |= ureg.IGNORECASE
            if "m" in flags_str:
                flags |= ureg.MULTILINE
            if "s" in flags_str:
                flags |= ureg.DOTALL
            return ureg.compile(pattern, flags)
        raise ValueError(f"predicate {name!r} not scannable; only contains_string / regex_match.")

    def _scan(
        self,
        domain: frozenset[str],
        unit_type: str,
        compiled,
    ) -> Tuple[List[str], List[str], List[str], List[Dict[str, Any]]]:
        scanned: List[str] = []
        positive: List[str] = []
        negative: List[str] = []
        citations: List[Dict[str, Any]] = []
        units = sorted(domain)
        for uid in units:
            text = self._unit_text(uid, unit_type)
            scanned.append(uid)
            if text and compiled.search(text):
                positive.append(uid)
                if len(citations) < _MAX_CITATIONS:
                    m = compiled.search(text)
                    citations.append({"unit_id": uid, "match": m.group(0)[:80] if m else ""})
            else:
                negative.append(uid)
        return scanned, positive, negative, citations

    def _unit_text(self, unit_id: str, unit_type: str) -> str:
        if unit_type == "page":
            page = self._page_store.get(unit_id)
            return page.text_markdown if page else ""
        if unit_type == "passage":
            atom = self._inventory.passage_store.get(unit_id)
            return atom.text if atom else ""
        if unit_type == "table_row":
            atom = self._inventory.table_row_store.get(unit_id)
            return atom.text if atom else ""
        return ""


def _err(code: str, message: str) -> Tuple[str, Dict[str, Any]]:
    return err(code, message), {"error": code}


def _obligation_status(obligations: List) -> Dict[str, Any]:
    closed = [o.id for o in obligations if o.status == "CLOSED"]
    open_required = [
        {
            "id": o.id,
            "kind": o.kind,
            "unit_type": o.unit_type,
            "field": getattr(o, "field", None),
            "score_field": getattr(o, "score_field", None),
            "failure_kind": o.failure_kind,
            "diagnostic_data": o.diagnostic_data,
        }
        for o in obligations
        if o.required and o.status != "CLOSED"
    ]
    return {
        "summary": f"{len(closed)} closed / {len(open_required)} open required",
        "closed_obligation_ids": closed,
        "open_required": open_required,
        "must_finalize_next": len(open_required) == 0 and len(obligations) > 0,
    }
