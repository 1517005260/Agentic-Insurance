"""``evidence_ingest`` — submit a claim derived from an observation.

The agent finds an observation it wants to count as evidence (e.g., a
``read_page`` result that shows AXA's premium rebate table) and submits
a typed claim. The plant validates the claim shape against the
observation, then auto-binds the claim to every matching open
obligation. Auto-extracted claims (e.g., ScanClaims from exhaustive
pattern_search) are emitted by the plant directly and do not require
this tool.
"""
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pydantic import ValidationError

from agentic.proof import Plant
from agentic.proof.schema import (
    EvidenceIngestArgsModel,
    compose_tool_description,
    flatten_refs,
    to_envelope,
)
from agentic.tools.base import BaseTool
from agentic.tools.proof._common import render

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DESCRIPTION = (
    "Submit a typed claim from an existing observation. The "
    "plant validates citations against the observation snapshot, "
    "evaluates the predicate against the cited span, checks "
    "scope/unit_type match, runs score-extractor verification "
    "(for value_map), and auto-binds the claim to every "
    "matching open obligation.\n\n"
    "v1 accepts two claim_types from the LLM:\n"
    "  - WitnessClaim: proves predicate(unit) holds for each "
    "    listed positive_unit, citing specific spans. For "
    "    argmax, also carries value_map mapping each unit to "
    "    its extracted score.\n"
    "  - ScanClaim: a COMPLETE partition of "
    "    inventory.units(scope, unit_type) into positive and "
    "    negative. ScanClaim must source from a "
    "    PAGE_HITS_EXHAUSTIVE observation whose scanned_units "
    "    cover every indexed page in scope. Auto-extracted "
    "    ScanClaims from exhaustive pattern_search do NOT "
    "    require this tool.\n\n"
    "EXACT claim_candidate shapes:\n\n"
    "  WitnessClaim (no value_map — for exists / forall etc):\n"
    "  {\n"
    "    \"claim_type\": \"WitnessClaim\",\n"
    "    \"scope\":      {\"file_ids\": [...], \"section_ids\": null|[...], \"sealed\": false},\n"
    "    \"unit_type\":  \"file\" | \"section\",\n"
    "    \"predicate\": {\"name\": ..., \"args\": {...}},   # MUST equal the obligation's predicate (or be entailed by it)\n"
    "    \"positive_units\": [\"<unit_id>\", ...],          # each must be cited below\n"
    "    \"negative_units\": [],\n"
    "    \"citations\": [\n"
    "      {\"file_id\": \"...\", \"page_id\": \"p_NNNN\", \"span\": \"<verbatim text from the page>\"}\n"
    "    ]\n"
    "  }\n\n"
    "  WitnessClaim WITH value_map (for argmax):\n"
    "  {\n"
    "    \"claim_type\": \"WitnessClaim\",\n"
    "    \"scope\": ..., \"unit_type\": ...,\n"
    "    \"predicate\": {\"name\": ..., \"args\": {...}},\n"
    "    \"score\":     {\"name\": \"numeric_amount\", \"args\": {}},  # MUST equal the obligation's score\n"
    "    \"positive_units\": [\"<unit_id>\"],\n"
    "    \"value_map\": {\"<unit_id>\": <number>},   # keys subset positive_units; plant verifies value vs cited span\n"
    "    \"citations\": [{\"file_id\": ..., \"page_id\": ..., \"span\": \"<contains the number>\"}]\n"
    "  }\n\n"
    "  ScanClaim (mirrors an exhaustive pattern_search; usually auto-extracted):\n"
    "  {\n"
    "    \"claim_type\": \"ScanClaim\",\n"
    "    \"scope\": ..., \"unit_type\": ...,\n"
    "    \"predicate\": {\"name\": ..., \"args\": {...}},\n"
    "    \"positive_units\": [...],   # MUST equal the observation's actual hits\n"
    "    \"negative_units\": [...],   # complement; positive ∪ negative = full inventory in scope\n"
    "    \"citations\": [{\"file_id\": ..., \"page_id\": \"<any positive page>\"}]\n"
    "  }\n\n"
    "PICKING observation_id:\n"
    "  The observation_id must point at an observation that "
    "ACTUALLY contains the snapshot of every cited page. In "
    "practice that means: cite a page only via the SAME "
    "`read_page` call that fetched that page (or via the "
    "exhaustive `pattern_search` whose `scanned_units` cover "
    "those pages). Search-tool observations (semantic_search, "
    "bm25_search) carry only snippets and are NOT valid "
    "citation sources for WitnessClaim. If you want to cite "
    "two pages, call `read_page` once with both page_ids and "
    "use that single observation_id.\n\n"
    "Validation hard rules (refused at ingest):\n"
    "  - WitnessClaim must declare a predicate; the plant "
    "    evaluates it against the cited span text and rejects "
    "    if it returns False on any positive unit's citation.\n"
    "  - Each positive_unit must have at least one citation "
    "    that resolves to that unit (file_id matches for "
    "    unit_type=\"file\"; the cited page lies inside the "
    "    section's page range for unit_type=\"section\").\n"
    "  - Citation span must appear verbatim in the observation "
    "    snapshot — payload['text'] for PAGE_CONTENT, or "
    "    payload['results'][i]['text_markdown'] for read_page-"
    "    style results that match the cited page_id.\n"
    "  - ScanClaim partition must EQUAL the observation's "
    "    hits; partial repaints are rejected.\n"
    "  - positive_units and negative_units must each contain "
    "    no duplicates and no overlap.\n\n"
    "Returns ``observation_id``, ``claim_id`` (use this in "
    "answer_finalize's cited_claim_ids), ``closures_triggered`` "
    "(obligations the new claim closed), and ``gate`` (post-"
    "call gate snapshot)."
)


class EvidenceIngestTool(BaseTool):
    def __init__(self, plant: Plant):
        self._plant = plant

    @property
    def name(self) -> str:
        return "evidence_ingest"

    def get_schema(self) -> Dict[str, Any]:
        parameters = flatten_refs(
            EvidenceIngestArgsModel.model_json_schema(mode="serialization")
        )
        return {
            "type": "function",
            "function": {
                "name": "evidence_ingest",
                "description": compose_tool_description(
                    EvidenceIngestArgsModel, _DESCRIPTION
                ),
                "parameters": parameters,
            },
        }

    def execute(
        self,
        context: "AgentContext",
        observation_id: Optional[str] = None,
        claim_candidate: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        from agentic.tools.acquisition._common import err
        try:
            args = EvidenceIngestArgsModel.model_validate(
                {
                    "observation_id": observation_id,
                    "claim_candidate": claim_candidate,
                }
            )
        except ValidationError as exc:
            from agentic.tools.proof._common import reject_validation
            return reject_validation(self._plant, exc, EvidenceIngestArgsModel)
        result = self._plant.handle_evidence_ingest(**args.to_domain())
        log: Dict[str, Any] = {"ok": result.ok, "observation_id": args.observation_id}
        if result.ok:
            log["claim_id"] = result.payload.get("claim_id")
            log["closures_triggered"] = result.payload.get("closures_triggered", [])
        else:
            log["error"] = (result.error or {}).get("code")
        context.add_retrieval_log(tool_name=self.name, tokens=0, metadata=log)
        return render(result, "EvidenceIngest"), log
