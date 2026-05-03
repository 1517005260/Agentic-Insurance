"""``obligation_create`` — initialise a proof obligation.

The first call (no parent_id, no discharges_challenge) creates the
ROOT obligation, which freezes its predicate and kind for the rest of
the run. Subsequent calls either:

* attach a sub-obligation under a DECOMPOSED parent (rare — use
  ``obligation_decompose`` for batch construction), or
* serve as a replacement obligation that discharges a pending challenge
  (``discharges_challenge`` set).
"""
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pydantic import ValidationError

from agentic.proof import Plant
from agentic.proof.schema import (
    ObligationCreateArgsModel,
    compose_tool_description,
    flatten_refs,
    to_envelope,
)
from agentic.tools.base import BaseTool
from agentic.tools.proof._common import render

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DESCRIPTION = (
    "Create a typed proof obligation that the gate will verify "
    "before answer_finalize will certify. The FIRST call sets "
    "the ROOT obligation; its predicate and kind are FROZEN.\n\n"
    "Pick `kind` from the question shape:\n"
    "  - exists   — \"is there any X / what is X\" (one witness closes it)\n"
    "  - count    — \"how many X\" (needs an exhaustive scan)\n"
    "  - set      — \"list all X\" (needs an exhaustive scan)\n"
    "  - forall   — \"for every Y, is X true\" (exhaustive scan)\n"
    "  - negation — \"is X absent everywhere\" (sealed scope, exhaustive)\n"
    "  - argmax   — \"which Y has the largest score\" (per-unit witnesses)\n\n"
    "EXACT spec shape (every required field MUST be present):\n"
    "  {\n"
    "    \"kind\":      \"exists|count|set|forall|negation|argmax\",\n"
    "    \"scope\":     {\"file_ids\": [\"<file_id>\", ...],\n"
    "                   \"section_ids\": null|[\"<file_id>:sec_NNN\", ...],\n"
    "                   \"sealed\": false|true},\n"
    "    \"unit_type\": \"file\" | \"section\",\n"
    "    \"predicate\": {\"name\": \"<primitive>\", \"args\": {...}},\n"
    "    \"score\":     {\"name\": \"<extractor>\", \"args\": {...}},   // ARGMAX ONLY\n"
    "    \"tie_policy\":\"first|all|error\"   // ARGMAX ONLY (default \"first\")\n"
    "  }\n\n"
    "unit_type is STRICTLY \"file\" or \"section\" — never \"page\".\n\n"
    "Predicate primitives + their REQUIRED args:\n"
    "  contains_string  args={\"pattern\": \"<literal>\", \"case_sensitive\": false}\n"
    "  regex_match      args={\"pattern\": \"<regex>\"}        # anchor with literal terms; .* and \"\" are rejected\n"
    "  field_equals     args={\"field_path\": \"<key>\", \"value\": <any>}\n"
    "  numeric_compare  args={\"field_path\": \"<key>\", \"op\": \"<|<=|=|>=|>\", \"value\": <number>}\n"
    "  date_compare     args={\"field_path\": \"<key>\", \"op\": ..., \"value\": \"YYYY-MM-DD\"}\n"
    "  type_is          args={\"field_path\": \"<key>\", \"type\": \"str|int|float|list|dict|bool\"}\n"
    "  table_cell_contains args={\"row_label\": \"...\", \"column_label\": \"...\", \"pattern\": \"...\"}\n"
    "  section_title_contains args={\"pattern\": \"...\"}\n"
    "  range_in         args={\"field_path\": \"<key>\", \"lo\": <num>, \"hi\": <num>}   # lo<=hi, no NaN/inf\n"
    "  list_contains    args={\"field_path\": \"<key>\", \"value\": <any>}\n"
    "  and              EITHER {\"name\":\"and\", \"conjuncts\":[<pred>, <pred>, ...]}\n"
    "                   OR     {\"name\":\"and\", \"args\":{\"conjuncts\":[<pred>, <pred>, ...]}}\n"
    "                   (top-level conjuncts is the canonical shape; args.conjuncts also works)\n\n"
    "Score extractors for argmax (must be NUMERIC):\n"
    "  numeric_amount   args={}                                 # parses USD 100,000 / 1.2e3 / 12,300\n"
    "  percentage       args={}                                 # parses '80%' / '4.5 %'\n"
    "  integer_count    args={}                                 # parses 7 / -3 etc\n"
    "  date_iso         args={}                                 # parses ISO and natural-language dates\n"
    "  text_field is NOT orderable; the gate refuses it for argmax.\n\n"
    "Worked examples:\n"
    "  # \"What is the max AFYP rebate %?\" — single fact lookup\n"
    "  spec={\"kind\":\"exists\", \"scope\":{\"file_ids\":[\"<fid>\"],\"section_ids\":null,\"sealed\":false},\n"
    "        \"unit_type\":\"file\", \"predicate\":{\"name\":\"contains_string\",\"args\":{\"pattern\":\"AFYP rebate\",\"case_sensitive\":false}}}\n"
    "  # \"How many files mention X?\"\n"
    "  spec={\"kind\":\"count\", \"scope\":{\"file_ids\":[\"<fid_a>\",\"<fid_b>\"],\"section_ids\":null,\"sealed\":false},\n"
    "        \"unit_type\":\"file\", \"predicate\":{\"name\":\"contains_string\",\"args\":{\"pattern\":\"X\"}}}\n"
    "  # \"Which section has the largest premium amount?\"\n"
    "  spec={\"kind\":\"argmax\", \"scope\":{\"file_ids\":[\"<fid>\"],\"section_ids\":null,\"sealed\":false},\n"
    "        \"unit_type\":\"section\", \"predicate\":{\"name\":\"contains_string\",\"args\":{\"pattern\":\"Premium\"}},\n"
    "        \"score\":{\"name\":\"numeric_amount\",\"args\":{}}, \"tie_policy\":\"first\"}\n"
    "  # AND-composition (note: conjuncts is at the top level of the predicate dict)\n"
    "  predicate={\"name\":\"and\",\"conjuncts\":[\n"
    "      {\"name\":\"contains_string\",\"args\":{\"pattern\":\"Premium\"}},\n"
    "      {\"name\":\"contains_string\",\"args\":{\"pattern\":\"USD\"}}]}\n\n"
    "Subsequent calls pass parent_id (sub-obligation under a DECOMPOSED parent) "
    "or discharges_challenge (replacement that resolves a pending challenge)."
)


class ObligationCreateTool(BaseTool):
    def __init__(self, plant: Plant):
        self._plant = plant

    @property
    def name(self) -> str:
        return "obligation_create"

    def get_schema(self) -> Dict[str, Any]:
        parameters = flatten_refs(
            ObligationCreateArgsModel.model_json_schema(mode="serialization")
        )
        return {
            "type": "function",
            "function": {
                "name": "obligation_create",
                "description": compose_tool_description(
                    ObligationCreateArgsModel, _DESCRIPTION
                ),
                "parameters": parameters,
            },
        }

    def execute(
        self,
        context: "AgentContext",
        spec: Optional[Dict[str, Any]] = None,
        discharges_challenge: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        from agentic.tools.proof._common import reject_validation
        try:
            args = ObligationCreateArgsModel.model_validate(
                {"spec": spec, "discharges_challenge": discharges_challenge}
            )
        except ValidationError as exc:
            return reject_validation(self._plant, exc, ObligationCreateArgsModel)
        result = self._plant.handle_obligation_create(**args.to_domain())
        log: Dict[str, Any] = {"ok": result.ok}
        if result.ok:
            log["obligation_id"] = result.payload.get("obligation_id")
            log["is_root"] = result.payload.get("is_root", False)
        else:
            log["error"] = (result.error or {}).get("code")
        context.add_retrieval_log(tool_name=self.name, tokens=0, metadata=log)
        return render(result, "ObligationCreate"), log
