"""Shared helpers for proof-state tools.

The five proof tools share an envelope shape so the LLM sees the same
``observation_type`` family on every state-changing call. The plant
returns a :class:`agentic.proof.PlantResult`; this module turns it into
the JSON-serialised result the agent loop expects.

A successful tool result carries two things the LLM needs: the per-tool
payload (e.g., ``obligation_id`` for create) and a ``gate`` snapshot
derived from :func:`Plant.gate_view`. The gate field is the post-call
state view of the proof gate.
"""
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from agentic.proof import GateView, Plant, PlantResult, ToolDiagnostic
from agentic.proof.schema.envelope import to_envelope as _to_envelope
from agentic.tools.acquisition._common import err as _acq_err


def _diag_dump(d: ToolDiagnostic) -> Dict[str, Any]:
    return _diag_to_dict(d)


def _gate_to_dict(gate: GateView) -> Dict[str, Any]:
    """LLM-facing wire format. ``open_obligations`` already carries
    each obligation's ``failure_kind`` / ``suggested_tools`` /
    ``suggested_repair_kind`` / ``cursor`` inline (see
    :func:`agentic.proof.gate_view.build_gate_view`), so the
    standalone ``diagnostics`` array — which would repeat the same
    strings — is intentionally NOT included in the wire format. The
    underlying :class:`GateView.diagnostics` field remains for
    Python-side consumers that want the typed ``ToolDiagnostic`` list
    (test stubs, telemetry)."""
    return {
        "open_obligations": gate.open_obligations,
        "closed_obligations": gate.closed_obligations,
        "challenged_obligations": gate.challenged_obligations,
        "recent_claims": gate.recent_claims,
        "abstain_recommended": gate.abstain_recommended,
        "abstain_reason": gate.abstain_reason,
    }


def _diag_to_dict(d: ToolDiagnostic) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "obligation_id": d.obligation_id,
        "failure_kind": d.failure_kind,
        "suggested_tools": list(d.suggested_tools),
    }
    if d.cursor is not None:
        out["cursor"] = d.cursor
    return out


def envelope_ok(observation_type: str, result: PlantResult, **fields: Any) -> str:
    payload: Dict[str, Any] = {
        "observation_type": observation_type,
        "ok": True,
    }
    payload.update(result.payload)
    payload.update(fields)
    if result.gate is not None:
        payload["gate"] = _gate_to_dict(result.gate)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def envelope_err(observation_type: str, result: PlantResult, **fields: Any) -> str:
    err = result.error or {"code": "unknown", "message": "plant returned no detail"}
    payload: Dict[str, Any] = {
        "observation_type": observation_type,
        "ok": False,
        "error": err,
    }
    if result.payload:
        payload.update(result.payload)
    payload.update(fields)
    if result.gate is not None:
        payload["gate"] = _gate_to_dict(result.gate)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render(result: PlantResult, observation_type: str) -> str:
    """Pick the right envelope based on ``result.ok``."""
    return envelope_ok(observation_type, result) if result.ok else envelope_err(observation_type, result)


def reject_validation(
    plant: Plant,
    exc: ValidationError,
    model: type[BaseModel],
) -> tuple[str, Dict[str, Any]]:
    """Convert a pydantic ``ValidationError`` into the same envelope
    shape plant rejections use, with the gate snapshot attached so
    the LLM reads one wire format regardless of which validation
    layer caught the error.

    Returns ``(rendered_json, log_dict)`` — the same ``(str, dict)``
    tuple every tool's ``execute()`` returns on the success path.
    """
    gate_dict = _gate_to_dict(plant.gate_view())
    envelope = _to_envelope(exc, gate=gate_dict, model=model)
    code = envelope["code"]
    return (
        _acq_err(
            code,
            envelope["message"],
            remediation=envelope.get("remediation"),
            valid_example=envelope.get("valid_example"),
            affected_fields=envelope.get("affected_fields"),
            **envelope.get("context", {}),
        ),
        {"error": code},
    )
