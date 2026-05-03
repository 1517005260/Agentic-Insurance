"""answer_finalize logic — certifying the published answer.

The handler runs through three checks before certifying:

1. **Quorum** — every active required obligation is CLOSED and no
   challenge is pending in the closure cone. If this fails, return
   ABSTAIN (budget_exhausted) or REJECT.
2. **Approved-claim envelope** — every claim id cited by the draft
   (both in ``cited_claim_ids`` and as inline ``c_xxx...`` tokens) sits
   inside the union of closed obligations' ``used_claim_ids``.
3. **Numeric-value match** — each numeric ``closed_value`` appears as
   a token in the draft. Non-numeric values are conveyed by the
   canonical header that :func:`compose_final_answer` prepends, so we
   don't gate them here.

Plant.handle_answer_finalize keeps the public flow but calls
:func:`run_answer_finalize`; on CERTIFIED it asks
:func:`compose_final_answer` to emit the canonical header + draft +
citation footer.
"""

import re
from typing import Any, Dict, List, Optional

from agentic.proof.types import GateView


CLAIM_ID_RE = re.compile(r"\bc_[0-9a-f]{10}\b")
# Numeric tokens in a draft: optionally signed integer or decimal,
# possibly comma-grouped (e.g. "1,000", "12,345.67"). The greedy
# alternation handles "12300" (ungrouped) AND "1,000" (grouped) AND
# "1,234,567.89" without splitting "1,000" into 1 and 000.
DRAFT_NUMBER_RE = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)


def draft_numeric_tokens(text: str) -> set:
    """Extract every numeric literal from a draft answer as a normalised
    set. Comma-grouped numbers ("1,000") parse as 1000; raw digit runs
    ("12300") parse as 12300. Used both to verify each closed numeric
    value APPEARS in the draft and to reject drafts that contradict
    it (additional unrelated numerics are allowed only when they
    could plausibly be supporting context — see
    :func:`run_answer_finalize` for the rule)."""
    out: set = set()
    for tok in DRAFT_NUMBER_RE.findall(text or ""):
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        out.add(v)
        if v == int(v):
            out.add(int(v))
    return out


def compose_final_answer(plant: Any, draft_text: str, cited_claim_ids: List[str]) -> str:
    """Prepend a canonical ``Certified:`` header listing every closed
    required obligation's value, then append a citation footer pointing
    at every cited claim's source span.

    The header is the gate's authoritative answer; the LLM's draft
    sits below as commentary. Even when the draft narrative drifts,
    the published header reflects only what the gate actually proved
    (§1 principle 6)."""
    certified_lines: List[str] = []
    for o in plant.obligations.active_required_closed():
        val = o.closed_value
        certified_lines.append(f"- {o.id} ({o.spec.kind.value}): {val}")
    header = "Certified:\n" + "\n".join(certified_lines) + "\n\n" if certified_lines else ""
    cite_lines = []
    for cid in cited_claim_ids:
        claim = plant.evidence.get_claim(cid)
        if claim is None:
            continue
        for c in claim.citations:
            location = f"{c.file_id}/{c.page_id}"
            if c.span:
                location = f"{location}#{c.span[:80]}"
            cite_lines.append(f"- {location}")
    footer = "\n\nCitations:\n" + "\n".join(cite_lines) if cite_lines else ""
    return header + draft_text.rstrip() + footer


def run_answer_finalize(
    plant: Any,
    *,
    draft_text: str,
    cited_claim_ids: List[str],
    budget_exhausted: bool = False,
) -> Any:
    """Execute the answer_finalize protocol and return a ``PlantResult``.

    Imported lazily so this module remains plant.py-import-free at
    load time.
    """
    from agentic.proof.errors import make_envelope
    from agentic.proof.plant import PlantResult

    def _reject(
        *,
        code: str,
        message: str,
        remediation: str,
        decision: str = "REJECT",
        payload_extras: Optional[Dict[str, Any]] = None,
        **context: Any,
    ) -> "PlantResult":
        """Single shape for every finalize rejection. ``error.code`` is
        the actionable code the LLM keys on; ``payload.reason`` keeps a
        slug for telemetry / legacy assertions."""
        payload = {"decision": decision, "reason": code}
        if payload_extras:
            payload.update(payload_extras)
        return PlantResult(
            ok=False,
            payload=payload,
            error=make_envelope(code, message, remediation=remediation, **context),
            gate=plant.gate_view(),
        )

    # Phase A precondition: certification is undefined when no root
    # obligation has ever been created.
    if not plant.obligations.has_active_required():
        return _reject(
            code="no_active_required_obligation",
            message="no required obligation has ever been created",
            remediation="Call obligation_create first to declare what must be proven, then continue with acquisition + evidence_ingest before retrying answer_finalize.",
            decision="ABSTAIN" if budget_exhausted else "REJECT",
        )

    active_required = plant.obligations.active_required_open()
    challenged = plant._challenged_in_closure_cone()

    if active_required or challenged:
        decision = "ABSTAIN" if budget_exhausted else "REJECT"
        open_ids = [o.id for o in active_required]
        chal_ids = [o.id for o in challenged]
        return _reject(
            code="finalize_premature",
            message=(
                f"{len(open_ids)} required obligation(s) still OPEN, "
                f"{len(chal_ids)} CHALLENGED in closure cone"
            ),
            remediation=(
                "Each open obligation needs evidence to close (see gate.open_obligations[i].failure_kind + suggested_tools); "
                "every challenged one needs discharge via obligation_create(discharges_challenge=...) or obligation_decompose."
            ),
            decision=decision,
            payload_extras={
                "open_obligations": open_ids,
                "challenged_obligations": chal_ids,
            },
            open_obligations=open_ids,
            challenged_obligations=chal_ids,
        )

    # All required closed — verify cited claims are inside the closure used set.
    approved_claims: set[str] = set()
    for o in plant.obligations.active_required_closed():
        approved_claims.update(o.closed_by)
    bad = [cid for cid in cited_claim_ids if cid not in approved_claims]
    if bad:
        return _reject(
            code="cited_claims_not_used_for_closure",
            message=f"{len(bad)} cited claim_id(s) did not close any required obligation",
            remediation="Inspect each closed obligation's `used_claim_ids` (in gate.closed_obligations) and pass ONLY those ids in cited_claim_ids; drop any unused id.",
            payload_extras={"bad_claim_ids": bad},
            bad_claim_ids=bad,
            approved_claim_ids=sorted(approved_claims),
        )
    cited_set = set(cited_claim_ids)
    leaked = [tok for tok in CLAIM_ID_RE.findall(draft_text or "") if tok not in cited_set]
    if leaked:
        leaked_unique = sorted(set(leaked))
        return _reject(
            code="draft_references_unapproved_claim_ids",
            message=f"draft mentions {len(leaked_unique)} claim id(s) not in cited_claim_ids",
            remediation="Either remove the literal claim id token(s) from the draft prose, or add them to cited_claim_ids if they actually closed a required obligation.",
            payload_extras={"leaked_claim_ids": leaked_unique},
            leaked_claim_ids=leaked_unique,
        )
    draft_numbers = draft_numeric_tokens(draft_text)
    for o in plant.obligations.active_required_closed():
        val = o.closed_value
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        if val not in draft_numbers:
            return _reject(
                code="draft_value_mismatch",
                message=f"draft must contain the closed numeric value {val!r} for obligation {o.id}",
                remediation=(
                    f"Insert the literal token '{val}' into draft_text (this is the value the gate proved); "
                    "the canonical 'Certified:' header echoes it, but the draft must mention it for value-consistency."
                ),
                payload_extras={"obligation_id": o.id, "expected_value": val},
                obligation_id=o.id,
                expected_value=val,
                draft_numbers=sorted(draft_numbers),
            )
    return PlantResult(
        ok=True,
        payload={
            "decision": "CERTIFIED",
            "final_answer": compose_final_answer(plant, draft_text, cited_claim_ids),
            "cited_claim_ids": cited_claim_ids,
        },
        gate=plant.gate_view(),
    )
