"""LLM-facing proof-state tools.

Five tools, no overlap:

* ``obligation_create``     — create a root or replacement obligation.
* ``obligation_decompose``  — split a parent into rule-validated children.
* ``obligation_challenge``  — open a mechanically dischargeable challenge.
* ``evidence_ingest``       — submit a typed claim from an observation.
* ``answer_finalize``       — request CERTIFIED / ABSTAIN / REJECT decision.

``obligation_retire``, ``obligation_try_close`` and ``evidence_inspect``
do NOT exist as LLM tools — the plant handles retirement/closure during
``reconcile`` and surfaces the state via gate.diagnose appended to
state-changing tool results.
"""

from agentic.tools.proof.answer_finalize import AnswerFinalizeTool
from agentic.tools.proof.evidence_ingest import EvidenceIngestTool
from agentic.tools.proof.obligation_challenge import ObligationChallengeTool
from agentic.tools.proof.obligation_create import ObligationCreateTool
from agentic.tools.proof.obligation_decompose import ObligationDecomposeTool


__all__ = [
    "AnswerFinalizeTool",
    "EvidenceIngestTool",
    "ObligationChallengeTool",
    "ObligationCreateTool",
    "ObligationDecomposeTool",
]
