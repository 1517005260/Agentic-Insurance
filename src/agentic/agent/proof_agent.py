"""ProofAgent — drives the typed closure kernel.

Composes the same tool-dispatch primitives BaseAgent uses but runs
its own loop: the kernel decides when finalization is allowed; the
LLM decides what acquisition step to take next. Strict mode only —
the run ends in CERTIFIED or ABSTAIN.

Per-run state lives on a fresh ``ProofSession``. The proof tools
(plan_init / gap_propose / claim_ingest / finalize) share that
session through closure; acquisition tool results are wrapped with
an ``observation_id`` and recorded so claim citations can be
verified verbatim.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config.shared import shared_tiktoken_encoder

from agentic.agent.base import _make_emitter
from agentic.agent.prompts.proof_system import PROOF_SYSTEM_PROMPT
from agentic.closure.budget import Budget
from agentic.closure.inventory import Inventory, InventoryAdapter
from agentic.closure.session import Observation, ProofSession
from agentic.core.context import AgentContext
from agentic.tools.acquisition._common import err
from agentic.tools.proof import (
    ProofClaimIngestTool,
    ProofScanTool,
    ProofFinalizeTool,
    ProofGapProposeTool,
    ProofPlanInitTool,
)
from agentic.tools.registry import ToolRegistry
from model_client import LLMClient
from tracer import TraceSession, Tracer


logger = logging.getLogger(__name__)


@dataclass
class ProofRunResult:
    decision: str  # CERTIFIED | ABSTAIN | LLM_ERROR | NO_FINALIZE
    answer: str
    obligations: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    candidate_gaps: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    loops: int = 0
    total_cost: float = 0.0
    exit_reason: str = ""


class ProofAgent:
    """Acquisition-aware loop with a kernel-controlled stopping rule."""

    def __init__(
        self,
        llm_client: LLMClient,
        acquisition_tools: ToolRegistry,
        inventory: Inventory,
        page_store,
        inventory_store,
        *,
        system_prompt: Optional[str] = None,
        max_loops: int = 16,
        max_token_budget: int = 128_000,
        verbose: bool = False,
    ) -> None:
        self.llm = llm_client
        self.acquisition_tools = acquisition_tools
        self.inventory = inventory
        self.page_store = page_store
        self.inventory_store = inventory_store     # raw store; proof_scan reads atom text
        self.system_prompt = system_prompt or PROOF_SYSTEM_PROMPT
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose
        self.tokenizer = shared_tiktoken_encoder("gpt-4o")

    def warm_up(self) -> Dict[str, float]:
        timings: Dict[str, float] = {}
        for name in self.acquisition_tools.list_tools():
            tool = self.acquisition_tools.get(name)
            hook = getattr(tool, "warm_up", None)
            if not callable(hook):
                continue
            t0 = time.perf_counter()
            try:
                hook()
                timings[name] = time.perf_counter() - t0
            except Exception as exc:
                logger.warning("ProofAgent.warm_up: %s failed: %s", name, exc)
                timings[name] = -1.0
        return timings

    # ------------------------------------------------------------- run

    def run(
        self,
        query: str,
        tracer: Optional[Tracer] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        *,
        max_loops: Optional[int] = None,
        max_token_budget: Optional[int] = None,
        system_prompt: Optional[str] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ProofRunResult:
        """Run the proof loop.

        ``on_event`` mirrors :meth:`BaseAgent.run`: optional sync
        ``(event, data)`` callback. In addition to ``status`` /
        ``tool_call`` / ``tool_result`` / ``final``, this loop emits
        ``obligation`` / ``claim`` / ``gap`` deltas after every tool
        call so a streaming UI can update the proof board.

        ``max_loops`` / ``max_token_budget`` / ``system_prompt`` (all
        optional, ``None`` → constructor value) let the web layer push
        the admin-config overrides per request without rebuilding the
        agent singleton.

        ``cancel_check`` (optional) is a no-arg predicate polled at
        each loop boundary; True exits with ``exit_reason='client_disconnect'``.
        Same contract as :meth:`BaseAgent.run` so the runner side can
        wire ``EventBus.is_closed`` uniformly.
        """
        effective_max_loops = max_loops if max_loops is not None else self.max_loops
        effective_max_tokens = (
            max_token_budget if max_token_budget is not None else self.max_token_budget
        )
        effective_system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        emit = _make_emitter(on_event)
        context = AgentContext()
        session = ProofSession.build(
            inventory=self.inventory,
            budget=Budget(remaining_steps=effective_max_loops, max_loops=effective_max_loops),
        )

        proof_tools = ToolRegistry()
        proof_tools.register(ProofPlanInitTool(session))
        proof_tools.register(ProofGapProposeTool(session))
        proof_tools.register(ProofClaimIngestTool(session))
        proof_tools.register(ProofScanTool(session, self.page_store, self.inventory_store))
        proof_tools.register(ProofFinalizeTool(session))

        all_tools = ToolRegistry()
        for name in self.acquisition_tools.list_tools():
            all_tools.register(self.acquisition_tools.get(name))
        for name in proof_tools.list_tools():
            all_tools.register(proof_tools.get(name))
        tool_schemas = all_tools.get_all_schemas()

        trace_session: Optional[TraceSession] = tracer.session(query) if tracer else None
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": query},
        ]
        trajectory: List[Dict[str, Any]] = []

        total_cost = 0.0
        observation_counter = 0
        finalized: Optional[Dict[str, Any]] = None
        loop_count = 0
        exit_reason = ""
        # Watchdog: when the previous turn ended on must_finalize_next,
        # the agent's next call must be proof_finalize. Otherwise we
        # inject a reminder.
        must_finalize_pending = False

        # State trackers for emitting obligation/claim/gap deltas. We
        # snapshot status by id before each tool dispatch and diff after,
        # rather than the agent code threading callbacks through every
        # internal mutation site.
        last_obligation_status: Dict[str, tuple] = {}
        seen_claim_ids: set[str] = set()
        last_gap_status: Dict[str, str] = {}

        for loop_idx in range(effective_max_loops):
            loop_count = loop_idx + 1
            session.budget = Budget(
                remaining_steps=effective_max_loops - loop_idx,
                max_loops=effective_max_loops,
            )

            # Cancellation gate (mirrors BaseAgent.run). At the loop
            # boundary, before any LLM / tool work; in-flight tool
            # results from the previous iteration are already in
            # ``messages`` so the agent state is consistent if a
            # later resume happens.
            if cancel_check is not None and cancel_check():
                emit("status", {"phase": "client_disconnect"})
                exit_reason = "client_disconnect"
                break

            if self._token_count(messages, system_prompt=effective_system_prompt) > effective_max_tokens:
                exit_reason = "token_budget_exceeded"
                break

            emit(
                "status",
                {
                    "phase": "thinking",
                    "loop": loop_count,
                    "max_loops": effective_max_loops,
                    "remaining_steps": session.budget.remaining_steps,
                },
            )

            if must_finalize_pending:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[kernel reminder] The previous proof_claim_ingest "
                            "/ proof_scan returned must_finalize_next=true "
                            "(0 open required obligations). Your next tool "
                            "call MUST be proof_finalize. No further "
                            "acquisition or proof_gap_propose."
                        ),
                    }
                )
                must_finalize_pending = False

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as exc:
                logger.warning("ProofAgent.run: llm error: %s", exc)
                exit_reason = "llm_error"
                break

            total_cost += response.get("cost", 0.0)
            message = response["message"]
            messages.append(message)

            tool_calls = message.get("tool_calls") or []
            # Emit "thought" only when tool_calls are also present.
            # ProofAgent's strict mode stops the loop the moment a tool
            # call is missing; that ``content`` isn't reasoning but the
            # tail of an abstain, so it should flow through final.answer
            # and not into the timeline.
            content_str = (message.get("content") or "").strip()
            if content_str and tool_calls:
                emit(
                    "thought",
                    {"loop": loop_count, "text": content_str},
                )
            if not tool_calls:
                # Strict mode: if the LLM produced free text without finalize, we treat it as a stop and abstain.
                exit_reason = "no_tool_calls"
                break

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                emit(
                    "tool_call",
                    {"loop": loop_count, "name": func_name, "args": func_args},
                )

                tool_result, tool_log = all_tools.execute(func_name, context, **func_args)

                # Record acquisition results as observations so claims can cite them.
                if func_name not in {
                    "proof_plan_init",
                    "proof_gap_propose",
                    "proof_claim_ingest",
                    "proof_finalize",
                }:
                    observation_counter += 1
                    obs_id = f"obs_{observation_counter:04d}"
                    session.append_observation(
                        Observation(id=obs_id, tool_name=func_name, text=tool_result),
                    )
                    tool_result = _annotate_with_observation_id(tool_result, obs_id)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )
                turn_record = {
                    "loop": loop_count,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log,
                }
                trajectory.append(turn_record)
                if trace_session is not None:
                    trace_session.event("trajectory", turn_record)

                emit(
                    "tool_result",
                    {
                        "loop": loop_count,
                        "name": func_name,
                        "preview": tool_result[:300],
                        "observation_id": (
                            f"obs_{observation_counter:04d}"
                            if func_name not in {
                                "proof_plan_init", "proof_gap_propose",
                                "proof_claim_ingest", "proof_finalize",
                            } else None
                        ),
                        "must_finalize_next": tool_log.get("must_finalize_next", False),
                        # Internal-only: full tool result for runner-side
                        # citation extraction. Stripped by the SSE runner
                        # before frames hit the wire — see BaseAgent.run.
                        "_full_result": tool_result,
                    },
                )

                # Re-plan wipes obligations / candidate_gaps in place;
                # if it reuses an id with the same (status, failure_kind)
                # the diff would emit nothing and the UI would keep
                # stale kind/scope from the old plan. Force-reset the
                # trackers so every post-replan obligation re-emits.
                if func_name == "proof_plan_init":
                    for oid in list(last_obligation_status):
                        emit("obligation", {"id": oid, "status": "REMOVED"})
                    last_obligation_status.clear()
                    for cid in list(seen_claim_ids):
                        emit("claim", {"id": cid, "status": "REMOVED"})
                    seen_claim_ids.clear()
                    for gid in list(last_gap_status):
                        emit("gap", {"id": gid, "status": "REMOVED"})
                    last_gap_status.clear()

                # Diff session state after the tool ran. Emit one event
                # per added/changed obligation, claim, or candidate gap.
                _emit_session_deltas(
                    emit, session, last_obligation_status, seen_claim_ids, last_gap_status
                )

                if func_name == "proof_finalize":
                    decision = tool_log.get("decision")
                    if decision in {"CERTIFIED", "ABSTAIN"}:
                        finalized = {"decision": decision, "result": tool_result}
                if tool_log.get("must_finalize_next"):
                    must_finalize_pending = True

            if finalized is not None:
                exit_reason = "finalized"
                break
        else:
            exit_reason = "max_loops_exhausted"

        decision, answer = _decide_outcome(finalized, exit_reason, session)

        if trace_session is not None:
            trace_session.finalize(
                answer=answer,
                summary={
                    "loops": loop_count,
                    "decision": decision,
                    "exit_reason": exit_reason,
                    "total_cost": total_cost,
                    "context_summary": context.get_summary(),
                },
            )

        emit(
            "final",
            {
                "answer": answer,
                "decision": decision,
                "exit_reason": exit_reason,
                "loops": loop_count,
                "total_cost": total_cost,
                "obligations_total": len(session.obligations),
                "claims_total": len(session.claims),
                "candidate_gaps_total": len(session.candidate_gaps),
            },
        )

        return ProofRunResult(
            decision=decision,
            answer=answer,
            obligations=list(session.obligations),
            claims=list(session.claims),
            candidate_gaps=list(session.candidate_gaps),
            trajectory=trajectory,
            loops=loop_count,
            total_cost=total_cost,
            exit_reason=exit_reason,
        )

    def _token_count(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> int:
        # ``is not None`` (not ``or``) so an empty-string override is
        # respected — symmetric with how ``run()`` decides which prompt
        # to inject as messages[0].
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        total = len(self.tokenizer.encode(prompt))
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += len(self.tokenizer.encode(item.get("text", "")))
            elif content:
                total += len(self.tokenizer.encode(str(content)))
        return total


# ---------------------------------------------------------------- helpers


def _emit_session_deltas(
    emit: Callable[[str, Dict[str, Any]], None],
    session,
    last_obligation_status: Dict[str, tuple],
    seen_claim_ids: set,
    last_gap_status: Dict[str, str],
) -> None:
    """Diff ProofSession state vs trackers and emit per-id change events.

    Trackers are mutated in place so the next call only fires on real
    deltas. Cheap because obligations / claims / gaps are O(few dozen)
    per run.

    ``proof_plan_init`` (and any future re-plan) wipes the obligation /
    candidate_gap lists; we emit ``status='REMOVED'`` for any tracked id
    that disappeared so a streaming UI can prune its proof board
    instead of leaving stale rows.
    """
    # ----------------------------------------------------------- obligations
    current_ob_ids = {ob.id for ob in session.obligations}
    for removed_id in [oid for oid in last_obligation_status if oid not in current_ob_ids]:
        del last_obligation_status[removed_id]
        emit("obligation", {"id": removed_id, "status": "REMOVED"})

    for ob in session.obligations:
        key = (ob.status, ob.failure_kind)
        if last_obligation_status.get(ob.id) != key:
            last_obligation_status[ob.id] = key
            emit(
                "obligation",
                {
                    "id": ob.id,
                    "kind": str(ob.kind),
                    "status": ob.status,
                    "required": ob.required,
                    "failure_kind": ob.failure_kind,
                },
            )

    # ---------------------------------------------------------------- claims
    current_claim_ids = {cl.id for cl in session.claims}
    for removed_id in list(seen_claim_ids - current_claim_ids):
        seen_claim_ids.discard(removed_id)
        emit("claim", {"id": removed_id, "status": "REMOVED"})

    for cl in session.claims:
        if cl.id in seen_claim_ids:
            continue
        seen_claim_ids.add(cl.id)
        # Claims have heterogeneous shapes; pick a stable subset.
        emit(
            "claim",
            {
                "id": cl.id,
                "kind": getattr(cl, "claim_type", type(cl).__name__),
                "by": [getattr(cl, "citation", None).observation_id] if hasattr(cl, "citation") and cl.citation else [],
            },
        )

    # ---------------------------------------------------------------- gaps
    current_gap_ids = {
        getattr(g, "id", None) for g in session.candidate_gaps if getattr(g, "id", None)
    }
    for removed_id in [gid for gid in last_gap_status if gid not in current_gap_ids]:
        del last_gap_status[removed_id]
        emit("gap", {"id": removed_id, "status": "REMOVED"})

    for gap in session.candidate_gaps:
        gap_id = getattr(gap, "id", None)
        gap_status = getattr(gap, "status", "ACTIVE")
        if gap_id is None:
            continue
        if last_gap_status.get(gap_id) != gap_status:
            last_gap_status[gap_id] = gap_status
            emit(
                "gap",
                {
                    "id": gap_id,
                    "kind": getattr(gap, "kind", None),
                    "status": gap_status,
                },
            )


def _annotate_with_observation_id(tool_result: str, observation_id: str) -> str:
    """Inject `observation_id` at the JSON top level if the result is JSON; else prefix a header."""

    try:
        parsed = json.loads(tool_result)
    except json.JSONDecodeError:
        return f"observation_id={observation_id}\n{tool_result}"
    if isinstance(parsed, dict):
        parsed["observation_id"] = observation_id
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return f"observation_id={observation_id}\n{tool_result}"


def _decide_outcome(
    finalized: Optional[Dict[str, Any]],
    exit_reason: str,
    session: ProofSession,
) -> tuple[str, str]:
    if finalized is not None:
        try:
            payload = json.loads(finalized["result"])
            answer = payload.get("answer") or ""
        except (json.JSONDecodeError, AttributeError):
            answer = ""
        return finalized["decision"], answer

    open_required = [o for o in session.obligations if o.required and o.status != "CLOSED"]
    if exit_reason == "max_loops_exhausted":
        reason = "max_loops_exhausted"
    elif exit_reason == "no_tool_calls":
        reason = "agent_emitted_no_tool_calls"
    elif exit_reason == "llm_error":
        reason = "llm_error"
    elif exit_reason == "token_budget_exceeded":
        reason = "token_budget_exceeded"
    else:
        reason = exit_reason or "unknown"

    summary_lines = [f"Abstain: {reason}"]
    for o in open_required:
        summary_lines.append(f"- {o.id} ({o.kind}) {o.failure_kind or 'open'}")
    return "ABSTAIN", "\n".join(summary_lines) + "\n"
