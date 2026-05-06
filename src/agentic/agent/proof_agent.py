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
from typing import Any, Dict, List, Optional

import tiktoken

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
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

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

    def run(self, query: str, tracer: Optional[Tracer] = None) -> ProofRunResult:
        context = AgentContext()
        session = ProofSession.build(
            inventory=self.inventory,
            budget=Budget(remaining_steps=self.max_loops, max_loops=self.max_loops),
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
            {"role": "system", "content": self.system_prompt},
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

        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1
            session.budget = Budget(
                remaining_steps=self.max_loops - loop_idx,
                max_loops=self.max_loops,
            )

            if self._token_count(messages) > self.max_token_budget:
                exit_reason = "token_budget_exceeded"
                break

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

    def _token_count(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
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
