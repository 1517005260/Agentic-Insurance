"""ProofAgent — gate-controlled agent loop.

Sibling to :class:`agentic.agent.base.BaseAgent`. Uses the same LLM
client, tokenizer, and tracer plumbing, but the loop is reshaped around
the proof gate:

* **Two-stage boot** (Phase A → Phase B). The plant gates evidence
  ingestion until at least one required obligation exists; before then
  the agent can only call acquisition tools and obligation_create /
  obligation_challenge.
* **No natural-final**. An assistant message without tool_calls does
  NOT terminate the run. Instead, the gate replies with a "you must
  call answer_finalize or another tool" nudge and the loop continues.
  The only terminal paths are CERTIFIED (from answer_finalize) or
  ABSTAIN (from budget exhaustion / T³ stall).
* **gate.diagnose** is appended to every state-changing tool result.
  The post-call gate snapshot is what the LLM sees alongside the raw
  observation payload.
* **T³ stall detection**. We track the size of the active-required-open
  set; if it stagnates for ``stall_window`` consecutive loops without
  any claim ingest or challenge change, we abandon the run with ABSTAIN
  rather than burning budget on uninformative tail.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from agentic.agent.prompts import SYSTEM_PROMPT
from agentic.core.context import AgentContext
from agentic.proof import (
    Citation,
    GateView,
    ObservationType,
    Plant,
)
from agentic.tools.acquisition._common import err
from agentic.tools.registry import ToolRegistry
from model_client import LLMClient
from tracer import TraceSession, Tracer


logger = logging.getLogger(__name__)


# tool_name → ObservationType for plant registration. Pure-acquisition
# tools whose results are pure navigation hints (no claim derivation)
# still register an observation so the agent can cite them in
# challenges via evidence_ids.
_TOOL_OBSERVATION_TYPE: Dict[str, ObservationType] = {
    "list_files": ObservationType.FILE_LIST,
    "toc": ObservationType.TOC,
    "semantic_search": ObservationType.PAGE_CANDIDATES,
    "bm25_search": ObservationType.PAGE_CANDIDATES,
    "graph_explore": ObservationType.PAGE_CANDIDATES,
    "pattern_search": ObservationType.PAGE_HITS_EXHAUSTIVE,
    "read_page": ObservationType.PAGE_CONTENT,
    "code_run": ObservationType.COMPUTE_RESULT,
}


_PROOF_TOOLS = frozenset({
    "obligation_create",
    "obligation_decompose",
    "obligation_challenge",
    "evidence_ingest",
    "answer_finalize",
})


class ProofAgent:
    """Gate-controlled agent. Uses the same toolset surface as BaseAgent
    plus the five proof tools, but governs termination through the
    plant rather than by trusting a no-tool LLM message."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        plant: Plant,
        system_prompt: Optional[str] = None,
        max_loops: int = 24,
        max_token_budget: int = 128000,
        stall_window: int = 4,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.plant = plant
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.stall_window = stall_window
        self.verbose = verbose
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------- warm_up

    def warm_up(self) -> Dict[str, float]:
        timings: Dict[str, float] = {}
        for name in self.tools.list_tools():
            tool = self.tools.get(name)
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

    # ------------------------------------------------------------- helpers

    def _calculate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += len(self.tokenizer.encode(str(content)))
        return total

    def _budget_exhausted(self, messages: List[Dict[str, Any]], loop_count: int) -> bool:
        if loop_count >= self.max_loops:
            return True
        return self._calculate_message_tokens(messages) > self.max_token_budget

    def _is_proof_tool(self, tool_name: str) -> bool:
        return tool_name in _PROOF_TOOLS

    def _maybe_register_observation(
        self,
        tool_name: str,
        tool_result_json: str,
    ) -> Tuple[Optional[str], str]:
        """Parse a successful acquisition tool result into a plant Observation,
        run the auto-extractor, and return the (possibly augmented) tool
        result for the LLM.

        The augmented result includes ``observation_id`` (so the LLM can
        cite it in evidence_ingest / obligation_challenge) and the
        post-extract ``gate`` snapshot when state changed. Proof tools
        already produce their own envelope and are skipped here.

        Returns ``(observation_id, augmented_result_json)``. The
        observation_id is ``None`` when the tool wasn't registered or
        the parse failed.
        """
        if self._is_proof_tool(tool_name):
            return None, tool_result_json
        obs_type = _TOOL_OBSERVATION_TYPE.get(tool_name)
        if obs_type is None:
            return None, tool_result_json
        try:
            payload = json.loads(tool_result_json)
        except (TypeError, json.JSONDecodeError):
            return None, tool_result_json
        if not isinstance(payload, dict) or not payload.get("ok"):
            return None, tool_result_json
        # pattern_search splits by exhaustive flag — non-exhaustive runs
        # are PAGE_HITS_PARTIAL (no auto-claim) so the LLM can still
        # cite them in challenges without certifying a scan.
        if tool_name == "pattern_search" and payload.get("exhaustive") is False:
            obs_type = ObservationType.PAGE_HITS_PARTIAL
        citations = self._collect_citations(payload)
        observation = self.plant.record_observation(
            tool_name=tool_name,
            observation_type=obs_type,
            payload=payload,
            citations=citations,
        )
        state_changed = False
        auto_extracted_ids: List[str] = []
        if obs_type == ObservationType.PAGE_HITS_EXHAUSTIVE:
            extracted = self.plant.auto_extract_and_ingest(observation)
            if extracted:
                self.plant.reconcile()
                state_changed = True
                auto_extracted_ids = [c.id for c in extracted]
        # Augment the result with observation_id (always), the auto-
        # extracted claim ids (so the LLM can pass them to
        # answer_finalize without guessing), and the gate.diagnose
        # snapshot (when state changed).
        payload["observation_id"] = observation.id
        if auto_extracted_ids:
            payload["auto_extract_claim_ids"] = auto_extracted_ids
        if state_changed:
            from agentic.tools.proof._common import _gate_to_dict
            payload["gate"] = _gate_to_dict(self.plant.gate_view())
        return observation.id, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _collect_citations(payload: Dict[str, Any]) -> List[Citation]:
        """Best-effort citation collection from acquisition tool payloads.

        Supports the citation shapes emitted by the eight acquisition
        tools currently in the registry: pattern_search's ``citations``
        list, semantic/bm25/graph hits with ``page_global_id``, and
        read_page's ``results`` (one per page).
        """
        out: List[Citation] = []
        for cite in payload.get("citations") or []:
            if isinstance(cite, dict):
                gid = cite.get("page_global_id") or ""
                if "/" in gid:
                    fid, _, pid = gid.partition("/")
                    out.append(Citation(
                        file_id=fid, page_id=pid,
                        span=cite.get("matched_text") or cite.get("snippet"),
                    ))
                else:
                    fid = cite.get("file_id") or ""
                    pid = cite.get("page_id") or ""
                    if fid and pid:
                        out.append(Citation(
                            file_id=str(fid), page_id=str(pid),
                            span=cite.get("matched_text") or cite.get("snippet"),
                        ))
        for hit in payload.get("hits") or []:
            if not isinstance(hit, dict):
                continue
            gid = hit.get("page_global_id") or hit.get("page_id") or ""
            if "/" in gid:
                fid, _, pid = gid.partition("/")
                out.append(Citation(file_id=fid, page_id=pid, span=hit.get("snippet")))
        # read_page packages multi-page results under ``results`` and
        # historically uses ``global_id`` (not ``page_global_id``); accept
        # both for compatibility with the current tool output and any
        # future renames.
        for entry in payload.get("results") or []:
            if not isinstance(entry, dict):
                continue
            gid = entry.get("page_global_id") or entry.get("global_id") or ""
            if "/" in gid:
                fid, _, pid = gid.partition("/")
                out.append(Citation(file_id=fid, page_id=pid))
        gid = payload.get("page_global_id") or payload.get("global_id")
        if isinstance(gid, str) and "/" in gid:
            fid, _, pid = gid.partition("/")
            out.append(Citation(file_id=fid, page_id=pid))
        return out

    def _stall_detected(
        self,
        history: List[Tuple[int, int, int]],
    ) -> bool:
        """T³-style stall test.

        ``history`` carries ``(open_count, claims_accepted, challenges_accepted)``
        snapshots after each loop. If for ``stall_window`` consecutive
        snapshots the open_count did not drop AND no new claim was
        accepted AND no new challenge was accepted, treat the run as
        stuck.

        Both counters are monotonic (accepted-ever) rather than pending
        — otherwise a challenge accepted-and-discharged within the same
        loop would be invisible to the stall detector.
        """
        if len(history) < self.stall_window:
            return False
        recent = history[-self.stall_window :]
        baseline_open = recent[0][0]
        for open_count, _, _ in recent:
            if open_count < baseline_open:
                return False
        baseline_claims = recent[0][1]
        baseline_challenges = recent[0][2]
        if any(c > baseline_claims for _, c, _ in recent[1:]):
            return False
        if any(ch > baseline_challenges for _, _, ch in recent[1:]):
            return False
        return True

    def _just_nudged_no_obligation(self, messages: List[Dict[str, Any]]) -> bool:
        """Has the most recent `user` message already been the
        no-obligation nudge? Avoid re-nudging on every loop until the
        LLM either responds or stalls out."""
        for m in reversed(messages):
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            return isinstance(content, str) and "the proof gate has zero obligations on file" in content
        return False

    def _state_snapshot(self) -> Tuple[int, int, int]:
        open_count = len(self.plant.obligations.active_required_open())
        claims_accepted = len(self.plant.evidence.claims())
        # ChallengeStore counts every inserted challenge regardless of
        # later discharge; per-loop monotonicity is what matters here.
        challenges_accepted = len(self.plant.challenges.all())
        return open_count, claims_accepted, challenges_accepted

    # ------------------------------------------------------------- run

    def run(
        self,
        query: str,
        tracer: Optional[Tracer] = None,
    ) -> Dict[str, Any]:
        session = tracer.session(query) if tracer is not None else None
        context = AgentContext()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        trajectory: List[Dict[str, Any]] = []
        total_cost = 0.0
        cached_tokens_total = 0
        input_tokens_total = 0
        output_tokens_total = 0
        loop_count = 0
        tool_schemas = self.tools.get_all_schemas()
        state_history: List[Tuple[int, int, int]] = []
        decision: Optional[str] = None
        final_answer: str = ""

        if session is not None:
            session.daily(
                "setup",
                {
                    "model": self.llm.model,
                    "system_prompt": self.system_prompt,
                    "tool_schemas": tool_schemas,
                    "max_loops": self.max_loops,
                    "max_token_budget": self.max_token_budget,
                    "stall_window": self.stall_window,
                    "agent_class": "ProofAgent",
                },
            )

        if self.verbose:
            print(f"\n{'=' * 60}\nQuestion: {query}\n{'=' * 60}\n")

        early_exit_reason: Optional[str] = None
        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1

            if self._budget_exhausted(messages, loop_idx):
                final_answer, total_cost, decision = self._abstain(
                    messages, total_cost, "budget_exhausted", session=session
                )
                early_exit_reason = "budget_exhausted"
                break

            if self.verbose:
                tokens = self._calculate_message_tokens(messages)
                print(
                    f"Loop {loop_count}/{self.max_loops} (Tokens: {tokens}/{self.max_token_budget})"
                )

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                logger.exception("LLM error in ProofAgent loop %d", loop_count)
                if session is not None:
                    session.event(
                        "llm_calls",
                        {"stage": "loop", "loop": loop_count, "error": f"{type(e).__name__}: {e}"},
                    )
                early_exit_reason = "llm_error"
                final_answer, total_cost, decision = self._abstain(
                    messages, total_cost, "llm_error", session=session
                )
                break

            total_cost += response["cost"]
            input_tokens_total += response.get("input_tokens", 0)
            cached_tokens_total += response.get("cached_tokens", 0)
            output_tokens_total += response.get("output_tokens", 0)
            message = response["message"]
            messages.append(message)

            if session is not None:
                session.event(
                    "llm_calls",
                    _llm_call_record(stage="loop", loop=loop_count, response=response),
                )

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                # No-op turn — nudge the LLM toward a tool. The proof
                # gate does NOT accept natural-final.
                if self.verbose:
                    print("Assistant emitted no tool_calls; nudging.")
                messages.append({
                    "role": "user",
                    "content": (
                        "You did not call a tool. The proof gate requires you "
                        "to drive every step through a registered tool. To "
                        "deliver the final answer, call `answer_finalize`."
                    ),
                })
                continue

            for tc_idx, tc in enumerate(tool_calls):
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                except Exception as e:
                    tool_result = err(
                        "tool_dispatch_exception",
                        f"{type(e).__name__}: {e}",
                        tool=func_name,
                    )
                    tool_log = {"retrieved_tokens": 0, "error": "tool_dispatch_exception"}

                # Acquisition-tool successful results become Observations
                # in the plant; this is also where auto-extraction fires.
                # The augmented result carries observation_id and (when
                # state changed) the post-call gate.diagnose snapshot,
                # so the LLM can cite the observation in challenges and
                # see proof-state movement in the same tool_result.
                _, tool_result = self._maybe_register_observation(func_name, tool_result)

                # If this was answer_finalize CERTIFIED, capture the
                # final answer and exit the loop early.
                if func_name == "answer_finalize":
                    decision_payload = self._inspect_finalize_result(tool_result)
                    if decision_payload is not None:
                        decision = decision_payload.get("decision")
                        if decision == "CERTIFIED":
                            final_answer = decision_payload.get("final_answer") or ""
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_result,
                            })
                            turn_record = self._turn_record(
                                loop_count, tc_idx, func_name, func_args, tool_result, tool_log
                            )
                            trajectory.append(turn_record)
                            if session is not None:
                                session.event("trajectory", turn_record)
                            early_exit_reason = "certified"
                            break
                        if decision == "ABSTAIN":
                            final_answer = self._compose_abstain_text(tool_result)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": tool_result,
                            })
                            turn_record = self._turn_record(
                                loop_count, tc_idx, func_name, func_args, tool_result, tool_log
                            )
                            trajectory.append(turn_record)
                            if session is not None:
                                session.event("trajectory", turn_record)
                            early_exit_reason = "abstained_at_finalize"
                            break

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
                turn_record = self._turn_record(
                    loop_count, tc_idx, func_name, func_args, tool_result, tool_log
                )
                trajectory.append(turn_record)
                if session is not None:
                    session.event("trajectory", turn_record)

            if early_exit_reason in ("certified", "abstained_at_finalize"):
                break

            # T³ stall detection — track end-of-loop state.
            state_history.append(self._state_snapshot())

            # Early-phase nudge: if the LLM has spent several loops on
            # acquisition without creating any obligation, the gate
            # cannot certify anything. Force a user message that names
            # the missing step before stall fires; "no obligation yet"
            # is the single most common stall cause and is fully
            # recoverable with one prompt.
            if (
                self.plant.obligations.root() is None
                and loop_count >= 2
                and not self._just_nudged_no_obligation(messages)
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "Reminder: the proof gate has zero obligations on file. "
                        "Until you call `obligation_create` to declare the typed "
                        "obligation for this question, no answer can be CERTIFIED. "
                        "Pick the kind that matches the question shape (exists / "
                        "count / set / forall / negation / argmax), set scope to "
                        "the file_ids and (optional) section_ids the answer lives "
                        "in, and a registered predicate. Then continue acquiring "
                        "evidence. Do this now."
                    ),
                })
                continue

            if self._stall_detected(state_history):
                if self.verbose:
                    print(f"T³ stall detected at loop {loop_count}")
                final_answer, total_cost, decision = self._abstain(
                    messages, total_cost, "stall_detected", session=session
                )
                early_exit_reason = "stall_detected"
                break

        # If we ran out of loops without certifying, fall back to ABSTAIN.
        if early_exit_reason is None:
            final_answer, total_cost, decision = self._abstain(
                messages, total_cost, "max_loops_exceeded", session=session
            )
            early_exit_reason = "max_loops_exceeded"

        result: Dict[str, Any] = {
            "answer": final_answer,
            "decision": decision,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "input_tokens_total": input_tokens_total,
            "cached_tokens_total": cached_tokens_total,
            "output_tokens_total": output_tokens_total,
            "loops": loop_count,
            "exit_reason": early_exit_reason,
            "obligations": [
                {
                    "id": o.id,
                    "kind": o.spec.kind.value,
                    "status": o.status.value,
                    "is_root": o.is_root,
                    "closed_value": o.closed_value,
                    "closed_by": list(o.closed_by),
                }
                for o in self.plant.obligations.all_obligations()
            ],
            "claims": [
                {
                    "id": c.id,
                    "claim_type": c.claim_type.value,
                    "unit_type": c.unit_type,
                    "derivation": c.derivation,
                    "positive_count": len(c.positive_units),
                    "negative_count": len(c.negative_units),
                }
                for c in self.plant.evidence.claims()
            ],
            **context.get_summary(),
        }

        if session is not None:
            session.finalize(
                answer=final_answer,
                summary={
                    "loops": loop_count,
                    "exit_reason": early_exit_reason,
                    "decision": decision,
                    "total_cost": total_cost,
                    "input_tokens_total": input_tokens_total,
                    "cached_tokens_total": cached_tokens_total,
                    "output_tokens_total": output_tokens_total,
                    "context_summary": context.get_summary(),
                    "obligation_count": len(self.plant.obligations.all_obligations()),
                    "claim_count": len(self.plant.evidence.claims()),
                },
            )
        return result

    # ------------------------------------------------------------- internal

    def _abstain(
        self,
        messages: List[Dict[str, Any]],
        total_cost: float,
        reason: str,
        session: Optional[TraceSession] = None,
    ) -> Tuple[str, float, str]:
        """Force ABSTAIN through the plant so the gate diagnostics are the
        ones the LLM-facing output reflects."""
        result = self.plant.handle_answer_finalize(
            draft_text="(abstaining)",
            cited_claim_ids=[],
            budget_exhausted=True,
        )
        # If the plant's gate happens to be CERTIFIED (e.g., everything closed
        # but the model never called answer_finalize), prefer CERTIFIED.
        if result.ok and result.payload.get("decision") == "CERTIFIED":
            final_answer = result.payload.get("final_answer") or ""
            return final_answer, total_cost, "CERTIFIED"
        # Otherwise compose an ABSTAIN narrative from the gate view.
        diagnostics = self.plant.gate_view()
        final_answer = self._format_abstain_text(reason, diagnostics)
        if session is not None:
            session.event(
                "abstain",
                {
                    "reason": reason,
                    "open_obligations": diagnostics.open_obligations,
                    "challenged_obligations": diagnostics.challenged_obligations,
                },
            )
        return final_answer, total_cost, "ABSTAIN"

    @staticmethod
    def _format_abstain_text(reason: str, gate: GateView) -> str:
        lines = [
            "I cannot certify an answer to this question under the proof gate.",
            f"Reason: {reason}.",
        ]
        if gate.open_obligations:
            lines.append("Open obligations:")
            for o in gate.open_obligations:
                lines.append(f"  - {o['id']} ({o['kind']}) failure={o['failure_kind']}")
        if gate.challenged_obligations:
            lines.append("Challenged obligations:")
            for o in gate.challenged_obligations:
                lines.append(
                    f"  - {o['obligation_id']} repair={o['repair_kind']} reason={o.get('reason','')}"
                )
        if gate.abstain_reason:
            lines.append(f"Gate abstain reason: {gate.abstain_reason}")
        return "\n".join(lines)

    @staticmethod
    def _inspect_finalize_result(tool_result: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(tool_result)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _compose_abstain_text(self, finalize_tool_result: str) -> str:
        payload = self._inspect_finalize_result(finalize_tool_result) or {}
        gate = payload.get("gate") or {}
        return self._format_abstain_text(
            "answer_finalize_returned_abstain",
            GateView(
                open_obligations=gate.get("open_obligations", []),
                closed_obligations=gate.get("closed_obligations", []),
                challenged_obligations=gate.get("challenged_obligations", []),
                diagnostics=[],
                recent_claims=gate.get("recent_claims", []),
                abstain_recommended=gate.get("abstain_recommended", True),
                abstain_reason=gate.get("abstain_reason"),
            ),
        )

    @staticmethod
    def _turn_record(
        loop_count: int,
        tc_idx: int,
        func_name: str,
        func_args: Dict[str, Any],
        tool_result: str,
        tool_log: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "loop": loop_count,
            "tool_call_index": tc_idx,
            "tool_name": func_name,
            "arguments": func_args,
            "tool_result": tool_result,
            **tool_log,
        }


def _llm_call_record(
    *,
    stage: str,
    response: Dict[str, Any],
    loop: Optional[int] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    msg = response.get("message") or {}
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    raw = response.get("raw_response") or {}
    finish_reason = ""
    try:
        finish_reason = raw.get("choices", [{}])[0].get("finish_reason", "")
    except Exception:
        pass
    record: Dict[str, Any] = {
        "stage": stage,
        "loop": loop,
        "input_tokens": response.get("input_tokens", 0),
        "cached_tokens": response.get("cached_tokens", 0),
        "output_tokens": response.get("output_tokens", 0),
        "cost": response.get("cost", 0.0),
        "finish_reason": finish_reason,
        "content_chars": len(content),
        "content_preview": content[:400],
        "tool_calls": [
            {"name": tc.get("function", {}).get("name"),
             "arguments": tc.get("function", {}).get("arguments", "")}
            for tc in tool_calls
        ],
    }
    if reason is not None:
        record["reason"] = reason
    return record
