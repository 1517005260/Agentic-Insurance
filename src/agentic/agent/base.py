"""Tool-calling agent loop with budget guards and pre-warm.

The loop alternates ``chat -> tool calls -> tool results`` until the
model emits a message without ``tool_calls`` (final answer) or one of
two budgets fires:

* token budget — the running message-stack token count exceeds
  ``max_token_budget``, at which point the model is asked to answer with
  what it has;
* loop budget — ``max_loops`` iterations elapsed.

Budget exhaustion always produces a final-answer attempt rather than
raising, so callers always get a result + trajectory even on stuck
runs.

Pre-warm:
  Some tools (notably ``graph_explore`` via GLiNER NER) absorb a 10–15 s
  one-time load on first use. :meth:`warm_up` walks every registered
  tool's optional ``warm_up()`` hook before the first user turn, so
  per-query latency stays comparable across runs.
"""

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from config.shared import shared_tiktoken_encoder

from agentic.agent.prompts import SYSTEM_PROMPT
from agentic.core.context import AgentContext
from agentic.tools.acquisition._common import err
from agentic.tools.registry import ToolRegistry
from model_client import LLMClient
from tracer import TraceSession, Tracer


logger = logging.getLogger(__name__)


class BaseAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = None,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose
        # Process-cached encoder — one tiktoken instance is ~50 MB.
        self.tokenizer = shared_tiktoken_encoder("gpt-4o")

    def warm_up(self) -> Dict[str, float]:
        """Invoke each tool's optional ``warm_up()`` hook.

        Returns ``{tool_name: elapsed_seconds}`` (negative on failure)
        for visibility — call sites typically log or print the dict
        once at startup so cold-start cost is observable.
        """
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
                logger.warning("BaseAgent.warm_up: %s failed: %s", name, exc)
                timings[name] = -1.0
        return timings

    def _calculate_message_tokens(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> int:
        # The system prompt is intentionally double-counted: it is seeded
        # here AND walked again as messages[0] below. The token-budget
        # exhaustion thresholds calibrated against this behavior, so the
        # double-count is part of the contract. ``is not None`` (rather
        # than ``or``) lets a caller override to an empty string.
        prompt = system_prompt if system_prompt is not None else self.system_prompt
        total = len(self.tokenizer.encode(prompt))
        for msg in messages:
            content = msg.get("content", "")
            if content:
                total += len(self.tokenizer.encode(str(content)))
        return total

    def _force_final_answer(
        self,
        messages: List[Dict[str, Any]],
        context: AgentContext,
        total_cost: float,
        reason: str,
        session: Optional[TraceSession] = None,
    ) -> tuple:
        # Force-final must succeed even when the in-loop chat call has
        # been failing (vLLM input-cap rejections, model context overflow,
        # etc).  Two safety nets:
        #
        # 1. Trim the transcript before the final call. Keep system +
        #    original user query + the most recent tool round(s); drop
        #    middle iterations whose evidence has already been
        #    superseded.  This is the single biggest reliability fix —
        #    the loop accumulates 25-200k tokens and the final call
        #    needs to fit in the model context.
        # 2. Use a tighter ``max_tokens`` for the final response so the
        #    server has more input room.
        trim_stats: Dict[str, int] = {}
        trimmed = self._trim_messages_for_final(messages, trim_stats=trim_stats)
        force_prompt = (
            "You have reached the budget limit. Stop calling tools.\n\n"
            "Based ONLY on the tool results above, give your final answer "
            "in 1-2 sentences (or a single named entity / number for "
            "factoid questions). Quote the most specific span verbatim "
            "where possible.\n\n"
            "After your answer, output one final line exactly:\n"
            "ANSWER: <shortest exact answer span verbatim, no extra words>\n\n"
            "If the gathered evidence does not contain the answer, "
            "write ANSWER: unanswerable."
        )
        trimmed.append({"role": "user", "content": force_prompt})

        try:
            response = self.llm.chat(
                messages=trimmed, tools=None, temperature=0.0,
                # Tight cap — final answer is short and we need every
                # token of input room we can get.
                max_tokens=1024,
            )
            total_cost += response["cost"]
            final_answer = response["message"].get("content", "")
            if session is not None:
                rec = _llm_call_record(stage="force_final", response=response, reason=reason)
                if trim_stats:
                    rec.update(trim_stats)
                session.event("llm_calls", rec)
            if self.verbose:
                print(f"Forced answer: {final_answer[:200]}...")
                print(f"Total cost: ${total_cost:.6f}")
                if trim_stats:
                    print(f"  trim stats: {trim_stats}")
        except Exception as e:
            if self.verbose:
                print(f"Error getting forced answer: {e}")
            final_answer = f"Error: {reason} and failed to generate final answer."
            if session is not None:
                rec = {"stage": "force_final", "error": f"{type(e).__name__}: {e}", "reason": reason}
                if trim_stats:
                    rec.update(trim_stats)
                session.event("llm_calls", rec)

        return final_answer, total_cost

    def _trim_messages_for_final(
        self,
        messages: List[Dict[str, Any]],
        *,
        target_tokens: int = 18_000,
        per_msg_cap_chars: int = 4_000,
        trim_stats: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Drop middle iterations so the final-call prompt fits in
        context.  Returns a list that is always a VALID chat history
        for OpenAI-style tool-calling servers — no orphan tool reply
        (a ``role=tool`` message whose tool_call_id has no matching
        assistant tool_calls) and no orphan tool_calls (an assistant
        message with tool_calls but missing some of the tool replies
        it referenced).

        Strategy:
          1. Group ``tail = messages[2:]`` into atomic
             ``[assistant-with-tool_calls, role=tool, role=tool, ...]``
             blocks.  Each block is dropped or kept WHOLE.
          2. Drop oldest blocks first until under ``target_tokens``;
             always keep the most recent block so the model still
             sees evidence to answer from.
          3. If still over budget after dropping all but the last
             block, truncate that block's tool-message contents to
             ``per_msg_cap_chars`` so a single huge tool result
             doesn't blow the model's input limit.
        """
        if len(messages) <= 4:
            return list(messages)
        head = messages[:2]
        sys_prompt = head[0].get("content", "") if head else ""

        # Group tail into atomic blocks. A block starts at an assistant
        # message that has tool_calls (or any assistant/user message
        # without tool_calls = standalone block).
        tail = messages[2:]
        blocks: List[List[Dict[str, Any]]] = []
        i = 0
        while i < len(tail):
            msg = tail[i]
            role = msg.get("role")
            tcs = msg.get("tool_calls") or []
            if role == "assistant" and tcs:
                # Pull in all tool replies for these tool_calls.
                expected = {tc.get("id") for tc in tcs if tc.get("id")}
                block = [msg]
                j = i + 1
                while j < len(tail) and tail[j].get("role") == "tool":
                    block.append(tail[j])
                    expected.discard(tail[j].get("tool_call_id"))
                    j += 1
                blocks.append(block)
                i = j
            else:
                blocks.append([msg])
                i += 1

        # Drop oldest blocks until budget fits; keep at least one.
        current = self._calculate_message_tokens(
            head + [m for blk in blocks for m in blk], system_prompt=sys_prompt
        )
        blocks_dropped = 0
        while current > target_tokens and len(blocks) > 1:
            blocks.pop(0)
            blocks_dropped += 1
            current = self._calculate_message_tokens(
                head + [m for blk in blocks for m in blk], system_prompt=sys_prompt
            )

        # Last-resort: still over budget? Cap each tool-message content
        # in the surviving blocks to ``per_msg_cap_chars``. Assistant
        # messages stay intact (they carry the reasoning + tool_call
        # ids the server expects to validate).  Build fresh blocks so
        # the original ``messages`` is never mutated — an in-place swap
        # risks overwriting an assistant message with a capped tool
        # reply when a block has interleaved roles.
        tool_chars_capped = 0
        if current > target_tokens:
            capped_blocks: List[List[Dict[str, Any]]] = []
            for blk in blocks:
                new_blk: List[Dict[str, Any]] = []
                for msg in blk:
                    if msg.get("role") != "tool":
                        new_blk.append(msg)
                        continue
                    c = msg.get("content") or ""
                    if isinstance(c, str) and len(c) > per_msg_cap_chars:
                        new_msg = dict(msg)
                        new_msg["content"] = (
                            c[:per_msg_cap_chars]
                            + "\n…[truncated for final-answer call]"
                        )
                        tool_chars_capped += len(c) - per_msg_cap_chars
                        new_blk.append(new_msg)
                    else:
                        new_blk.append(msg)
                capped_blocks.append(new_blk)
            blocks = capped_blocks

        # Drop standalone ``role=tool`` blocks (orphan replies whose
        # parent assistant message was trimmed): they would fail
        # validation on most OpenAI-compatible servers.
        orphan_tool_dropped = sum(
            1 for blk in blocks
            if len(blk) == 1 and blk[0].get("role") == "tool"
        )
        cleaned = [
            blk for blk in blocks
            if not (len(blk) == 1 and blk[0].get("role") == "tool")
        ]
        if trim_stats is not None:
            trim_stats["force_final_trim_blocks_dropped"] = blocks_dropped
            trim_stats["force_final_tool_chars_capped"] = tool_chars_capped
            trim_stats["force_final_orphan_tool_dropped"] = orphan_tool_dropped
        return head + [m for blk in cleaned for m in blk]

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
    ) -> Dict[str, Any]:
        """Run the tool-calling loop.

        ``on_event`` (optional) is a sync callback invoked at every
        observable boundary so a streaming consumer (e.g. SSE runner)
        can mirror progress. Signature ``(event_name: str, data: dict)``.
        Default ``None`` skips callbacks — useful for experiment scripts
        and the CLI.

        ``max_loops`` / ``max_token_budget`` / ``system_prompt`` (all
        optional) override the constructor values for this run only;
        ``None`` falls back to the per-instance defaults.

        ``cancel_check`` (optional) is a no-arg predicate polled at
        each loop boundary; when it returns True the loop exits with
        ``exit_reason='client_disconnect'``. The runner side wires it
        to ``EventBus.is_closed`` so a TCP disconnect stops the agent
        from spending more LLM tokens. Default ``None`` disables the
        check (always-False).

        Emitted events:
        * ``status`` — phase transitions (``thinking`` per loop,
          ``force_final`` on budget hit).
        * ``tool_call`` / ``tool_result`` — one per tool execution.
        * ``final`` — summary frame at the end (always emitted before
          the runner closes the stream).
        """
        effective_max_loops = max_loops if max_loops is not None else self.max_loops
        effective_max_tokens = (
            max_token_budget if max_token_budget is not None else self.max_token_budget
        )
        effective_system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        emit = _make_emitter(on_event)
        session = tracer.session(query) if tracer is not None else None
        context = AgentContext()
        messages = [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": query},
        ]

        trajectory: List[Dict[str, Any]] = []
        total_cost = 0.0
        cached_tokens_total = 0
        input_tokens_total = 0
        output_tokens_total = 0
        loop_count = 0
        # Tool-validation-error iterations are "free" — they don't count
        # against the loop budget so the agent can self-correct (e.g.
        # retry a read tool that rejected bare file_ids). Hard-capped so
        # a persistent bad-arg pattern can't infinite-loop the agent.
        MAX_VALIDATION_FREEBIES = 5
        validation_error_freebies = 0
        tool_schemas = self.tools.get_all_schemas()

        if session is not None:
            # The (model, system_prompt, tool_schemas, budgets) bundle
            # is identical across every run on a given day, so we
            # write it at the day level. Per-run query.json gets
            # stamped with a content hash so a postmortem can find
            # the exact setup that applied to this run even if the
            # prompt changed mid-day.
            session.daily(
                "setup",
                {
                    "model": self.llm.model,
                    "system_prompt": effective_system_prompt,
                    "tool_schemas": tool_schemas,
                    "max_loops": effective_max_loops,
                    "max_token_budget": effective_max_tokens,
                },
            )

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Question: {query}")
            print(f"{'=' * 60}\n")

        early_exit_reason: Optional[str] = None
        final_answer: str = ""

        # While-loop (was for-loop) so tool-validation-error iterations
        # can be "free" — loop_count only advances when the iteration
        # actually used the LLM productively. raw_iter is the hard
        # outer cap (effective_max_loops + MAX_VALIDATION_FREEBIES).
        for raw_iter in range(effective_max_loops + MAX_VALIDATION_FREEBIES):
            if loop_count >= effective_max_loops:
                break  # exhausted the real loop budget
            loop_count = loop_count + 1
            # Per-iteration tool-log buffer: filled inside the tool_calls
            # loop below; used at the end to decide whether to refund
            # this loop_count (validation-error iterations only).
            iter_tool_logs: List[Dict[str, Any]] = []

            # Cancellation gate. Polled at the loop boundary so the
            # in-flight tool-call / LLM round-trip is allowed to finish
            # — interrupting a tool mid-execute would leave the agent's
            # message history desynced from any side effects. The
            # runner side wires this to ``EventBus.is_closed`` so a
            # client disconnect stops the agent from spinning more
            # loops.
            if cancel_check is not None and cancel_check():
                emit("status", {"phase": "client_disconnect"})
                early_exit_reason = "client_disconnect"
                break

            current_tokens = self._calculate_message_tokens(
                messages, system_prompt=effective_system_prompt
            )
            if current_tokens > effective_max_tokens:
                if self.verbose:
                    print(
                        f"Token budget exceeded ({current_tokens} > {effective_max_tokens}), "
                        f"forcing answer..."
                    )
                emit("status", {"phase": "force_final", "reason": "token_budget_exceeded"})
                final_answer, total_cost = self._force_final_answer(
                    messages, context, total_cost, "Token budget exceeded", session=session
                )
                early_exit_reason = "token_budget_exceeded"
                break

            if self.verbose:
                print(
                    f"Loop {loop_count}/{effective_max_loops} "
                    f"(Tokens: {current_tokens}/{effective_max_tokens})"
                )
            emit(
                "status",
                {
                    "phase": "thinking",
                    "loop": loop_count,
                    "max_loops": effective_max_loops,
                    "tokens_used": current_tokens,
                },
            )

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                # The vLLM input cap (max_model_len − max_tokens) is the
                # most common cause: a long tool result pushes the next
                # request past the cap and the server 400s. Rather than
                # returning empty (the historical bug — 263/1950 v6
                # empties were from this path), fall through to
                # ``_force_final_answer`` which trims the transcript and
                # asks the model to answer from what's already in
                # context. The original exception is preserved in the
                # session log so the runner can still inspect it.
                if self.verbose:
                    print(f"LLM error: {e}; forcing answer with truncated context...")
                if session is not None:
                    session.event(
                        "llm_calls",
                        {"stage": "loop", "loop": loop_count, "error": f"{type(e).__name__}: {e}"},
                    )
                emit(
                    "status",
                    {"phase": "force_final", "reason": "llm_error", "error": str(e)[:200]},
                )
                final_answer, total_cost = self._force_final_answer(
                    messages, context, total_cost,
                    f"LLM error: {type(e).__name__}; answer from context already gathered",
                    session=session,
                )
                early_exit_reason = "llm_error"
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

            if self.verbose and message.get("content"):
                print(f"Assistant: {message['content'][:200]}...")

            tool_calls = message.get("tool_calls")
            # Emit ``content`` as a "thought" only when tool_calls are
            # also present — that's the LLM's reasoning before invoking
            # a tool. Without tool_calls, ``content`` is the final
            # answer and will reach the frontend via the ``final`` event
            # / ``answer`` field; routing it to the timeline too would
            # duplicate the text in the UI.
            content_str = (message.get("content") or "").strip()
            if content_str and tool_calls:
                emit(
                    "thought",
                    {"loop": loop_count, "text": content_str},
                )

            if not tool_calls:
                final_answer = message.get("content", "")
                early_exit_reason = "natural"
                break

            for tc_idx, tc in enumerate(tool_calls):
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                if self.verbose:
                    print(f"Tool: {func_name}")
                    print(f"  Args: {func_args}")

                emit(
                    "tool_call",
                    {"loop": loop_count, "name": func_name, "args": func_args},
                )

                try:
                    tool_result, tool_log = self.tools.execute(func_name, context, **func_args)
                except Exception as e:
                    tool_result = err(
                        "tool_dispatch_exception",
                        f"{type(e).__name__}: {e}",
                        tool=func_name,
                    )
                    tool_log = {"retrieved_tokens": 0, "error": "tool_dispatch_exception"}

                if self.verbose:
                    output_preview = (
                        tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                    )
                    print(f"  Result: {output_preview}")
                    if tool_log.get("retrieved_tokens", 0) > 0:
                        print(f"  Tokens: {tool_log['retrieved_tokens']}")
                    print()

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    }
                )

                turn_record = {
                    "loop": loop_count,
                    "tool_call_index": tc_idx,
                    "tool_name": func_name,
                    "arguments": func_args,
                    "tool_result": tool_result,
                    **tool_log,
                }
                trajectory.append(turn_record)
                iter_tool_logs.append(tool_log)
                if session is not None:
                    session.event("trajectory", turn_record)
                emit(
                    "tool_result",
                    {
                        "loop": loop_count,
                        "name": func_name,
                        "preview": tool_result[:300],
                        "retrieved_tokens": tool_log.get("retrieved_tokens", 0),
                        "error": tool_log.get("error"),
                        # Full tool result envelope for runner-side
                        # consumers (e.g. citation extraction). The
                        # underscore prefix flags it as internal: the
                        # streaming runner strips it before pushing the
                        # frame to the SSE bus, since the JSON can be
                        # tens of KB and the SSE client only needs the
                        # 300-char preview.
                        "_full_result": tool_result,
                    },
                )

            # End-of-iteration: refund this loop if every tool call
            # failed input validation (the LLM gets a free retry to
            # self-correct based on the error message in the tool
            # result). Capped by MAX_VALIDATION_FREEBIES.
            if (
                iter_tool_logs
                and validation_error_freebies < MAX_VALIDATION_FREEBIES
                and all(t.get("error") == "invalid_argument" for t in iter_tool_logs)
            ):
                validation_error_freebies += 1
                loop_count -= 1
                if self.verbose:
                    print(
                        f"  [refund] all tools returned invalid_argument; "
                        f"loop refunded (freebies used: "
                        f"{validation_error_freebies}/{MAX_VALIDATION_FREEBIES})"
                    )

        # Loop exited without natural break or early exit ⇒ max loops hit.
        if early_exit_reason is None:
            if self.verbose:
                print(f"Max loops reached ({effective_max_loops}), forcing answer...")
            emit("status", {"phase": "force_final", "reason": "max_loops_exceeded"})
            final_answer, total_cost = self._force_final_answer(
                messages, context, total_cost, "Maximum loops exceeded", session=session
            )
            early_exit_reason = "max_loops_exceeded"

        result: Dict[str, Any] = {
            "answer": final_answer,
            "trajectory": trajectory,
            "total_cost": total_cost,
            "input_tokens_total": input_tokens_total,
            "cached_tokens_total": cached_tokens_total,
            "output_tokens_total": output_tokens_total,
            "loops": loop_count,
            "exit_reason": early_exit_reason,
            **context.get_summary(),
        }
        if early_exit_reason == "token_budget_exceeded":
            result["token_budget_exceeded"] = True
        if early_exit_reason == "max_loops_exceeded":
            result["max_loops_exceeded"] = True

        if session is not None:
            session.finalize(
                answer=final_answer,
                summary={
                    "loops": loop_count,
                    "exit_reason": early_exit_reason,
                    "total_cost": total_cost,
                    "input_tokens_total": input_tokens_total,
                    "cached_tokens_total": cached_tokens_total,
                    "output_tokens_total": output_tokens_total,
                    "cache_hit_rate": (
                        round(cached_tokens_total / input_tokens_total, 4)
                        if input_tokens_total
                        else 0.0
                    ),
                    "context_summary": context.get_summary(),
                },
            )

        emit(
            "final",
            {
                "answer": final_answer,
                "exit_reason": early_exit_reason,
                "loops": loop_count,
                "total_cost": total_cost,
                "input_tokens_total": input_tokens_total,
                "cached_tokens_total": cached_tokens_total,
                "output_tokens_total": output_tokens_total,
            },
        )

        return result



def _make_emitter(
    on_event: Optional[Callable[[str, Dict[str, Any]], None]],
) -> Callable[[str, Dict[str, Any]], None]:
    """Wrap an optional callback so call sites can fire events unconditionally.

    A callback exception must NOT poison the agent loop — the consumer
    side might disconnect mid-run. We log and swallow.
    """
    if on_event is None:
        def _noop(_event: str, _data: Dict[str, Any]) -> None:
            return
        return _noop

    def _emit(event: str, data: Dict[str, Any]) -> None:
        try:
            on_event(event, data)
        except Exception:
            logger.exception("on_event callback failed for %s", event)

    return _emit


def _llm_call_record(
    *,
    stage: str,
    response: Dict[str, Any],
    loop: Optional[int] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the JSON-serializable record for one LLM round-trip.

    Pure serializer — does not touch the session. Caller decides where
    the record lands (typically ``session.event("llm_calls", record)``).
    """
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
