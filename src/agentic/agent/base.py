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
  Some tools (notably ``graph_explore`` via spaCy NER) absorb a 10–15 s
  one-time load on first use. :meth:`warm_up` walks every registered
  tool's optional ``warm_up()`` hook before the first user turn, so
  per-query latency stays comparable across runs.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import tiktoken

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
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

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

    def _calculate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = len(self.tokenizer.encode(self.system_prompt))
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
        force_prompt = (
            "You have reached the limit. "
            "You MUST now provide a final answer based on the information you have gathered so far. "
            "Do NOT call any more tools. Synthesize the available information and respond directly."
        )
        messages.append({"role": "user", "content": force_prompt})

        try:
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
            total_cost += response["cost"]
            final_answer = response["message"].get("content", "")
            if session is not None:
                session.event(
                    "llm_calls",
                    _llm_call_record(stage="force_final", response=response, reason=reason),
                )
            if self.verbose:
                print(f"Forced answer: {final_answer[:200]}...")
                print(f"Total cost: ${total_cost:.6f}")
        except Exception as e:
            if self.verbose:
                print(f"Error getting forced answer: {e}")
            final_answer = f"Error: {reason} and failed to generate final answer."
            if session is not None:
                session.event(
                    "llm_calls",
                    {"stage": "force_final", "error": f"{type(e).__name__}: {e}", "reason": reason},
                )

        return final_answer, total_cost

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
                    "system_prompt": self.system_prompt,
                    "tool_schemas": tool_schemas,
                    "max_loops": self.max_loops,
                    "max_token_budget": self.max_token_budget,
                },
            )

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Question: {query}")
            print(f"{'=' * 60}\n")

        early_exit_reason: Optional[str] = None
        final_answer: str = ""

        for loop_idx in range(self.max_loops):
            loop_count = loop_idx + 1

            current_tokens = self._calculate_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                if self.verbose:
                    print(
                        f"Token budget exceeded ({current_tokens} > {self.max_token_budget}), "
                        f"forcing answer..."
                    )
                final_answer, total_cost = self._force_final_answer(
                    messages, context, total_cost, "Token budget exceeded", session=session
                )
                early_exit_reason = "token_budget_exceeded"
                break

            if self.verbose:
                print(
                    f"Loop {loop_count}/{self.max_loops} "
                    f"(Tokens: {current_tokens}/{self.max_token_budget})"
                )

            try:
                response = self.llm.chat(messages=messages, tools=tool_schemas)
            except Exception as e:
                if self.verbose:
                    print(f"LLM error: {e}")
                if session is not None:
                    session.event(
                        "llm_calls",
                        {"stage": "loop", "loop": loop_count, "error": f"{type(e).__name__}: {e}"},
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
                if session is not None:
                    session.event("trajectory", turn_record)

        # Loop exited without natural break or early exit ⇒ max loops hit.
        if early_exit_reason is None:
            if self.verbose:
                print(f"Max loops reached ({self.max_loops}), forcing answer...")
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

        return result



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
