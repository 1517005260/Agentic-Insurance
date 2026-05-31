"""Chat / tool-calling client over an OpenAI-compatible endpoint.

Includes per-model USD pricing so the agent loop can attribute cost per turn.
Falls back to a `default` price tuple for unknown model strings.
"""

import json
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional

from config.shared import shared_tiktoken_encoder

logger = logging.getLogger(__name__)


class StreamProtocolError(RuntimeError):
    """The chat-completions stream ended without a clean termination.

    Raised when the SSE stream closes without seeing either ``[DONE]``
    or a frame carrying ``finish_reason``. The accumulated text up to
    that point is incomplete; surfacing this as an exception lets the
    runner emit an ``error`` event rather than presenting truncated
    output as if it were a clean answer.
    """

from config.http import make_retry_session
from config.shared import shared_session
from config.settings import CHAT_API_BASE_URL, CHAT_API_KEY, CHAT_MODEL


class LLMClient:
    # USD per 1M tokens: (input, cached_input, output)
    PRICING = {
        "gpt-5.2-pro": (21.0, 2.1, 168.0),
        "gpt-5.2": (1.75, 0.175, 14.0),
        "gpt-5.1": (1.5, 0.15, 12.0),
        "gpt-5-mini": (0.25, 0.025, 2.0),
        "gpt-5": (1.5, 0.15, 12.0),
        "gpt-4.1-mini": (0.4, 0.1, 1.6),
        "gpt-4.1-nano": (0.1, 0.025, 0.4),
        "gpt-4.1": (2.0, 0.5, 8.0),
        "gpt-4o-mini": (0.15, 0.075, 0.6),
        "gpt-4o": (2.5, 1.25, 10.0),
        "gpt-4-turbo": (10.0, 5.0, 30.0),
        "o4-mini": (1.1, 0.275, 4.4),
        "o3-mini": (1.1, 0.275, 4.4),
        "o3": (10.0, 2.5, 40.0),
        "o1-mini": (1.1, 0.275, 4.4),
        "o1": (15.0, 3.75, 60.0),
        "claude-4.5-opus": (5.0, 0.5, 25.0),
        "claude-4.5-sonnet": (3.0, 0.3, 15.0),
        "claude-4.5-haiku": (1.0, 0.1, 5.0),
        "claude-4-opus": (15.0, 1.5, 75.0),
        "claude-4-sonnet": (3.0, 0.3, 15.0),
        "claude-sonnet": (3.0, 0.3, 15.0),
        "claude-opus": (5.0, 0.5, 25.0),
        "claude-haiku": (1.0, 0.1, 5.0),
        "gemini-3-pro": (2.0, 0.2, 12.0),
        "gemini-3-flash": (0.5, 0.05, 3.0),
        "gemini-2.5-pro": (1.25, 0.125, 10.0),
        "gemini-2.5-flash": (0.3, 0.075, 2.5),
        "gemini-2.0-flash": (0.1, 0.025, 0.4),
        "gemini-pro": (1.25, 0.125, 5.0),
        "gemini-flash": (0.075, 0.02, 0.3),
        "default": (1.0, 0.1, 5.0),
    }

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        reasoning_effort: Optional[str] = None,
        disable_thinking: bool = True,
    ):
        self.model = model or CHAT_MODEL or "gpt-4o-mini"
        self.api_key = api_key or CHAT_API_KEY
        self.base_url = (base_url or CHAT_API_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        # DeepSeek / Qwen and many self-hosted vLLM relays default to
        # emitting ``reasoning_content`` (chain-of-thought) on every
        # token; for our task surfaces that's bandwidth + cost waste
        # AND triggers the malformed-frame path on CJK-heavy SSE. We
        # opt out at the API boundary; callers that want reasoning
        # (proof-style audits) can pass ``disable_thinking=False``.
        self.disable_thinking = disable_thinking

        if not self.api_key:
            raise ValueError(
                "Chat API key required. Set CHAT_API_KEY in .env or pass api_key."
            )

        # ``read=0`` disables retries on read timeout — combined with
        # the per-call ``timeout=(10, 120)`` it caps the wall-clock
        # any chat() / chat_stream() call can stall at ≤ 120 s + a
        # bit of connect/backoff. Without this override, the default
        # ``read=total`` (=5) would let a hung relay block for
        # 5 × 120 s before the caller sees a TimeoutError, and the
        # preprocess wall-clock fallback (130 s) would never trigger.
        # status / connect retries are still useful for transient
        # 5xx and DNS hiccups, so we keep those.
        # Process-wide pool — distinct profile so this tight retry
        # policy doesn't bleed into other clients.
        self._session = shared_session(
            "chat-no-read-retry", lambda: make_retry_session(read_retries=0)
        )

        # Process-cached encoder shared with all agent / tool callsites.
        self.tokenizer = shared_tiktoken_encoder("gpt-4o")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total += self.count_tokens(item.get("text", ""))
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    total += self.count_tokens(str(tc.get("function", {})))
        return total

    def calculate_cost(self, usage: dict) -> float:
        model_lower = self.model.lower()

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_details = usage.get("prompt_tokens_details", {}) or {}
        cached_tokens = prompt_details.get("cached_tokens", 0)
        input_tokens = max(prompt_tokens - cached_tokens, 0)

        for key in self.PRICING:
            if key in model_lower:
                input_price, cached_price, output_price = self.PRICING[key]
                break
        else:
            input_price, cached_price, output_price = self.PRICING["default"]

        usd_cost = (
            (input_tokens / 1_000_000) * input_price
            + (cached_tokens / 1_000_000) * cached_price
            + (completion_tokens / 1_000_000) * output_price
        )
        return round(usd_cost, 6)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.disable_thinking:
            _apply_thinking_off(payload)

        # ``(connect, read)`` instead of a single 300s wall: with one
        # number, a hung relay (TCP open, no body bytes) blocks for the
        # full 5 min and the runner can only emit a giant silent gap.
        # 10s connect catches dead endpoints quickly; 120s read covers
        # slow but live generations without rewarding indefinite stalls.
        # See preprocess() for the per-call timeout fallback that turns
        # a ReadTimeout into a usable {rewrite=query} so the pipeline
        # still produces an answer instead of dying on the first hang.
        response = self._session.post(
            url, headers=headers, json=payload, timeout=(10, 120),
        )
        response.raise_for_status()
        result = response.json()
        usage = result.get("usage", {})

        # Cached-token reporting varies by provider:
        #   OpenAI / DeepSeek / Anthropic-via-relay expose it under
        #   ``usage.prompt_tokens_details.cached_tokens``; some self-hosted
        #   relays put it at ``usage.prompt_cache_hit_tokens``. Prefer the
        #   nested key, accept the flat key as a fallback.
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached_tokens = (
            prompt_details.get("cached_tokens")
            or usage.get("prompt_cache_hit_tokens")
            or 0
        )

        return {
            "message": result["choices"][0]["message"],
            "input_tokens": usage.get("prompt_tokens", 0),
            "cached_tokens": int(cached_tokens),
            "output_tokens": usage.get("completion_tokens", 0),
            "cost": self.calculate_cost(usage),
            "raw_response": result,
        }

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream the assistant message in OpenAI SSE wire format.

        Yields a sequence of dicts, each carrying one or more of:

        * ``"delta"``      — incremental text fragment (most frames)
        * ``"finish_reason"`` — set on the final delta (``"stop"``,
          ``"length"``, ``"tool_calls"``, ...). The frame may also carry
          a final ``"delta"``.
        * ``"usage"``      — token counts dict, only on the very last
          frame and only when the provider sets ``stream_options.include_usage``.
        * ``"cost"``       — USD cost computed from ``usage`` (same frame).

        Tool calls are intentionally NOT supported — passing ``tools``
        raises ``NotImplementedError``. Streaming tool_calls correctly
        requires assembling the full structured array across deltas
        and we have no caller that needs it; the streaming path is for
        the *answer* stage of RAG / the final natural-text turn of an
        agent loop where output is plain text.

        Raises :class:`StreamProtocolError` when the stream ends without
        either an ``[DONE]`` sentinel or any frame carrying a
        ``finish_reason``. Connection drops mid-stream therefore become
        loud failures the runner can convert into an SSE ``error`` frame
        instead of silently presenting a truncated answer.
        """
        if tools:
            raise NotImplementedError(
                "chat_stream() does not support tools; use chat() for tool-calling turns."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": True,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.disable_thinking:
            _apply_thinking_off(payload)

        saw_done = False
        saw_finish_reason = False
        saw_malformed_data = False
        # ``(connect, read)`` tuple: see chat() for the rationale.
        # ``read=120`` doubles as a between-byte watchdog — if the
        # provider stalls between SSE chunks, ``iter_lines`` raises
        # rather than waiting out a full 5-min wall on a single-int
        # timeout.
        with self._session.post(
            url, headers=headers, json=payload, timeout=(10, 120), stream=True
        ) as response:
            response.raise_for_status()
            # Some relays send ``Content-Type: text/event-stream`` without an
            # explicit charset; requests then defaults to ISO-8859-1. With
            # ``decode_unicode=True`` ``iter_lines`` calls ``str.splitlines``,
            # which splits on U+0085 (NEL) — and the third byte of many CJK
            # UTF-8 sequences is 0x85 (e.g. ``待`` = E5 BE 85). That fragments
            # otherwise-valid frames and breaks every CJK-heavy stream. Force
            # UTF-8: every OpenAI-style SSE relay we use is UTF-8 in practice.
            response.encoding = "utf-8"
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                # OpenAI SSE: each event is "data: <json>"; the relay
                # may also send comment lines (": foo") for keepalive.
                if not raw_line.startswith("data:"):
                    continue
                payload_text = raw_line[len("data:"):].strip()
                if payload_text == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(payload_text)
                except json.JSONDecodeError:
                    # A malformed JSON frame is a relay bug — log,
                    # remember, and continue (so we still capture any
                    # following finish_reason for diagnostics). At end
                    # of stream we raise StreamProtocolError because
                    # the dropped frame may have carried a token, in
                    # which case the assembled text is silently corrupt.
                    logger.warning(
                        "chat_stream: malformed data line: %r",
                        payload_text[:200],
                    )
                    saw_malformed_data = True
                    continue

                out: Dict[str, Any] = {}
                choices = chunk.get("choices") or []
                if choices:
                    choice0 = choices[0]
                    delta = (choice0.get("delta") or {}).get("content")
                    if delta:
                        out["delta"] = delta
                    finish = choice0.get("finish_reason")
                    if finish:
                        out["finish_reason"] = finish
                        saw_finish_reason = True

                usage = chunk.get("usage")
                if usage:
                    out["usage"] = {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "cached_tokens": (
                            (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                            or usage.get("prompt_cache_hit_tokens")
                            or 0
                        ),
                        "output_tokens": usage.get("completion_tokens", 0),
                    }
                    out["cost"] = self.calculate_cost(usage)

                if out:
                    yield out

        if not saw_done and not saw_finish_reason:
            # Stream closed cleanly at the TCP layer but the protocol
            # never reached a terminal frame — partial content is at
            # best truncated, at worst missing entirely.
            raise StreamProtocolError(
                "chat completion stream ended without [DONE] or finish_reason"
            )
        if saw_malformed_data:
            # We kept reading after the bad frame to capture
            # finish_reason for the log; surface the failure now so the
            # runner doesn't present a possibly-corrupt answer as clean.
            raise StreamProtocolError(
                "chat completion stream contained malformed data frame(s); "
                "assembled text may be corrupt"
            )


# ----------------------------------------------------- thinking-off helper ----


def _apply_thinking_off(payload: Dict[str, Any]) -> None:
    """Mute chain-of-thought on relays that route to thinking-prone models.

    DeepSeek / Qwen / vLLM use ``enable_thinking`` / ``chat_template_kwargs``
    / ``extra_body``; sending all three is the cheapest way to cover them.

    BUT: some OpenAI-compatible relays (e.g. yunwu.ai's gpt-4o-mini route)
    strict-validate the request body and reject unknown root fields with
    ``new_api_error: shell_api_error``. We therefore gate the override on
    the model name — only send the flags for models that need them.
    GPT / Claude / Gemini routes don't ship reasoning_content on every
    turn (their reasoning APIs are explicit, like ``reasoning_effort``),
    so skipping is safe.
    """
    model = (payload.get("model") or "").lower()
    # Models that emit ``reasoning_content`` even when the caller
    # doesn't ask for thinking — needs the explicit kill switch.
    thinking_prone = (
        "deepseek" in model
        or "qwen" in model
        or "yi-" in model
        or model.startswith("yi")
        or "glm" in model
        or "thinking" in model
        or "reason" in model
    )
    if not thinking_prone:
        return
    payload["enable_thinking"] = False
    payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    payload.setdefault("extra_body", {})["enable_thinking"] = False


# ----------------------------------------------------- module-level cache ----

# (model, base_url, api_key, temperature, max_tokens, reasoning_effort) → instance.
# Admin can change the chat model from the config center; we don't want
# to rebuild the underlying ``requests.Session`` (urllib3 retry pool +
# tiktoken encoder load) on every request. Insertion-order eviction
# would be overkill — the live distinct-key set is at most a handful.
_INSTANCE_CACHE: Dict[tuple, "LLMClient"] = {}
_INSTANCE_LOCK = threading.Lock()


def get_cached_client(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16384,
    reasoning_effort: Optional[str] = None,
) -> "LLMClient":
    """Return a memoized ``LLMClient`` for the given parameter tuple.

    Designed for the API path where admin can swap models per request.
    Experiment scripts should keep instantiating ``LLMClient(...)``
    directly so each run gets a fresh, independent session.
    """
    key = (
        model or CHAT_MODEL,
        (base_url or CHAT_API_BASE_URL).rstrip("/"),
        api_key or CHAT_API_KEY,
        float(temperature),
        int(max_tokens),
        reasoning_effort,
    )
    with _INSTANCE_LOCK:
        cached = _INSTANCE_CACHE.get(key)
        if cached is not None:
            return cached
        instance = LLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        _INSTANCE_CACHE[key] = instance
        return instance
