"""Chat / tool-calling client over an OpenAI-compatible endpoint.

Includes per-model USD pricing so the agent loop can attribute cost per turn.
Falls back to a `default` price tuple for unknown model strings.
"""

from typing import Any, Dict, List, Optional

import requests
import tiktoken

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
    ):
        self.model = model or CHAT_MODEL or "gpt-4o-mini"
        self.api_key = api_key or CHAT_API_KEY
        self.base_url = (base_url or CHAT_API_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort

        if not self.api_key:
            raise ValueError(
                "Chat API key required. Set CHAT_API_KEY in .env or pass api_key."
            )

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

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

        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        usage = result.get("usage", {})

        # Cached-token reporting varies by provider:
        #   OpenAI / DeepSeek / Anthropic-via-relay typically expose it under
        #   ``usage.prompt_tokens_details.cached_tokens``; a few self-hosted
        #   relays use ``usage.prompt_cache_hit_tokens`` instead. We prefer
        #   the standard nested key and fall back to the legacy flat key.
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
