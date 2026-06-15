"""LLMClient must retry transient connection drops but never read/connect timeouts.

The chat session runs ``read_retries=0`` to cap wall-clock on a hung relay, which
also suppresses retries on instant connection drops (RemoteDisconnected → a plain
``requests.ConnectionError``). ``_post_with_conn_retry`` re-enables retries for
that case only: a ``ConnectionError`` is cheap to retry and almost always
recovers, while ``ReadTimeout`` (slow generation) and ``ConnectTimeout`` (host
unreachable, already retried at the urllib3 layer) — both ``Timeout`` subclasses —
must propagate immediately so the wall-clock cap holds.
"""
from unittest.mock import MagicMock

import pytest
import requests

from model_client.chat import LLMClient


def _client(conn_retries=2):
    c = LLMClient.__new__(LLMClient)
    c.base_url = "http://test/v1"
    c.api_key = "k"
    c.model = "gpt-4o-mini"
    c.temperature = 0.0
    c.max_tokens = 1024
    c.reasoning_effort = None
    c.disable_thinking = True
    c.conn_retries = conn_retries
    c.conn_backoff = 0.0  # no real sleeps in tests
    c._session = MagicMock()
    return c


def _ok_response():
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    return r


def _chat(client):
    return client.chat(messages=[{"role": "user", "content": "q"}])


def test_retries_connection_drop_then_succeeds():
    c = _client()
    c._session.post.side_effect = [
        requests.exceptions.ConnectionError("RemoteDisconnected"),
        _ok_response(),
    ]
    assert _chat(c)["message"]["content"] == "ok"
    assert c._session.post.call_count == 2


def test_exhausts_retries_then_raises():
    c = _client(conn_retries=2)
    c._session.post.side_effect = requests.exceptions.ConnectionError("down")
    with pytest.raises(requests.exceptions.ConnectionError):
        _chat(c)
    assert c._session.post.call_count == 3  # 1 + 2 retries


def test_read_timeout_not_retried():
    c = _client()
    c._session.post.side_effect = requests.exceptions.ReadTimeout("slow")
    with pytest.raises(requests.exceptions.ReadTimeout):
        _chat(c)
    assert c._session.post.call_count == 1


def test_connect_timeout_not_retried():
    # ConnectTimeout subclasses both ConnectionError and Timeout; it must NOT
    # be re-looped (the urllib3 connect retries already ran).
    c = _client()
    c._session.post.side_effect = requests.exceptions.ConnectTimeout("unreachable")
    with pytest.raises(requests.exceptions.ConnectTimeout):
        _chat(c)
    assert c._session.post.call_count == 1
