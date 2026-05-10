"""Shared HTTP session factory with transport-level retries.

Every outbound HTTP(S) call in the project — chat, embedding, rerank,
PaddleOCR, Tavily, VLM — should run through a session built here, not
through ``requests.get`` / ``requests.post`` at module top level.

Why a session, not a decorator
------------------------------

Retries are handled at the urllib3 transport layer via
``urllib3.util.retry.Retry`` mounted on a ``requests.Session``. This is
the canonical approach for ``requests``: connection-level retries
(timeouts, DNS, broken streams) AND status-code retries (429, 5xx) are
unified, the ``Retry-After`` header is honored automatically, and the
caller's stack is never polluted by a wrapper ``try/except``.

Defaults
--------

* ``total=5``  — five attempts max
* ``backoff_factor=0.6`` — sleeps roughly 0.6, 1.2, 2.4, 4.8 s between
  retries (capped by urllib3 at 120 s)
* ``status_forcelist=(408, 425, 429, 500, 502, 503, 504)`` — transient
  server signals that should retry; 4xx auth / payload errors do NOT
* ``allowed_methods`` includes POST because every model API we hit is
  POST (chat completion, embedding, rerank). LLM endpoints are
  expensive to retry but a single failed attempt costs even more.
* ``raise_on_status=False`` — leave it to the caller's
  ``response.raise_for_status()`` so the exception type stays standard
  ``requests.HTTPError`` rather than urllib3's ``MaxRetryError``.

Usage
-----

::

    from config.http import make_retry_session

    class FooClient:
        def __init__(self) -> None:
            self._session = make_retry_session()

        def call(self):
            resp = self._session.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
"""

from typing import Iterable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_DEFAULT_STATUS_FORCELIST = (408, 425, 429, 500, 502, 503, 504)
_DEFAULT_ALLOWED_METHODS = frozenset({"HEAD", "GET", "PUT", "DELETE", "POST", "OPTIONS"})


def make_retry_session(
    *,
    total: int = 5,
    backoff_factor: float = 0.6,
    status_forcelist: Iterable[int] = _DEFAULT_STATUS_FORCELIST,
    allowed_methods: Optional[Iterable[str]] = None,
    pool_connections: int = 16,
    pool_maxsize: int = 16,
    read_retries: Optional[int] = None,
) -> requests.Session:
    """Build a :class:`requests.Session` with transport-level retries.

    Parameters mirror :class:`urllib3.util.retry.Retry` directly so the
    caller can override per-client (e.g. embedding endpoints often want
    a higher ``total`` because batches are expensive to redo).

    ``read_retries`` (None → use ``total``, 0 → disable) is broken out
    so callers that already enforce a per-call ``timeout=(connect, read)``
    can opt out of the additional 5×backoff that ``Retry(read=total)``
    introduces. The chat client uses 0 — without it a hung LLM relay
    blocks for ``5 × read_timeout`` instead of the documented one-shot,
    and any wall-clock fallback above the chat client never fires.
    """
    retry = Retry(
        total=total,
        connect=total,
        read=total if read_retries is None else read_retries,
        status=total,
        status_forcelist=tuple(status_forcelist),
        allowed_methods=frozenset(allowed_methods) if allowed_methods else _DEFAULT_ALLOWED_METHODS,
        backoff_factor=backoff_factor,
        respect_retry_after_header=True,
        raise_on_status=False,
        raise_on_redirect=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


__all__ = ["make_retry_session"]
