"""Fetch a single URL and return cleaned plaintext.

Companion to :class:`WebSearchTool`: search discovers candidates,
fetch reads the full page so the agent can cite verbatim. The
extractor is intentionally minimal — script/style/head are stripped,
remaining tags dropped, whitespace normalized. JS-rendered SPAs
return mostly boilerplate, in which case the agent should fall back
to the search snippet.

PDF / non-HTML payloads are rejected; the agent receives an error
envelope so it can choose another URL or give up the cite.
"""

import ipaddress
import logging
import os
import re
import socket
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse

import requests

from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool
from config.http import make_retry_session


# Hard escape hatch for environments where DNS is intercepted by a
# corporate / WSL gateway and every hostname resolves to a private IP
# inside the SSRF block range. Setting WEB_FETCH_DISABLE_SSRF_GUARD=1
# in the env disables the resolved-IP check; the URL scheme + literal-
# IP guard still applies. Keep this OFF in production.
_SSRF_GUARD_DISABLED_ENV = "WEB_FETCH_DISABLE_SSRF_GUARD"


def _ssrf_guard_disabled() -> bool:
    return os.environ.get(_SSRF_GUARD_DISABLED_ENV, "").strip() in (
        "1",
        "true",
        "yes",
    )

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = 15.0
_DEFAULT_MAX_CHARS = 8000
_MAX_CHARS_CAP = 32000
_MAX_REDIRECTS = 5
_USER_AGENT = (
    "Mozilla/5.0 (compatible; agentic-research/1.0; "
    "+https://example.invalid/about)"
)


# SSRF guard: any URL whose hostname resolves to one of these address
# classes is refused before the GET fires. Loopback / private / link-
# local / multicast / reserved cover the standard internal-network
# attack surface (cloud metadata services, in-cluster RPC, intranets).
def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_and_check_url(url: str) -> "tuple[Optional[str], Optional[str]]":
    """Validate ``url`` and return the IP we should pin the connection to.

    Returns ``(reason_if_blocked, pinned_ip)``:
      * ``(reason, None)``  — URL is refused; do not connect.
      * ``(None, ip)``      — URL is OK; pin the TCP connect to this IP
        so a DNS rebinding round between validate and connect cannot
        smuggle in a private-range answer.
      * ``(None, None)``    — URL passed the literal-IP / scheme guard
        but the resolved-IP check was skipped via the env opt-out.

    Rationale for IP pinning: ``socket.getaddrinfo`` here happens once.
    Without pinning, ``requests`` would call ``getaddrinfo`` again at
    connect time; an attacker-controlled DNS server can return a
    public IP for the validate call and a loopback IP for the connect
    call (TOCTOU). Pinning closes that window.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return f"URL parse failed: {exc}", None
    if parsed.scheme not in ("http", "https"):
        return f"unsupported scheme {parsed.scheme!r}", None
    host = parsed.hostname
    if not host:
        return "missing hostname", None

    # Literal-IP check always applies (cheap, no env override). Even
    # with the env opt-out below, a literal `http://10.0.0.1/...`
    # remains refused — the opt-out is for hostname-resolves-to-
    # private-IP cases caused by NAT'd DNS.
    try:
        ip = ipaddress.ip_address(host)
        if _ip_is_blocked(ip):
            return f"refusing private / loopback / reserved IP {host}", None
        return None, host  # literal IP — pin to itself
    except ValueError:
        pass

    if _ssrf_guard_disabled():
        # Trust the operator: corporate / WSL DNS gateways often map
        # public hostnames to private IPs which would block legitimate
        # Tavily-returned URLs. Skip the resolved-IP check; do NOT
        # pin (let requests resolve normally). The literal-IP and
        # scheme guards above still apply.
        return None, None

    # DNS resolution. Block any address in the answer set so a
    # multi-A response containing 127.0.0.1 still fails closed.
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return f"DNS resolution failed: {exc}", None
    chosen_ip: Optional[str] = None
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError:
            continue
        if _ip_is_blocked(ip):
            return (
                f"hostname {host} resolves to blocked address {addr} "
                f"(loopback / private / link-local / reserved). "
                f"Set {_SSRF_GUARD_DISABLED_ENV}=1 if your env "
                f"intercepts public DNS to a private gateway."
            ), None
        if chosen_ip is None:
            chosen_ip = str(ip)
    return None, chosen_ip


def _ip_is_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _reresolve_safe(host: str, expected_ip: str) -> "tuple[bool, Optional[str]]":
    """Re-resolve ``host`` and confirm the answer is still safe.

    Returns ``(safe, reason_if_unsafe)``. A "safe" re-resolution
    requires both:

      * ``expected_ip`` (the IP we validated up-front) is still in
        the answer set, AND
      * NO address in the new answer set is in a blocked range.

    The second condition guards against rebind answers that smuggle
    a blocked IP alongside the original (e.g. multi-A response
    ``[127.0.0.1, original_public_ip]``); ``requests`` would then
    be free to pick either, including the blocked one.

    Tightens the DNS rebinding window from "between validate and
    connect" (which can be hundreds of ms once the request body is
    being assembled) to "between two adjacent ``getaddrinfo`` calls"
    (sub-millisecond on a warm resolver cache). Not bulletproof —
    a full fix needs a custom urllib3 connection that pins the TCP
    target IP while preserving SNI / cert verification, which we
    deliberately don't ship in-process; production deployments
    should add an egress proxy that does its own filtering.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return False, f"DNS re-resolution failed: {exc}"
    expected_present = False
    for info in infos:
        addr = info[4][0].split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _ip_is_blocked(ip):
            return False, (
                f"DNS for {host} now includes blocked address {addr} "
                f"(possible rebind)"
            )
        if str(ip) == expected_ip:
            expected_present = True
    if not expected_present:
        return False, (
            f"DNS for {host} no longer includes the validated IP "
            f"{expected_ip} (possible rebind)"
        )
    return True, None


class _TextExtractor(HTMLParser):
    """Minimal HTML → text. Drop script/style/head; emit text + spaces.

    Heading / paragraph boundaries become double newlines so the
    output stays readable; inline tags merge with surrounding text.
    """

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "head"})
    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "li", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "header", "footer", "main",
        "blockquote", "pre",
    })

    def __init__(self) -> None:
        super().__init__()
        self._buf: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Any]) -> None:
        tl = tag.lower()
        if tl in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tl in self._BLOCK_TAGS and self._skip_depth == 0:
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tl = tag.lower()
        if tl in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tl in self._BLOCK_TAGS and self._skip_depth == 0:
            self._buf.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._buf.append(data)

    def get_text(self) -> str:
        text = "".join(self._buf)
        # Collapse whitespace but preserve paragraph breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class WebFetchTool(BaseTool):
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = float(timeout)
        # Lower retries than the LLM/embedding session — unreachable
        # external sites are noise; we shouldn't burn 5 retries on
        # every 404. 2 attempts cover transient DNS / TCP blips.
        self._session = make_retry_session(total=2, backoff_factor=0.5)

    @property
    def name(self) -> str:
        return "web_fetch"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": (
                    "GET an HTTP(S) URL and return cleaned plaintext.\n\n"
                    "Use this AFTER `web_search` to read a candidate's "
                    "full content for verbatim cite — the search "
                    "snippet (~300 chars) is rarely enough.\n\n"
                    "Result is HTML-stripped (script/style/head "
                    f"removed), truncated to `max_chars` (default "
                    f"{_DEFAULT_MAX_CHARS}, max {_MAX_CHARS_CAP}). "
                    "Cannot read PDFs or JavaScript-rendered SPAs; if "
                    "the result looks empty or boilerplate, try a "
                    "different URL or fall back to the search snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Absolute http(s) URL.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": (
                                f"Truncate to this many characters; "
                                f"capped at {_MAX_CHARS_CAP}. Default "
                                f"{_DEFAULT_MAX_CHARS}."
                            ),
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        url: str,
        max_chars: Optional[int] = None,
    ):
        if not url or not str(url).strip():
            return err(
                "invalid_argument",
                "`url` must be a non-empty string.",
                remediation="Pass a fully-qualified http(s) URL.",
                valid_example={"url": "https://www.ia.org.hk/en/index.html"},
            ), {"error": "invalid_argument"}
        url = str(url).strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return err(
                "invalid_url",
                "Only http:// and https:// URLs are supported.",
                remediation=(
                    "Prefix with https://; web_fetch refuses file://, "
                    "data://, ftp:// and other non-web schemes."
                ),
                url=url,
            ), {"error": "invalid_url"}

        try:
            cap = (
                _DEFAULT_MAX_CHARS
                if max_chars is None
                else max(500, min(int(max_chars), _MAX_CHARS_CAP))
            )
        except (TypeError, ValueError):
            cap = _DEFAULT_MAX_CHARS

        # Manual redirect chain so we can re-validate every hop. The
        # default ``allow_redirects=True`` would let an attacker
        # serve a 302 → http://127.0.0.1/.. that bypasses the up-front
        # SSRF check. Cap at _MAX_REDIRECTS hops so a redirect loop
        # exits cleanly.
        try:
            resp, final_url, hop_err = self._fetch_with_safe_redirects(url)
        except requests.exceptions.Timeout:
            return err(
                "timeout",
                f"GET {url} timed out after {self._timeout}s.",
                remediation="Try a different URL; this site may be slow or geo-blocked.",
            ), {"error": "timeout"}
        except requests.exceptions.RequestException as exc:
            logger.info("web_fetch failed url=%s: %r", url, exc)
            return err(
                "fetch_error",
                f"GET {url} failed: {type(exc).__name__}: {exc}",
                remediation="Try a different URL; the page may be down, geo-blocked, or behind auth.",
            ), {"error": "fetch_error"}
        if hop_err is not None:
            return err(
                "blocked_url",
                hop_err,
                remediation="Pass a public-internet URL; web_fetch refuses loopback / private / link-local / reserved address ranges.",
                url=url,
            ), {"error": "blocked_url"}
        if resp is None:
            return err(
                "fetch_error",
                f"GET {url} failed without a response.",
                remediation="Try a different URL.",
            ), {"error": "fetch_error"}
        try:
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.info("web_fetch http error url=%s: %r", final_url, exc)
            return err(
                "fetch_error",
                f"GET {final_url} returned {resp.status_code}: {exc}",
                remediation="Try a different URL; the page may be down, geo-blocked, or behind auth.",
            ), {"error": "fetch_error"}

        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "html" not in ctype and "text" not in ctype:
            return err(
                "unsupported_content_type",
                f"Unsupported content-type: {ctype or '<missing>'}",
                remediation=(
                    "web_fetch only supports text/html. For PDFs, hint "
                    "the user that the source is a PDF and cite the "
                    "URL only."
                ),
                content_type=ctype,
            ), {"error": "unsupported_content_type"}

        # requests guesses encoding from Content-Type charset; if the
        # header lacks one it falls back to ISO-8859-1 (a HTTP/1.1 quirk),
        # which mangles UTF-8 CJK pages. Override to apparent_encoding
        # when the header doesn't pin a charset.
        if "charset=" not in ctype:
            resp.encoding = resp.apparent_encoding or "utf-8"
        raw = resp.text

        parser = _TextExtractor()
        try:
            parser.feed(raw)
            extracted = parser.get_text()
        except Exception as exc:
            logger.info("web_fetch html parse fallback url=%s: %r", url, exc)
            extracted = raw  # last resort: return raw

        truncated = len(extracted) > cap
        if truncated:
            extracted = extracted[:cap].rstrip() + "\n…[truncated]…"

        title = self._extract_title(raw)
        approx_tokens = len(extracted) // 4
        context.add_retrieval_log(
            tool_name="web_fetch",
            tokens=approx_tokens,
            metadata={
                "url": url,
                "final_url": final_url,
                "status": resp.status_code,
                "chars": len(extracted),
                "truncated": truncated,
            },
        )
        return (
            ok(
                "WebFetchObservation",
                url=final_url,
                title=title,
                text=extracted,
                chars=len(extracted),
                truncated=truncated,
            ),
            {
                "retrieved_tokens": approx_tokens,
                "url": url,
            },
        )

    def _fetch_with_safe_redirects(
        self, url: str
    ) -> "tuple[Optional[requests.Response], str, Optional[str]]":
        """GET ``url``, validating SSRF guards on each redirect hop.

        Returns ``(response, final_url, block_reason)``. If
        ``block_reason`` is set, the URL (or one of its redirect
        targets) failed the private-IP check and the response is the
        final 3xx that landed there (so the caller can inspect status
        + headers if useful for diagnostics).

        Each hop calls :func:`_resolve_and_check_url` which returns a
        validated IP. We pass that IP as ``socket_options`` /
        ``DNS-resolved`` IP via the per-call adapter at
        :meth:`_get_with_pinned_ip`, closing the DNS-rebinding gap
        between validate and connect.
        """
        from urllib.parse import urljoin

        current = url
        last_response: Optional[requests.Response] = None
        for hop in range(_MAX_REDIRECTS + 1):
            block_reason, pinned_ip = _resolve_and_check_url(current)
            if block_reason is not None:
                return last_response, current, block_reason
            resp = self._get_with_pinned_ip(current, pinned_ip)
            last_response = resp
            if not (300 <= resp.status_code < 400 and "Location" in resp.headers):
                return resp, current, None
            next_url = urljoin(current, resp.headers["Location"])
            current = next_url
        return last_response, current, "redirect chain exceeded maximum hops"

    def _get_with_pinned_ip(
        self, url: str, pinned_ip: Optional[str]
    ) -> requests.Response:
        """Issue a GET; re-validate DNS just before the TCP connect.

        We can't fully pin the connection's target IP without a
        custom urllib3 connection class that preserves SNI / cert
        verification (annoying in-process, easy to get wrong). The
        in-process mitigation we DO apply: re-resolve the hostname
        right before issuing the request and refuse if the IP set
        changed since validate. That tightens the DNS-rebinding
        window from "validate → connect" (hundreds of ms) to
        "two adjacent ``getaddrinfo`` calls" (microseconds on a
        warm resolver cache).

        For tightening this further, deploy behind an egress proxy
        (squid, envoy) that enforces its own private-IP block on
        outbound connections.
        """
        from urllib.parse import urlparse

        if pinned_ip is not None:
            host = urlparse(url).hostname
            if host and not _ip_is_literal(host):
                safe, reason = _reresolve_safe(host, pinned_ip)
                if not safe:
                    raise requests.exceptions.RequestException(
                        reason or "rebind detected"
                    )
        return self._session.get(
            url,
            timeout=self._timeout,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,*/*",
            },
            allow_redirects=False,
        )

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(
            r"<title[^>]*>(.*?)</title>",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return ""
        return re.sub(r"\s+", " ", m.group(1)).strip()[:200]
