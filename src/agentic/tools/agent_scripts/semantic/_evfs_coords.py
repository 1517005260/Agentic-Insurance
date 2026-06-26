"""Thin stdlib client for the EvidenceFS semantic sidecar (not an agent command).

The semantic-tier scripts run inside the agent's hermetic shell sandbox — no
venv, no network, no GPU. The heavy retrieval channels live in a host-side
sidecar (``agentic.tools.evfs_sidecar``) that loads them once and serves the
ops over a unix-domain socket bound into the sandbox at ``$EVFS_SIDECAR_SOCK``.
This module is that socket's client: one request line out, one response line in.
"""
import json
import os
import socket
import sys


def call(op, **args):
    """Send one ``{"op": op, **args}`` request to the sidecar and return its
    decoded response dict. Raises ``SystemExit`` on a sidecar-side error or an
    unset / unreachable socket."""
    sock_path = os.environ.get("EVFS_SIDECAR_SOCK")
    if not sock_path:
        raise SystemExit("EVFS_SIDECAR_SOCK is not set (no semantic sidecar bound)")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(sock_path)
    except OSError as exc:
        raise SystemExit(f"cannot reach semantic sidecar at {sock_path}: {exc}")
    try:
        s.sendall((json.dumps({"op": op, **args}) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    line = buf.split(b"\n", 1)[0]
    if not line:
        raise SystemExit("empty response from semantic sidecar")
    resp = json.loads(line.decode("utf-8"))
    if "error" in resp:
        raise SystemExit(resp["error"])
    return resp


def print_rows(resp) -> None:
    """Print a ``{"header": [...], "rows": [...]}`` response as TSV; an optional
    ``"note"`` goes to stderr."""
    print("\t".join(resp["header"]))
    for row in resp["rows"]:
        print("\t".join(str(c) for c in row))
    note = resp.get("note")
    if note:
        print(note, file=sys.stderr)
