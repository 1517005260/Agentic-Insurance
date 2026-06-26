"""Host-side warm sidecar for the EvidenceFS semantic scripts.

The five semantic scripts (``rank_passages`` / ``search_dense`` /
``seed_surfaces`` / ``candidate_aliases`` / ``semantic_bridge``) reuse the real
retrieval channels (``GraphPPRChannel`` / ``SemanticChannel``) with zero
degradation. Loading those channels — GLiNER + faiss + igraph + the embedding
client — costs ~45s, and the agent runs each script as a fresh subprocess in its
sandbox, so a per-call load is the dominant cost.

This sidecar loads the channels (and the EvidenceFS coordinate maps) **once** in
a host process and serves the five ops over a unix-domain socket. The scripts
become thin stdlib clients (``agent_scripts/semantic/_evfs_coords.py``). Because
AF_UNIX works across a bwrap mount namespace WITHOUT the network namespace, the
semantic sandbox can stay fully hermetic (``--unshare-all`` + ``--clearenv``)
with only the one socket bound in — all venv / network / GPU / creds stay on the
host side, here.

Protocol: one request = one JSON line ``{"op": "...", ...args}``; one response =
one JSON line ``{"header": [...], "rows": [[...], ...]}`` (optionally a
``"note"``) or ``{"error": "..."}``.

Run directly as ``python -m agentic.tools.evfs_sidecar <socket_path>``.
"""

import csv
import fcntl
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from config.settings import evidence_fs_root

logger = logging.getLogger("evfs_sidecar")

# Mounted-into-sandbox socket path the clients connect to (host path differs).
SIDECAR_MOUNT = "/run/evfs_sidecar.sock"
# Exit after this long with no request, so a sidecar never lingers.
IDLE_TIMEOUT_S = 900
# Budget for ``ensure_running`` to spawn + warm the daemon (one-time channel load).
SPAWN_TIMEOUT_S = 120
_RECV = 65536


# ----------------------------------------------------------- coordinate maps
# Loaded once at startup; the channels rank in LinearRAG coordinates and these
# maps project that ranking back to EvidenceFS coordinates the agent can read.

def _load_doc_index(fs_root: Path) -> dict:
    """``documents.tsv`` ``source_path`` -> ``doc_id`` (== channel ``file_id``)."""
    out = {}
    with (fs_root / "nodes" / "documents.tsv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["source_path"]] = row["doc_id"]
    return out


def _load_surface_index(fs_root: Path):
    """``surface_index.tsv`` -> (norm->(sf, raw, label), sf->(norm, raw, label))."""
    by_norm, by_sf = {}, {}
    with (fs_root / "views" / "surface_index.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            by_norm[r["surface_norm"]] = (r["surface_id"], r["surface_raw"], r["ner_label"])
            by_sf[r["surface_id"]] = (r["surface_norm"], r["surface_raw"], r["ner_label"])
    return by_norm, by_sf


def _load_sentence_index(fs_root: Path) -> dict:
    """``sentence_text.tsv`` text -> ``sentence_id`` (for the bridge ``via``)."""
    out = {}
    with (fs_root / "views" / "sentence_text.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["text"]] = r["sentence_id"]
    return out


def _preview(page_file: Path, n: int = 140) -> str:
    if not page_file.is_file():
        return ""
    return " ".join(page_file.read_text("utf-8", errors="replace").split())[:n]


def _clean(text: str) -> str:
    """Match the emitter's ``sentence_text.tsv`` hygiene (collapse tab/CR/LF) so
    the raw channel sentence text maps to the (already-collapsed) EvidenceFS s_*."""
    return re.sub(r"[\t\r\n]+", " ", text).strip()


class _Server:
    """Holds the loaded channels + coordinate maps; one op handler per request.
    Requests are served serially under a lock (the channels are not reentrant)."""

    _OPS = ("ppr", "dense", "seed", "aliases", "bridge")

    def __init__(self, fs_root: Path, ppr, dense) -> None:
        self.fs_root = Path(fs_root)
        self.ppr = ppr
        self.dense = dense
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self.doc_by_fid = _load_doc_index(self.fs_root)
        self.surf_by_norm, self.surf_by_sf = _load_surface_index(self.fs_root)
        self.sent_text_to_id = _load_sentence_index(self.fs_root)
        # surface_norm -> entity hash (the bridge tail lookup).
        self.text2hash = {t: h for h, t in self.ppr.entity_store.hash_id_to_text.items()}

    # --------------------------------------------------------------- ranking
    def _ranked_rows(self, hits, top_k: int) -> dict:
        """Project channel ``ChannelHit`` page ranking into EvidenceFS rows."""
        header = ["rank", "doc_id", "page", "score", "page_file", "preview"]
        rows = []
        seen = set()
        for hit in hits:
            doc_id = self.doc_by_fid.get(hit.file_id)
            if not doc_id:
                continue
            try:
                page_n = int(str(hit.page_id).split("_")[-1])
            except ValueError:
                continue
            key = (doc_id, page_n)
            if key in seen:  # channel ranks paragraph passages; fold to one page
                continue
            page_file = f"documents/{doc_id}/pages/page_{page_n:04d}.md"
            if not (self.fs_root / page_file).is_file():  # mapping miss: skip
                continue
            seen.add(key)
            rows.append([
                str(len(rows) + 1), doc_id, str(page_n), f"{hit.score:.6f}",
                page_file, _preview(self.fs_root / page_file),
            ])
            if len(rows) >= top_k:
                break
        resp = {"header": header, "rows": rows}
        if not rows:
            resp["note"] = ("(no rows mapped — the live KG store and this "
                            "evidence_fs/ may be from different builds)")
        return resp

    # ------------------------------------------------------------------- ops
    def op_ppr(self, query, top_k=10, file_ids=None) -> dict:
        from rag.preprocess import QueryContext
        ctx = QueryContext(
            query=query, hyde="", rewrite="", lang="", regexes=[],
            file_ids=file_ids or None, enable_ppr_seed_fallback=True,
        )
        hits, _debug = self.ppr.retrieve_with_debug(ctx)
        return self._ranked_rows(hits, top_k)

    def op_dense(self, query, top_k=10, file_ids=None) -> dict:
        from rag.preprocess import QueryContext
        ctx = QueryContext(
            query=query, hyde="", rewrite="", lang="", regexes=[],
            file_ids=file_ids or None,
        )
        hits = self.dense.retrieve(ctx)
        return self._ranked_rows(hits, top_k)

    def op_seed(self, query, top_k=10) -> dict:
        emb = self.ppr.embedding_client.encode(query, is_query=True)
        if getattr(emb, "ndim", 1) == 2:
            emb = emb[0]
        hits = self.ppr.entity_store.topk(emb, top_k * 3)
        text_of = self.ppr.entity_store.hash_id_to_text
        rows = []
        seen = set()
        for hash_id, score in hits:
            row = self.surf_by_norm.get(text_of.get(hash_id, ""))
            if not row or row[0] in seen:
                continue
            seen.add(row[0])
            rows.append([str(len(rows) + 1), row[0], row[1], row[2], f"{float(score):.4f}"])
            if len(rows) >= top_k:
                break
        return {"header": ["rank", "surface_id", "surface_raw", "ner_label", "sim"], "rows": rows}

    def op_aliases(self, surface_id, top_k=10) -> dict:
        if surface_id not in self.surf_by_sf:
            return {"error": f"no such surface: {surface_id}"}
        # Embed the NORMALIZED surface (the form the entity store is keyed on),
        # not the raw mention — casing / traditional-CJK / markup differences in
        # the raw text shift the nearest-neighbour result.
        norm = self.surf_by_sf[surface_id][0]
        emb = self.ppr.embedding_client.encode(norm)
        if getattr(emb, "ndim", 1) == 2:
            emb = emb[0]
        hits = self.ppr.entity_store.topk(emb, top_k * 3 + 1)
        text_of = self.ppr.entity_store.hash_id_to_text
        rows = []
        seen = {surface_id}
        for hash_id, score in hits:
            row = self.surf_by_norm.get(text_of.get(hash_id, ""))
            if not row or row[0] in seen:
                continue
            seen.add(row[0])
            rows.append([str(len(rows) + 1), row[0], row[1], row[2], f"{float(score):.4f}"])
            if len(rows) >= top_k:
                break
        return {"header": ["rank", "surface_id", "surface_raw", "ner_label", "sim"], "rows": rows}

    def op_bridge(self, surface_id, query, top_k=10) -> dict:
        import numpy as np
        if surface_id not in self.surf_by_sf:
            return {"error": f"no such surface: {surface_id}"}
        tail_hash = self.text2hash.get(self.surf_by_sf[surface_id][0])
        if tail_hash is None:
            return {"error": f"surface not in entity store: {surface_id}"}

        qemb = self.ppr.embedding_client.encode(query, is_query=True)
        if getattr(qemb, "ndim", 1) == 2:
            qemb = qemb[0]
        all_hits = self.ppr.sentence_store.topk(qemb, len(self.ppr.sentence_store))
        sent_idx = {h: i for i, (h, _) in enumerate(all_hits)}
        sent_sims = np.asarray([s for _, s in all_hits], dtype=np.float64)

        neighbors = self.ppr.cooccurrence_neighbors(
            tail_hash, sent_sims, sent_idx, top_l=top_k * 2,
        )
        surf_text = self.ppr.entity_store.hash_id_to_text
        sent_text = self.ppr.sentence_store.hash_id_to_text
        rows = []
        seen = {surface_id}
        for nb in neighbors:
            row = self.surf_by_norm.get(surf_text.get(nb["hash_id"], ""))
            if not row or row[0] in seen:
                continue
            seen.add(row[0])
            via_hash = nb["via_sids"][0] if nb["via_sids"] else None
            vtext = _clean(sent_text.get(via_hash, "")) if via_hash else ""
            # Prefer the s_* id (so the agent can `show_sentence`); on a mapping
            # miss fall back to the cleaned bridge sentence itself.
            via = self.sent_text_to_id.get(vtext) or (vtext[:70] + ("…" if len(vtext) > 70 else ""))
            rows.append([str(len(rows) + 1), row[0], row[1], f"{nb['max_cos']:.4f}", via])
            if len(rows) >= top_k:
                break
        return {"header": ["rank", "surface_b", "surface_raw", "score", "via_sentence"], "rows": rows}

    # ---------------------------------------------------------------- server
    def serve(self, socket_path: str) -> None:
        parent = os.path.dirname(socket_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            os.unlink(socket_path)  # clear a stale socket from a prior daemon
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(socket_path)
        srv.listen(64)
        self._last = time.monotonic()
        threading.Thread(target=self._watchdog, daemon=True).start()
        logger.info("listening on %s", socket_path)
        while True:
            conn, _ = srv.accept()
            self._last = time.monotonic()
            try:
                self._handle(conn)
            except Exception:  # noqa: BLE001 — never let one request kill the daemon
                logger.exception("request handling failed")
            finally:
                conn.close()
                self._last = time.monotonic()

    def _handle(self, conn: socket.socket) -> None:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(_RECV)
            if not chunk:
                break
            buf += chunk
        line = buf.split(b"\n", 1)[0]
        if not line:
            return
        try:
            req = json.loads(line.decode("utf-8"))
            op = req.pop("op")
        except Exception as exc:  # noqa: BLE001
            self._send(conn, {"error": f"bad request: {exc}"})
            return
        if op not in self._OPS:
            self._send(conn, {"error": f"unknown op: {op!r}"})
            return
        handler = getattr(self, f"op_{op}")
        with self._lock:
            try:
                resp = handler(**req)
            except Exception as exc:  # noqa: BLE001
                logger.exception("op %s failed", op)
                resp = {"error": f"{type(exc).__name__}: {exc}"}
        self._send(conn, resp)

    @staticmethod
    def _send(conn: socket.socket, obj: dict) -> None:
        conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))

    def _watchdog(self) -> None:
        while True:
            time.sleep(30)
            if time.monotonic() - self._last > IDLE_TIMEOUT_S:
                logger.info("idle %ds — exiting", IDLE_TIMEOUT_S)
                os._exit(0)


# -------------------------------------------------------------- entrypoints
def main(socket_path) -> None:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s evfs-sidecar %(levelname)s %(message)s",
    )
    # The embedding API is reached through the env base URL; the WSL/proxy
    # gateway breaks it, so drop proxies (mirrors the app's call sites).
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(k, None)
    from rag.channels.graph_ppr import GraphPPRChannel
    from rag.channels.semantic import SemanticChannel

    logger.info("loading retrieval channels (one-time) …")
    t0 = time.monotonic()
    ppr = GraphPPRChannel()
    ppr.reload()
    dense = SemanticChannel()
    logger.info("channels loaded in %.1fs", time.monotonic() - t0)
    _Server(evidence_fs_root(), ppr, dense).serve(str(socket_path))


def _can_connect(socket_path: str) -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect(socket_path)
        return True
    except OSError:
        return False
    finally:
        s.close()


def ensure_running(socket_path) -> None:
    """Idempotent, race-safe: if the sidecar already answers on ``socket_path``
    return; otherwise spawn it host-side (detached, full env) and poll-connect
    until it answers. A file lock under the socket dir serializes concurrent
    callers so only one daemon is ever spawned."""
    socket_path = str(socket_path)
    if _can_connect(socket_path):
        return
    parent = os.path.dirname(socket_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(socket_path + ".lock", "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        if _can_connect(socket_path):  # another caller won the race
            return
        log_path = os.path.join(parent, "sidecar.log")
        log_fh = open(log_path, "ab")
        env = dict(os.environ)
        # Ensure the detached child can import ``agentic`` regardless of how the
        # host process was launched (uv venv / editable install / raw src path).
        src = str(Path(__file__).resolve().parents[2])  # <repo>/src
        env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
        subprocess.Popen(
            [sys.executable, "-m", "agentic.tools.evfs_sidecar", socket_path],
            start_new_session=True, stdout=log_fh, stderr=log_fh, env=env,
        )
        deadline = time.monotonic() + SPAWN_TIMEOUT_S
        while time.monotonic() < deadline:
            if _can_connect(socket_path):
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"evfs sidecar did not answer on {socket_path} within "
            f"{SPAWN_TIMEOUT_S}s (see {log_path})"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m agentic.tools.evfs_sidecar <socket_path>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
