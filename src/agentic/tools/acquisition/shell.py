"""A sandboxed Unix shell over the raw document corpus — the faithful
Direct-Corpus-Interaction (DCI) baseline.

The agent gets ONE tool, ``shell``, that runs read-only bash commands
(``grep``/``rg``/``find``/``ls``/``sed``/``head``/``tail``/``cat``/``awk`` …)
inside the markdown corpus directory. This is the 2026 "agents don't need a
vector DB, they need a terminal" paradigm (DCI, arXiv 2605.05242; Claude Code /
Vercel grep-over-files) — no embedding, no index, no graph; the agent's only
locator is the shell command it writes.

Isolation (``bwrap``/bubblewrap, when available — the strong path):
  * the corpus dir + the minimal system dirs (``/usr /bin /lib /lib64 /etc``)
    are bind-mounted **read-only**; nothing else is visible (no ``/home``,
    ``/root``, project creds, other storage);
  * ``--unshare-all`` removes the network; the only writable place is a private
    ``tmpfs`` ``/tmp``; ``--die-with-parent`` cleans up on exit.
So the agent literally cannot escape the corpus, write outside tmpfs, or reach
the network — "full bash, scoped to the md folder."

Fallback (no ``bwrap``): run with ``cwd`` pinned to the corpus + a denylist that
rejects writes / in-place edits / network / path-escape, so it stays read-only.
"""

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple, TYPE_CHECKING

from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext

logger = logging.getLogger(__name__)

_TIMEOUT_S = 30
_MAX_OUTPUT_CHARS = 8000
# Read-only system roots the shell tools need to run; deliberately excludes
# /home, /root, /autodl-*, and the project tree so the sandbox sees only the
# corpus + the binaries to read it.
_SYSTEM_BINDS = ("/usr", "/bin", "/lib", "/lib64", "/etc")
_SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
# Where the semantic sidecar's unix socket is bound inside the sandbox. AF_UNIX
# works across the bwrap mount namespace WITHOUT the network namespace, so the
# scripts reach the host-side channels while the sandbox stays hermetic.
_SIDECAR_MOUNT = "/run/evfs_sidecar.sock"
# Fallback-only guard (bwrap makes these harmless, so it is not applied there).
_DENY = re.compile(
    r"(?:^|[\s;&|`$(])\s*(?:rm|rmdir|mv|cp|dd|mkfs|chmod|chown|chgrp|ln|truncate|"
    r"shred|tee|install|mkdir|touch|curl|wget|nc|ncat|ssh|scp|sftp|telnet|sudo|"
    r"su|kill|pkill|reboot|shutdown|mount|umount)\b"
    r"|sed\b[^|;]*-i"      # in-place edit
    r"|>>?"               # write redirect
    r"|\.\./",            # path escape
    re.IGNORECASE,
)


class ShellTool(BaseTool):
    def __init__(self, corpus_root: Path, extra_ro_binds=None, scripts_dirs=None,
                 socket_path=None, timeout_s=_TIMEOUT_S):
        self.corpus_root = Path(corpus_root).resolve()
        self._bwrap = shutil.which("bwrap")
        # Per-command wall clock. The lexical/base shells are sub-second
        # (grep/awk/sed), so the tight default protects against a runaway
        # pattern; the semantic tier's first call may wait on the sidecar's
        # one-time channel load, so it passes a longer ceiling.
        self._timeout_s = int(timeout_s)
        # The semantic sidecar's unix socket on the HOST. When set, it is bound
        # rw into the sandbox at ``_SIDECAR_MOUNT`` and exported as
        # ``EVFS_SIDECAR_SOCK`` — the only channel the otherwise-hermetic
        # semantic scripts have to the host-side retrieval channels.
        self.socket_path = Path(socket_path).resolve() if socket_path else None
        # Extra read-only trees bound at their REAL host path so absolute
        # symlinks inside the corpus resolve (e.g. EvidenceFS
        # ``documents/<doc>/combined.md`` → ``paddle_ocr/<doc>/combined.md``).
        # Without this, bwrap mounts only the corpus and the symlink dangles.
        self.extra_ro_binds = [Path(p).resolve() for p in (extra_ro_binds or [])]
        # The EvidenceFS shell ABI, one dir per capability tier (``lexical`` /
        # ``semantic``). Each is bound read-only at ``/scripts/<tier>`` and
        # prepended to PATH, so the agent invokes the programs by bare name from
        # the FS root exactly as the emitted README/EXAMPLES describe. Empty →
        # a pure DCI shell (no scripts). Listing multiple dirs composes tiers,
        # which is what lets the agent be ablated (lexical-only vs lexical+semantic).
        self.scripts_dirs = [Path(p).resolve() for p in (scripts_dirs or [])]

    @property
    def name(self) -> str:
        return "shell"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "shell",
                "description": (
                    "Run a READ-ONLY bash command in a directory of markdown "
                    "documents (the corpus). No network, no writes. Use it to "
                    "locate and read evidence: `rg`/`grep -rn` for exact terms / "
                    "numbers / clauses, `find`/`ls` to navigate, `sed -n` / "
                    "`head` / `tail` to read a window around a hit, `cat` to read "
                    "a file, pipes (`| head`, `| sort -u`, `| wc -l`) to shape "
                    "output. Paths are relative to the corpus root. Output is "
                    "truncated; narrow with line ranges / head."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "A single read-only bash command line (pipes allowed).",
                        }
                    },
                    "required": ["command"],
                },
            },
        }

    def execute(self, context: "AgentContext", command: str = "", **_: Any) -> Tuple[str, Dict[str, Any]]:
        command = (command or "").strip()
        if not command:
            return err(
                "invalid_argument",
                "`command` must be a non-empty bash command line.",
                valid_example={"command": "rg -n 'surrender charge' --type md"},
            ), {"error": "invalid_argument"}

        argv = self._build_argv(command)
        if argv is None:  # fallback denylist hit
            return err(
                "blocked",
                "Command rejected: this shell is read-only and scoped to the "
                "corpus (no writes, in-place edits, network, or path escape).",
                remediation="Use read-only locators: rg/grep/find/sed -n/head/tail/cat with pipes.",
            ), {"error": "blocked"}

        # In bwrap mode PATH / EVFS_SIDECAR_SOCK are set via --setenv; in the
        # fallback the child inherits this env, so prepend the scripts dirs at
        # their real host path and point the socket var at the host socket.
        env = None
        present = [d for d in self.scripts_dirs if d.exists()]
        if not self._bwrap and (present or self.socket_path):
            env = dict(os.environ)
            if present:
                extra = ":".join(str(d) for d in present)
                env["PATH"] = f"{extra}:{env.get('PATH', '')}"
            if self.socket_path:
                env["EVFS_SIDECAR_SOCK"] = str(self.socket_path)

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                cwd=str(self.corpus_root),
                env=env,
            )
            out = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            return err(
                "timeout",
                f"Command exceeded {self._timeout_s}s. Narrow the search (add literal "
                "anchors, --max-count, or a file/dir scope).",
            ), {"error": "timeout"}
        except Exception as exc:  # noqa: BLE001
            return err("shell_error", f"{type(exc).__name__}: {exc}"), {"error": "shell_error"}

        truncated = len(out) > _MAX_OUTPUT_CHARS
        if truncated:
            out = out[:_MAX_OUTPUT_CHARS] + "\n…[truncated]"
        # Surface stderr only when the command failed and produced no stdout,
        # so a non-zero "no matches" (grep exit 1) stays quiet.
        tail_err = ""
        if exit_code != 0 and not out.strip() and stderr.strip():
            tail_err = stderr.strip()[:600]

        context.add_retrieval_log(
            tool_name="shell",
            tokens=0,
            metadata={"command": command, "exit_code": exit_code, "sandbox": "bwrap" if self._bwrap else "cwd"},
        )
        return (
            ok(
                "ShellObservation",
                command=command,
                exit_code=exit_code,
                stdout=out,
                stderr=tail_err,
                truncated=truncated,
                sandbox="bwrap" if self._bwrap else "cwd-pinned",
            ),
            {"retrieved_tokens": 0, "exit_code": exit_code},
        )

    # ------------------------------------------------------------------ sandbox
    def _build_argv(self, command: str):
        if self._bwrap:
            argv = [self._bwrap]
            for p in _SYSTEM_BINDS:
                if os.path.exists(p):
                    argv += ["--ro-bind", p, p]
            for p in self.extra_ro_binds:
                if p.exists():
                    argv += ["--ro-bind", str(p), str(p)]
            argv += [
                # mount the corpus at a neutral /corpus so the host path
                # (/home/<user>/…) is never revealed and no skeleton dirs leak.
                "--ro-bind", str(self.corpus_root), "/corpus",
            ]
            # Each tier mounted at /scripts/<tier>, prepended to PATH, so the
            # agent runs `find_surface …` by bare name from /corpus.
            present = [d for d in self.scripts_dirs if d.exists()]
            script_mounts = [f"/scripts/{d.name}" for d in present]
            for d, mount in zip(present, script_mounts):
                argv += ["--ro-bind", str(d), mount]
            # Bind the sidecar socket rw (a client needs write access to
            # connect); AF_UNIX needs no net namespace, so the sandbox below
            # can still --unshare-all and stay hermetic.
            if self.socket_path:
                argv += ["--bind", str(self.socket_path), _SIDECAR_MOUNT]

            # Clear the env entirely — the stdlib scripts need nothing but the
            # socket, so no host cred is ever visible. Then set the minimal env.
            argv += ["--clearenv"]
            argv += ["--setenv", "PATH", ":".join(script_mounts + [_SYSTEM_PATH])]
            argv += ["--setenv", "HOME", "/tmp"]
            argv += ["--setenv", "LANG", "C.UTF-8"]
            if self.socket_path:
                argv += ["--setenv", "EVFS_SIDECAR_SOCK", _SIDECAR_MOUNT]
            argv += [
                "--tmpfs", "/tmp",
                "--proc", "/proc",
                "--dev", "/dev",
                "--unshare-all",        # no pid / ipc / net / … namespace
                "--die-with-parent",
                "--chdir", "/corpus",
                "/bin/bash", "-c", command,
            ]
            return argv
        # Fallback: no bwrap → cwd-pinned + read-only denylist.
        if _DENY.search(command):
            return None
        return ["/bin/bash", "-c", command]
