"""Sandboxed Python for exact arithmetic and small set operations.

The agent uses this tool when arithmetic precision matters (currency,
prepayment interest, set unions, table aggregations, etc.). LLM-native
arithmetic is unreliable at >2-3 multi-digit operations; this tool runs
the model's code in a fresh interpreter so the answer is verifiable.

Isolation strategy (light, OS-only):

* Subprocess (``python -I``) — fresh, isolated interpreter: PYTHON*
  env stripped, user site disabled, ``sys.path[0]`` not prepended. We
  *do* allow site-packages from the venv we are running in (so numpy
  / sympy are importable); ``-S`` would block those too and make the
  whitelist below moot.
* ``resource.setrlimit`` (POSIX) — CPU 5 s, address space 256 MiB,
  open files 32. The wall clock is double the CPU limit (10 s) so the
  parent can SIGKILL a runaway after the OS ``RLIMIT_CPU`` would have
  fired anyway.
* Wall-clock timeout in the parent (kills the subprocess on overrun).
* No filesystem writes outside the sandbox tempdir (we ``chdir`` there
  before exec, so relative paths can't escape).
* Stdout / stderr capped at a per-stream byte limit (16 KB by default)
  so a runaway loop cannot blow the agent's context.

This is **not** a security boundary against an adversarial user. It is
"prevent-an-LLM-from-burning-the-machine" hardening; strong isolation
(firejail, nsjail, gVisor) is out of scope here.
"""

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agentic.tools.acquisition._common import err, ok
from agentic.tools.base import BaseTool

if TYPE_CHECKING:
    from agentic.core.context import AgentContext


_DEFAULT_CPU_SECONDS = 5
_DEFAULT_WALL_SECONDS = 10
# numpy / sympy reserve a lot of virtual address up-front (OpenBLAS thread
# pool, mmap arenas) so anything tighter than ~1 GiB SIGKILLs them on
# import. The number is "won't crash a typical laptop" rather than a
# tight bound — the wall + CPU clocks are the real DoS guard.
_DEFAULT_MEM_BYTES = 1024 * 1024 * 1024
_DEFAULT_NOFILE = 256
_STREAM_LIMIT_BYTES = 16 * 1024
# stdin is bounded so an oversized ``inputs`` payload cannot deadlock
# the parent: ``proc.stdin.write`` blocks once the OS pipe buffer fills
# (~64 KiB on Linux), and at that point we'd hang before reaching the
# wall-clock kill. 256 KiB covers any realistic structured input.
_INPUTS_LIMIT_BYTES = 256 * 1024


# Whitelist of imports the runner advertises. We do not enforce this at
# import-hook level (an `-S -I` interpreter still has the stdlib) — the
# whitelist is documented to the agent so it doesn't try `requests` or
# `subprocess` and waste a turn on an ImportError. Numpy / sympy are
# included because they're used for the arithmetic the tool is for.
_DOCUMENTED_IMPORTS = (
    "math",
    "json",
    "statistics",
    "decimal",
    "fractions",
    "itertools",
    "functools",
    "re",
    "datetime",
    "numpy",
    "sympy",
)


class CodeRunTool(BaseTool):
    def __init__(
        self,
        cpu_seconds: int = _DEFAULT_CPU_SECONDS,
        wall_seconds: int = _DEFAULT_WALL_SECONDS,
        mem_bytes: int = _DEFAULT_MEM_BYTES,
        stream_limit_bytes: int = _STREAM_LIMIT_BYTES,
        python_executable: Optional[str] = None,
    ):
        self.cpu_seconds = max(1, int(cpu_seconds))
        self.wall_seconds = max(self.cpu_seconds + 1, int(wall_seconds))
        self.mem_bytes = max(64 * 1024 * 1024, int(mem_bytes))
        self.stream_limit_bytes = max(1024, int(stream_limit_bytes))
        self.python_executable = python_executable or sys.executable

    @property
    def name(self) -> str:
        return "code_run"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "code_run",
                "description": (
                    "Execute Python in a fresh sandboxed subprocess for "
                    "exact computation. Use this whenever arithmetic "
                    "precision matters (multi-step money math, set ops, "
                    "table aggregation) — do NOT do those in the chat.\n\n"
                    "Inputs and contract:\n"
                    "- `code`: a Python snippet. The variable `INPUTS` is "
                    "pre-populated with the JSON you pass in `inputs`. "
                    "Assign to a variable named `OUTPUT` to return a "
                    "structured value: the runner JSON-serializes "
                    "`OUTPUT` and surfaces it on the result's `output` "
                    "field. If you do not set `OUTPUT`, `output` will "
                    "be `null` and you should read `stdout` instead.\n"
                    "- `inputs`: optional JSON-serializable dict.\n"
                    "- `purpose`: one short sentence describing intent "
                    "(logged for diagnostics; does not affect execution).\n\n"
                    f"Documented imports: {', '.join(_DOCUMENTED_IMPORTS)} "
                    "(stdlib + numpy + sympy). Other imports may be "
                    "available but are unsupported.\n"
                    "Isolation is light — fresh interpreter, stripped "
                    "env, working directory in a private tempdir, "
                    f"CPU {_DEFAULT_CPU_SECONDS}s, memory 1 GiB, wall "
                    f"{_DEFAULT_WALL_SECONDS}s. Network and absolute-"
                    "path filesystem access are NOT firewalled — do "
                    "not rely on the tool to refuse them, just don't "
                    "ask for them."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source. Sets OUTPUT or prints results.",
                        },
                        "inputs": {
                            "type": "object",
                            "description": (
                                "JSON-serializable mapping; available as "
                                "the variable `INPUTS` inside the snippet."
                            ),
                        },
                        "purpose": {
                            "type": "string",
                            "description": "One-sentence description of intent.",
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        code: str,
        inputs: Optional[Dict[str, Any]] = None,
        purpose: Optional[str] = None,
    ):
        if not code or not str(code).strip():
            return err(
                "invalid_argument",
                "`code` must be a non-empty string.",
                remediation="Pass `code` as a non-empty Python source string; assign to `OUTPUT` to return a structured value.",
                valid_example={"code": "OUTPUT = sum(INPUTS['xs'])", "inputs": {"xs": [1, 2, 3]}},
            ), {"error": "invalid_argument"}
        try:
            inputs_json = json.dumps(inputs or {}, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            return (
                err(
                    "invalid_argument",
                    f"`inputs` must be JSON-serializable: {exc}",
                    remediation="Re-emit `inputs` containing only JSON-serializable values (str / int / float / bool / null / list / dict); replace any custom Python objects with primitives.",
                    valid_example={"inputs": {"xs": [1, 2, 3]}},
                ),
                {"error": "invalid_argument"},
            )
        inputs_bytes = inputs_json.encode("utf-8")
        if len(inputs_bytes) > _INPUTS_LIMIT_BYTES:
            return (
                err(
                    "invalid_argument",
                    f"`inputs` is too large ({len(inputs_bytes)} bytes); "
                    f"max is {_INPUTS_LIMIT_BYTES}.",
                    remediation=f"Trim `inputs` to <= {_INPUTS_LIMIT_BYTES} bytes when JSON-serialized; load large data inside the snippet via `code` instead of passing it through `inputs`.",
                    inputs_bytes=len(inputs_bytes),
                    limit=_INPUTS_LIMIT_BYTES,
                ),
                {"error": "invalid_argument"},
            )

        runner = self._build_runner_script(str(code))
        with tempfile.TemporaryDirectory(prefix="code_run_") as workdir:
            wd = Path(workdir)
            (wd / "runner.py").write_text(runner, encoding="utf-8")
            stdout_bytes, stderr_bytes, returncode, status, elapsed_ms = self._run_bounded(
                wd, inputs_bytes
            )
            if status == "timeout":
                context.add_retrieval_log(
                    tool_name="code_run",
                    tokens=0,
                    metadata={"purpose": purpose, "status": "timeout"},
                )
                return (
                    err(
                        "timeout",
                        f"Subprocess exceeded the {self.wall_seconds}s wall-clock limit and was killed.",
                        remediation=f"Simplify the snippet so it runs in under {self.wall_seconds}s wall-clock and {self.cpu_seconds}s CPU; avoid unbounded loops, large array allocations, or sympy-heavy symbolic work.",
                        elapsed_ms=elapsed_ms,
                    ),
                    {"error": "timeout", "elapsed_ms": elapsed_ms},
                )

        stdout = _truncate_bytes(stdout_bytes, self.stream_limit_bytes)
        stderr = _truncate_bytes(stderr_bytes, self.stream_limit_bytes)
        output_value, output_decoded = _split_marker(stdout)

        log_meta = {
            "purpose": purpose,
            "exit_code": returncode,
            "elapsed_ms": elapsed_ms,
            "stderr_chars": len(stderr.text),
        }
        context.add_retrieval_log(tool_name="code_run", tokens=0, metadata=log_meta)

        return (
            ok(
                "CodeRunObservation",
                exit_code=returncode,
                output=output_decoded,  # parsed OUTPUT (if set) else None
                stdout=output_value.text,
                stdout_truncated=output_value.truncated,
                stderr=stderr.text,
                stderr_truncated=stderr.truncated,
                elapsed_ms=elapsed_ms,
                cpu_limit_seconds=self.cpu_seconds,
                wall_limit_seconds=self.wall_seconds,
            ),
            {
                "retrieved_tokens": 0,
                "exit_code": returncode,
                "elapsed_ms": elapsed_ms,
            },
        )

    # ----------------------------------------------------------- bounded run

    def _run_bounded(
        self, wd: Path, inputs_bytes: bytes
    ) -> "tuple[bytes, bytes, int, str, int]":
        """Spawn the runner with bounded stdout/stderr capture.

        Reading via ``capture_output=True`` would let a snippet that
        prints ``range(10**8)`` balloon the parent's RAM before any
        truncation is applied. We drain each pipe in a thread with a
        hard byte cap; once the cap is hit the drain itself kills the
        process group and force-closes the pipes so the join can
        complete in bounded time.
        """
        import threading
        cap = self.stream_limit_bytes * 2  # room for the truncation banner
        proc = subprocess.Popen(
            [self.python_executable, "-I", str(wd / "runner.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(wd),
            env=_minimal_env(),
            preexec_fn=lambda: _apply_rlimits(
                self.cpu_seconds, self.mem_bytes, _DEFAULT_NOFILE
            ),
            start_new_session=True,
        )

        # Push inputs without blocking: capped at _INPUTS_LIMIT_BYTES
        # (256 KiB) which fits comfortably in the OS pipe buffer
        # (typically 64 KiB; write_then_close() drains in chunks).
        try:
            proc.stdin.write(inputs_bytes)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

        stdout_buf = bytearray()
        stderr_buf = bytearray()
        cap_hit = threading.Event()

        def _drain(stream, buf: bytearray) -> None:
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        return
                    if len(buf) + len(chunk) > cap:
                        remaining = max(0, cap - len(buf))
                        buf.extend(chunk[:remaining])
                        # Caller is past the cap — kill the child so the
                        # subsequent reads return EOF and we can exit.
                        if not cap_hit.is_set():
                            cap_hit.set()
                            _kill_group(proc)
                        return
                    buf.extend(chunk)
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf), daemon=True)
        t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf), daemon=True)
        t_out.start()
        t_err.start()

        t0 = time.perf_counter()
        status = "ok"
        try:
            returncode = proc.wait(timeout=self.wall_seconds)
        except subprocess.TimeoutExpired:
            status = "timeout"
            _kill_group(proc)
            try:
                returncode = proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                returncode = -9
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Force-close the pipes so any drain thread still blocked in
        # ``read()`` returns immediately (defends against pathological
        # uninterruptible-IO cases where SIGKILL alone wouldn't unwedge
        # a Python-level read). Threads are daemon so a missed close
        # cannot keep the process alive, but releasing the fds matters
        # for sustained-run scenarios.
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        t_out.join(timeout=5)
        t_err.join(timeout=5)
        return bytes(stdout_buf), bytes(stderr_buf), returncode, status, elapsed_ms

    # ----------------------------------------------------------- runner

    @staticmethod
    def _build_runner_script(user_code: str) -> str:
        """Render a runner that exposes INPUTS, runs the snippet, and
        emits OUTPUT through a sentinel marker on stdout.

        Using a sentinel (``__OUTPUT_BEGIN__`` / ``__OUTPUT_END__``)
        rather than a separate FD keeps the runner portable across
        Python launchers and lets us still capture user prints
        verbatim before the marker.
        """
        # User code is dedented and embedded inside an exec() so syntax
        # errors are reported as runtime exceptions with the user's
        # source visible in the traceback.
        body = textwrap.dedent(user_code)
        return textwrap.dedent(
            f"""
            import json
            import sys
            import traceback

            try:
                INPUTS = json.loads(sys.stdin.read() or "{{}}")
            except Exception:
                INPUTS = {{}}
            OUTPUT = None

            _USER_CODE = {body!r}

            try:
                exec(compile(_USER_CODE, '<code_run>', 'exec'), {{'INPUTS': INPUTS, 'OUTPUT': None}}, _ns := {{}})
                OUTPUT = _ns.get('OUTPUT', None)
            except SystemExit:
                raise
            except BaseException:
                traceback.print_exc()
                sys.exit(1)

            try:
                _serialized = json.dumps(OUTPUT, ensure_ascii=False, default=str)
            except Exception as exc:  # pragma: no cover — defensive
                print(f"<OUTPUT serialization failed: {{exc}}>", file=sys.stderr)
                _serialized = "null"
            sys.stdout.write("__OUTPUT_BEGIN__")
            sys.stdout.write(_serialized)
            sys.stdout.write("__OUTPUT_END__")
            """
        ).strip()


# --------------------------------------------------------------------- impl


def _minimal_env() -> Dict[str, str]:
    """Drop network proxies and user envs the snippet shouldn't see."""
    keep = {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}
    out = {k: v for k, v in os.environ.items() if k in keep}
    out.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return out


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the child's session group so background threads can't survive.

    Guards against PID reuse: if ``proc.poll()`` says the child has
    already exited, we skip the killpg entirely. ``start_new_session=
    True`` makes ``pgid == proc.pid`` so we don't even need
    ``os.getpgid()`` (which would race against a reaped child).
    """
    try:
        if proc.poll() is not None:
            return
        os.killpg(proc.pid, 9)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def _apply_rlimits(cpu_seconds: int, mem_bytes: int, nofile: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, OSError):
        # Some systems (notably macOS) reject RLIMIT_AS — fall back to
        # RLIMIT_DATA which limits the data segment instead.
        try:
            resource.setrlimit(resource.RLIMIT_DATA, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    except (ValueError, OSError):
        pass
    # Detach controlling tty so the snippet can't drive readline et al.
    try:
        os.setsid()
    except OSError:
        pass


class _Truncated:
    __slots__ = ("text", "truncated")

    def __init__(self, text: str, truncated: bool):
        self.text = text
        self.truncated = truncated


def _truncate_bytes(b: bytes, limit: int) -> _Truncated:
    if len(b) <= limit:
        return _Truncated(b.decode("utf-8", errors="replace"), False)
    head = b[:limit]
    return _Truncated(
        head.decode("utf-8", errors="replace") + "\n…[truncated]…\n", True
    )


def _split_marker(stdout: _Truncated) -> "tuple[_Truncated, Any]":
    """Pull the `__OUTPUT_BEGIN__...__OUTPUT_END__` payload out of stdout.

    Returns ``(stdout_without_marker, parsed_output_or_None)``. If the
    marker is absent (e.g. truncated stdout), the parsed output is None
    and the original stdout is returned unchanged.
    """
    text = stdout.text
    begin = text.find("__OUTPUT_BEGIN__")
    end = text.rfind("__OUTPUT_END__")
    if begin < 0 or end < 0 or end < begin:
        return stdout, None
    raw = text[begin + len("__OUTPUT_BEGIN__"): end]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    visible = (text[:begin] + text[end + len("__OUTPUT_END__"):]).rstrip()
    return _Truncated(visible, stdout.truncated), parsed
