#!/usr/bin/env python3
"""Launch a local vLLM OpenAI-compatible server for the QA generator.

Production-faithful complement of the API CHAT path: the agent's
``LLMClient`` is a plain OpenAI-compatible HTTP client, so once this
server is up and ``CHAT_API_BASE_URL`` / ``CHAT_MODEL`` / ``CHAT_API_KEY``
point at it, the rest of the codebase is unchanged.

The CLI defaults encode the configuration that was validated against
the real graph-nav agent on Qwen3-8B (bf16):

* ``--max-model-len 40960`` — Qwen3-8B's full context window. The agent
  emits the full accumulated transcript on every turn, so a tight cap
  (e.g. 16384) ends up smaller than ``LLMClient``'s ``max_tokens`` +
  prompt and vLLM 400s with ``This model's maximum context length is X``.
* ``--gpu-memory-utilization 0.80`` — leaves ~6 GB on a 32 GB card for
  the answer-time co-residents (GLiNER ~0.6 GB, local Qwen3 text
  embedding ~0.6 GB, their torch / CUDA contexts).
* ``--enable-auto-tool-choice --tool-call-parser hermes`` — the agent
  sends ``tool_choice="auto"``; without an explicit parser vLLM 400s
  with ``"auto" tool choice requires --enable-auto-tool-choice and
  --tool-call-parser to be set``. Hermes is the documented Qwen3
  parser (Qwen3-Instruct emits ``<tool_call>{json}</tool_call>``).
* ``VLLM_USE_FLASHINFER_SAMPLER=0`` — the FlashInfer top-k/top-p
  sampling op JIT-compiles a CUDA kernel at first use and requires
  ``ninja``; the native PyTorch sampler is numerically equivalent and
  has no toolchain dependency.
* ``--served-model-name Qwen3-8B gen-local`` — both names route to the
  same model. ``Qwen3-8B`` is the production name (``LLMClient`` mutes
  Qwen3 thinking via ``chat_template_kwargs.enable_thinking=False`` when
  the model string matches "qwen"); ``gen-local`` is provided as a
  thinking-on alias for ad-hoc diagnostics (no auto-mute).

Why a separate venv: vLLM is intentionally NOT a pyproject dependency. It
transitively requires ``torchvision==0.26`` (→ ``torch 2.11``), which
collides with the base env's ``torchvision==0.25`` / ``torch 2.10`` pin
needed by ``qwen-vl-utils``. uv resolves the lockfile to satisfy ALL
extras simultaneously, so even adding vllm as an opt-in extra makes
``uv sync --extra dev`` unsolvable. The sanctioned install path is a
DEDICATED venv that lives outside the project's dependency graph; this
script ``os.execvp``'s the vllm binary FROM THAT VENV (auto-detected via
``Path(sys.executable).parent / "vllm"``).

Setup (one-time)::

    # 1. Create the isolated venv (Python 3.12).
    uv venv .venv-local-llm --python 3.12

    # 2. Install vllm into it. Version pin matches what was validated on
    #    the experiment server (CUDA 13 driver + RTX 4080 SUPER 32 GB):
    VIRTUAL_ENV=$(pwd)/.venv-local-llm uv pip install 'vllm>=0.21.0,<0.22'

    # 3. Fetch the Qwen3-8B weights (~16 GB) into STORAGE_PATH/models/:
    uv run python download_models.py --local-llm

Run (every time you want a local generator)::

    VIRTUAL_ENV=$(pwd)/.venv-local-llm uv run python serve_local_llm.py
    # or override defaults:
    VIRTUAL_ENV=$(pwd)/.venv-local-llm uv run python serve_local_llm.py \
        --port 8001 --gpu-memory-utilization 0.70

Then in ``.env`` point the agent at the local server::

    CHAT_API_BASE_URL=http://127.0.0.1:8000/v1
    CHAT_MODEL=Qwen3-8B
    CHAT_API_KEY=local-noauth
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


def _project_storage_models() -> Path:
    """Mirror of ``download_models.py::_project_storage_models``."""
    storage = os.environ.get("STORAGE_PATH") or "local_storage"
    storage_path = Path(storage)
    if not storage_path.is_absolute():
        storage_path = Path(__file__).parent / storage_path
    return storage_path / "models"


DEFAULT_MODEL_DIR = _project_storage_models() / "Qwen3-8B"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Path to the model snapshot (default: {DEFAULT_MODEL_DIR}).",
    )
    p.add_argument(
        "--served-model-name",
        nargs="+",
        default=["Qwen3-8B", "gen-local"],
        help="One or more aliases the server answers to (default: Qwen3-8B gen-local).",
    )
    p.add_argument("--dtype", default="bfloat16", help="Tensor dtype (default: bfloat16).")
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tool-call-parser", default="hermes")
    p.add_argument(
        "--no-tool-calling",
        action="store_true",
        help="Skip --enable-auto-tool-choice + --tool-call-parser (the agent path needs them; off only for ad-hoc text-only use).",
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"[serve_local_llm] model dir not found: {model_path}\n"
            f"  → fetch with: python download_models.py --local-llm",
            file=sys.stderr,
        )
        return 2

    try:
        import vllm  # noqa: F401
    except ImportError:
        print(
            "[serve_local_llm] vllm not installed in this Python environment.\n"
            "  vllm is not a pyproject dep (torchvision pin conflict — see\n"
            "  pyproject.toml policy comment). Install into an isolated venv:\n"
            "    uv venv .venv-local-llm --python 3.12\n"
            "    VIRTUAL_ENV=$(pwd)/.venv-local-llm uv pip install 'vllm>=0.21.0,<0.22'",
            file=sys.stderr,
        )
        return 3

    # FlashInfer top-k/top-p sampler JIT-builds a CUDA kernel that needs
    # ninja; the native sampler is numerically equivalent and avoids the
    # toolchain dependency. See module docstring.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    # Prefer the venv's own ``vllm`` binary so we hit the CLI parser the
    # smoke test exercised; fall back to PATH only if missing.
    vllm_bin = Path(sys.executable).parent / "vllm"
    cmd_head = [str(vllm_bin)] if vllm_bin.exists() else ["vllm"]
    cmd = cmd_head + [
        "serve", str(model_path),
        "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--host", args.host,
        "--port", str(args.port),
        "--served-model-name", *args.served_model_name,
    ]
    if not args.no_tool_calling:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", args.tool_call_parser]

    print("[serve_local_llm] exec:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
