#!/usr/bin/env python3
"""Pre-download model weights for offline-first deployment.

Two destinations:

* **HF cache** (``~/.cache/huggingface/hub/``) — GLiNER NER backbone.
  GLiNER's tokenizer probes the HF Hub for the base-model repo even on
  a local-path load (see ``config.shared._gliner_cached`` notes), so
  the standard HF cache layout is what its offline mode expects.
* **STORAGE_PATH/models/** — Qwen3-Reranker and Qwen3-Embedding.
  Loaded via plain ``AutoModel*.from_pretrained(local_dir)`` so a
  fully-materialised snapshot in the project's storage volume keeps
  the models with the rest of the data (faiss / bm25 / paddle_ocr
  caches) instead of in a per-user directory.

Defaults to the ``hf-mirror.com`` mirror for HuggingFace traffic;
override with ``--endpoint`` or ``HF_ENDPOINT``.

Usage::

    uv run python download_models.py
    uv run python download_models.py --endpoint https://huggingface.co
    uv run python download_models.py --force
    uv run python download_models.py urchade/gliner_small-v2.1   # different repo
"""
import argparse
import os
from pathlib import Path
from typing import Iterable, List

from huggingface_hub import snapshot_download


DEFAULT_ENDPOINT = "https://hf-mirror.com"


def _project_storage_models() -> Path:
    """Resolve ``STORAGE_PATH/models`` without depending on settings.py.

    ``settings`` imports a large config tree (RAG / agent / web defaults);
    keeping ``download_models.py`` independent lets it run on a fresh
    deploy where Python deps for those modules might not yet be present.
    Mirrors the convention in ``src/config/settings.py::models_root``.
    """
    storage = os.environ.get("STORAGE_PATH") or "local_storage"
    storage_path = Path(storage)
    if not storage_path.is_absolute():
        storage_path = Path(__file__).parent / storage_path
    return storage_path / "models"


# Models that go into the HF cache (~/.cache/huggingface/hub/). GLiNER's
# tokenizer expects this layout because its ``_load_tokenizer`` probes
# the Hub for the base-model repo even with a local snapshot path.
HF_CACHE_MODELS: List[str] = [
    "urchade/gliner_multiv2.1",
    # GLiNER multi-v2.1's backbone. ``GLiNER.from_pretrained`` resolves
    # the tokenizer/encoder by *base-model repo id* (not the gliner
    # snapshot path), so under the offline discipline in
    # ``config.shared._gliner_cached`` this must already sit in the HF
    # cache or the load raises LocalEntryNotFoundError on a fresh host.
    "microsoft/mdeberta-v3-base",
]

# Models that go under STORAGE_PATH/models/<basename>. Both are loaded
# by ``AutoModel*.from_pretrained(local_dir)`` with no upstream
# tokenizer probe, so a flat local snapshot is sufficient.
#
# Repo ids are read from the same env vars the loaders use
# (``settings.rerank_model_dir`` / ``settings.embed_model_dir``) so
# switching the runtime model needs only one env var change and this
# script stays in lockstep. Defaults match ``settings``. The embedding
# model is only consumed at runtime when ``EMBEDDING_BACKEND=local``,
# but pre-fetching it unconditionally keeps a fresh deploy / server
# transfer one command away from either backend.
STORAGE_MODELS: List[str] = [
    os.environ.get("RERANK_MODEL_ID") or "Qwen/Qwen3-Reranker-0.6B",
    os.environ.get("EMBED_MODEL_ID") or "Qwen/Qwen3-Embedding-0.6B",
    os.environ.get("VL_EMBED_MODEL_ID") or "Qwen/Qwen3-VL-Embedding-2B",
]


def download_to_hf_cache(repo_id: str, endpoint: str, force: bool = False) -> None:
    """Snapshot ``repo_id`` into the standard HF cache."""
    print(f"[hf-cache:{repo_id}] endpoint={endpoint}", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        endpoint=endpoint,
        force_download=force,
    )
    print(f"[hf-cache:{repo_id}] cached at {local_path}", flush=True)


def download_to_storage(repo_id: str, endpoint: str, force: bool = False) -> None:
    """Snapshot ``repo_id`` into ``STORAGE_PATH/models/<basename>``."""
    target = _project_storage_models() / repo_id.split("/")[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[storage:{repo_id}] endpoint={endpoint} → {target}", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        endpoint=endpoint,
        force_download=force,
        local_dir=str(target),
    )
    print(f"[storage:{repo_id}] materialised at {local_path}", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "repos",
        nargs="*",
        help=(
            "HuggingFace repo ids to download. If omitted, downloads "
            "the project defaults (GLiNER NER + Qwen3-Reranker + "
            "Qwen3-Embedding + Qwen3-VL-Embedding)."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"HF endpoint (default: {DEFAULT_ENDPOINT}, also reads HF_ENDPOINT).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they're already cached.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    endpoint = args.endpoint.rstrip("/")
    print(f"HF endpoint: {endpoint}", flush=True)

    if args.repos:
        # Manual override — route by **basename** match against
        # STORAGE_MODELS. Basename comparison (rather than full
        # ``org/name``) is the right granularity: ``settings.rerank_model_dir``
        # also routes by basename, so a manual download of a Qwen3-Reranker
        # variant ends up where the loader will look for it even if the
        # owner prefix differs (e.g. a private mirror of the same repo).
        storage_basenames = {m.split("/")[-1] for m in STORAGE_MODELS}
        for repo in args.repos:
            if repo.split("/")[-1] in storage_basenames:
                download_to_storage(repo, endpoint, args.force)
            else:
                download_to_hf_cache(repo, endpoint, args.force)
        return 0

    for repo_id in HF_CACHE_MODELS:
        download_to_hf_cache(repo_id, endpoint, args.force)
    for repo_id in STORAGE_MODELS:
        download_to_storage(repo_id, endpoint, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
