#!/usr/bin/env python3
"""Pre-download model weights into the standard HuggingFace cache.

GLiNER's ``from_pretrained(repo_id)`` already downloads on demand into
``~/.cache/huggingface/hub/`` the first time it runs, but pre-downloading
at deploy time keeps the lifespan startup latency below the FastAPI
worker's first health-check, and surfaces network errors before user
traffic arrives.

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
from typing import Iterable, List

from huggingface_hub import snapshot_download


DEFAULT_ENDPOINT = "https://hf-mirror.com"

# Models the project depends on at runtime. GLiNER multi-v2.1 is the
# open-set NER backbone (mT5, ~1.1 GB FP32 / ~0.6 GB FP16) used by
# every ingest + the PPR seeding path. Add to this list as new local
# components come online.
DEFAULT_MODELS: List[str] = [
    "urchade/gliner_multiv2.1",
]


def download_repo(repo_id: str, endpoint: str, force: bool = False) -> None:
    """Snapshot the repo into the standard HF cache."""
    print(f"[{repo_id}] endpoint={endpoint}", flush=True)
    local_path = snapshot_download(
        repo_id=repo_id,
        endpoint=endpoint,
        force_download=force,
    )
    print(f"[{repo_id}] cached at {local_path}", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "repos",
        nargs="*",
        help=(
            "HuggingFace repo ids to download. "
            "If omitted, downloads the project defaults."
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
    repos = args.repos or DEFAULT_MODELS
    endpoint = args.endpoint.rstrip("/")
    print(f"HF endpoint: {endpoint}", flush=True)
    for repo_id in repos:
        download_repo(repo_id=repo_id, endpoint=endpoint, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
