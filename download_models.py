#!/usr/bin/env python3
"""Download model weights into ``STORAGE_PATH/models/``.

Defaults to the ``hf-mirror.com`` mirror for HuggingFace traffic; override
with ``--endpoint`` or the standard ``HF_ENDPOINT`` env var.

The only dep used is ``requests`` (plus ``tqdm`` for the progress bar) — no
``huggingface_hub`` so we keep the model-runtime deps off the project.

Usage::

    uv run python download_models.py                       # download project defaults
    uv run python download_models.py --force               # re-download even if files exist
    uv run python download_models.py spacy/en_core_web_sm  # download a different repo
    uv run python download_models.py --endpoint https://huggingface.co  # use upstream
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import requests
from tqdm import tqdm

# Make `from config.settings import ...` work whether the package is installed
# editably (uv sync) or the script is run from a fresh clone before sync.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config.settings import models_root  # noqa: E402


DEFAULT_ENDPOINT = "https://hf-mirror.com"

# Models the project depends on by default. Add to this list as new local
# components (NER / VLM previewer / etc.) come online.
#
# Chinese coverage: spaCy ships a single Chinese pipeline (zh_core_web_*).
# It targets Simplified Chinese; Traditional Chinese input is supported by
# normalizing Traditional → Simplified upstream (e.g. with OpenCC) before
# running NER, since no separate Traditional spaCy model exists.
DEFAULT_MODELS: List[str] = [
    "spacy/en_core_web_trf",
    "spacy/zh_core_web_trf",
]


def list_repo_files(repo_id: str, endpoint: str, revision: str = "main") -> List[str]:
    """Return every file path in a HuggingFace repo (recursive)."""
    url = f"{endpoint}/api/models/{repo_id}/tree/{revision}?recursive=true"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return [item["path"] for item in resp.json() if item.get("type") == "file"]


def download_file(
    repo_id: str,
    filename: str,
    dest_path: Path,
    endpoint: str,
    revision: str = "main",
    force: bool = False,
) -> None:
    """Stream a single file from HF to ``dest_path``. Skips if already complete."""
    url = f"{endpoint}/{repo_id}/resolve/{revision}/{filename}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size = None
    if dest_path.exists() and not force:
        try:
            head = requests.head(url, allow_redirects=True, timeout=60)
            head.raise_for_status()
            expected_size = int(head.headers.get("Content-Length") or 0)
        except Exception:
            expected_size = None
        if expected_size and dest_path.stat().st_size == expected_size:
            return

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
        with (
            open(tmp_path, "wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                leave=False,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))
        tmp_path.replace(dest_path)


def download_repo(
    repo_id: str,
    dest_dir: Path,
    endpoint: str,
    force: bool = False,
) -> None:
    print(f"[{repo_id}] endpoint={endpoint}")
    print(f"[{repo_id}] dest={dest_dir}")
    files = list_repo_files(repo_id, endpoint)
    print(f"[{repo_id}] {len(files)} file(s)")
    for filename in files:
        download_file(repo_id, filename, dest_dir / filename, endpoint, force=force)
    print(f"[{repo_id}] done")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "repos",
        nargs="*",
        help=(
            "HuggingFace repo ids to download (e.g. spacy/en_core_web_trf). "
            "If omitted, downloads the project defaults."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"HF endpoint (default: {DEFAULT_ENDPOINT}, also reads HF_ENDPOINT).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Override destination root (default: STORAGE_PATH/models/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist with the right size.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repos = args.repos or DEFAULT_MODELS
    dest_root = (args.dest or models_root()).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"Models root: {dest_root}")
    for repo_id in repos:
        subdir = repo_id.split("/")[-1]
        download_repo(
            repo_id=repo_id,
            dest_dir=dest_root / subdir,
            endpoint=args.endpoint.rstrip("/"),
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
