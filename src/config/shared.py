"""Process-level singleton cache for heavy resources.

The web layer kept hitting OOM on 8 GB hosts because the same heavy
resource (GLiNER multi-v2.1 ~1 GB, faiss EmbeddingStore ~500 MB,
tiktoken encoder ~50 MB × 6 callsites) was instantiated independently
by lifespan, ingest, RAG channels and agent tools. This module gives
those resources one process-level cache that every callsite can share.

Why not just sprinkle ``@functools.lru_cache`` at each callsite?

* Centralises the contract documentation: each helper says exactly
  what guarantees the cached object provides (thread-safe? mutable?
  needs ``refresh()`` after disk write?).
* Forces a stable cache key (``str`` paths, not ``Path``) so two
  callsites that resolve the same on-disk artifact actually hit the
  cache instead of getting separate handles.
* Single ``clear_caches()`` for tests / hot-reload paths.

Concurrency model:

* Each cached factory is wrapped by an **outer per-key lock** so the
  first-call load runs only once even under concurrent miss. Plain
  ``functools.lru_cache`` only locks the cache dict — two threads
  simultaneously missing the same key both run the factory body, then
  one wins the cache write while the other's expensive work is
  thrown away. For large model loads or 500 MB faiss reads, that
  duplication is exactly the OOM scenario this module exists to
  prevent. The pattern: ``with _LOCKS.get_or_create(key): return
  _cached(key)``.
* Cached objects are shared across threads. Model handles and faiss
  ``read_index`` handles are thread-safe for read; ``EmbeddingStore``
  serialises read+write internally via its own per-instance lock so
  ingest writers and query readers can't race.
* Per-request user state (session id, auth, query body) is NEVER cached
  — that lives in the request scope. Only the resource itself, which
  is stateless after construction, lives in this cache.
"""
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, get_args


# Closed enumeration of session profiles. Adding a new client?
# Extend this Literal — type-checkers (mypy / pyright / IDE) will
# refuse a typo at the callsite, which is the bug class the previous
# bare-string design enabled (e.g. "chat_no_read_retry" vs
# "chat-no-read-retry" silently created two pools).
SessionProfile = Literal[
    "chat-no-read-retry",
    "embedding-default",
    "visual-embedding-default",
    "tavily-tight",
    "paddle-ocr-default",
    "web-fetch-tight",
]

# Runtime guard backing the static ``Literal``. CPython doesn't enforce
# ``Literal`` at runtime — without this set the typo escape hatch is
# still open. ``get_args`` returns the tuple of literal values; we
# materialise to a frozenset for O(1) membership checks. Keep in sync
# with ``SessionProfile`` above (mypy will flag the drift).
_VALID_SESSION_PROFILES: frozenset[str] = frozenset(get_args(SessionProfile))

if TYPE_CHECKING:  # pragma: no cover — types only
    import requests
    import tiktoken
    import torch
    from gliner import GLiNER
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from storage.embedding_store import EmbeddingStore


logger = logging.getLogger(__name__)


# ----------------------------------------------------- per-key lock pool ----

# Per-key lock pool so each unique cache key has its own factory-call
# lock. Without this, a single global lock would serialise *all*
# first-call loads (e.g. a GLiNER model + faiss passage index waiting
# on each other) when in fact they could run concurrently — they touch
# different resources. Per-key locks keep concurrent miss correctness
# while permitting parallelism across keys.
_KEY_LOCK_REGISTRY_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}


def _key_lock(key: str) -> threading.Lock:
    """Get-or-create a ``threading.Lock`` for ``key`` from the registry."""
    with _KEY_LOCK_REGISTRY_LOCK:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


# ----------------------------------------------------------- GLiNER ----


@lru_cache(maxsize=4)
def _gliner_cached(model_id: str) -> "GLiNER":
    """Inner cached loader; the public wrapper adds a per-key lock.

    GPU is mandatory by project policy. We import ``torch`` first to
    surface a clear ``RuntimeError`` on CPU-only hosts before paying the
    HuggingFace download cost. Then we:

    1. Resolve the GLiNER snapshot via
       ``snapshot_download(model_id, local_files_only=True)``. Cache
       miss → one-shot online download for first-time deploys (the
       only network call this function makes; subsequent process
       starts never touch the wire). Operators should pre-warm via
       ``python download_models.py``.
    2. **Force ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` before
       importing ``gliner``**. GLiNER's ``_load_tokenizer`` calls
       ``AutoTokenizer.from_pretrained(config.model_name, …)`` with
       the base-model repo id (``microsoft/mdeberta-v3-base`` for
       gliner_multiv2.1) — *not* the local snapshot path — so even
       a local-path-only load triggers a HuggingFace Hub metadata
       probe (``…/tree/main/additional_chat_templates``). On
       high-latency networks that probe can take up to 60 s per cold
       start, dominating ingest time.
       The env vars must be set *before* transformers is imported the
       first time, because transformers caches the offline flag at
       module import. Our project never imports transformers
       directly, so the ``from gliner import GLiNER`` below is the
       first chance — fine as long as we set the env vars first.
    3. Load the weights from the resolved local path and move to
       cuda fp16 (~0.6 GB VRAM resident, no measurable accuracy
       loss vs fp32).

    GLiNER runs on native PyTorch, so no per-thread backend
    re-activation is needed: PyTorch tensors carry their device with
    them, and any thread's forward pass on a cuda-resident model stays
    on GPU without extra bookkeeping.
    """
    import os

    # CRITICAL: set offline mode BEFORE importing huggingface_hub /
    # transformers / gliner. Both libs read this env var ONCE at
    # module-import time and cache it into a module-level constant
    # (``huggingface_hub.constants.HF_HUB_OFFLINE``,
    # ``transformers.utils.hub._is_offline_mode``); a later
    # ``os.environ.setdefault`` is silently ignored. If we set env
    # after ``from huggingface_hub import snapshot_download`` the
    # constants are already False and the GLiNER tokenizer load
    # below will hit api/models/microsoft/mdeberta-v3-base on every
    # call, incurring the remote-probe timeout (up to 60 s) per cold
    # start on high-latency networks.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GLiNER requires CUDA; CPU inference is unsupported by project "
            "policy. Set up a CUDA-enabled torch build or run on a GPU host."
        )

    try:
        local_dir = snapshot_download(model_id, local_files_only=True)
        logger.debug(
            "_gliner_cached: resolved %s to local snapshot %s", model_id, local_dir
        )
    except LocalEntryNotFoundError:
        # First-time bootstrap: cache miss in offline mode → temporarily
        # flip online for a one-shot download, then restore offline so
        # the subsequent ``from gliner import GLiNER`` and the GLiNER
        # tokenizer load below stay offline. ``importlib.reload`` of
        # the ``huggingface_hub.constants`` module is the documented
        # way to refresh the cached flag after toggling the env var
        # at runtime; there is no public API for this.
        import importlib
        import huggingface_hub.constants as _hf_const

        logger.warning(
            "_gliner_cached: %s not in HF cache; downloading once. "
            "Run `python download_models.py` at deploy time to skip this.",
            model_id,
        )
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        importlib.reload(_hf_const)
        try:
            local_dir = snapshot_download(model_id)
        finally:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            importlib.reload(_hf_const)

    from gliner import GLiNER

    logger.debug("_gliner_cached: loading %s from %s on cuda fp16", model_id, local_dir)
    model = GLiNER.from_pretrained(local_dir).to("cuda").half()
    model.eval()
    return model


def shared_gliner(model_id: str) -> "GLiNER":
    """Return the cached GLiNER model for ``model_id`` (HF repo id).

    First call downloads weights into the standard HF cache and loads
    onto GPU in FP16 (~0.6 GB resident, ~0.7 GB peak during inference);
    subsequent calls are O(1). The model is shared across threads —
    PyTorch ``forward`` is read-only and the model is in ``eval()``
    mode, so concurrent batch inference is safe.

    The cache is keyed by the HF ``repo_id`` string, so two callers
    using the same model id share a single resident copy. Per-key lock
    guarantees the underlying ``from_pretrained`` (~5-10 s on first
    cold-start, even longer on a fresh HF cache) runs exactly once;
    plain ``lru_cache`` would let concurrent miss callers each pull the
    weights, doubling RSS during cold-start which is exactly what the
    OOM-prone 8 GB host scenario fails on.
    """
    with _key_lock(f"gliner:{model_id}"):
        return _gliner_cached(model_id)


# ----------------------------------------------------- Qwen3-Reranker ----

# Cache key is the on-disk model directory (a Path resolved by
# settings.rerank_model_dir()), not a HF repo id — the reranker is
# loaded from a fully-materialised local snapshot under
# STORAGE_PATH/models/<basename>, so two consumers of the same dir get
# the same tokenizer/model handle without a hub probe. Same offline-mode
# discipline as GLiNER: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set
# BEFORE the first transformers import.


@lru_cache(maxsize=2)
def _qwen_reranker_cached(model_dir_str: str) -> "tuple[PreTrainedTokenizerBase, PreTrainedModel, int, int]":
    """Inner cached loader. Returns (tokenizer, model, yes_id, no_id).

    The reranker is a causal LM scored by yes/no logit at the last
    position; we pre-compute the two token ids so the hot path is one
    forward + one tensor slice (no per-call token lookup).
    """
    import os

    # Match GLiNER's offline-before-import discipline — see notes in
    # ``_gliner_cached``. Without this, transformers / huggingface_hub
    # cache the offline flag at import time and the tokenizer load
    # below silently hits the Hub for ``additional_chat_templates``.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen3-Reranker requires CUDA; CPU inference is unsupported by "
            "project policy. Set up a CUDA-enabled torch build or run on a "
            "GPU host."
        )
    if not Path(model_dir_str).is_dir():
        raise FileNotFoundError(
            f"Reranker model dir not found: {model_dir_str}. Run "
            f"`python download_models.py` to fetch Qwen3-Reranker-0.6B."
        )
    logger.debug("_qwen_reranker_cached: loading %s on cuda fp16", model_dir_str)
    tokenizer = AutoTokenizer.from_pretrained(model_dir_str, padding_side="left")
    model = (
        AutoModelForCausalLM.from_pretrained(model_dir_str, torch_dtype=torch.float16)
        .cuda()
        .eval()
    )
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    return tokenizer, model, yes_id, no_id


def shared_qwen_reranker(
    model_dir: Path,
) -> "tuple[PreTrainedTokenizerBase, PreTrainedModel, int, int]":
    """Return the cached Qwen3-Reranker handle for ``model_dir``.

    Returns ``(tokenizer, model, yes_token_id, no_token_id)``. First call
    materialises the model on GPU FP16 (~1.2 GB VRAM resident); the
    handle is shared across threads because the model is in ``eval()``
    mode and the forward path is read-only.
    """
    canonical = str(model_dir.expanduser().resolve(strict=False))
    with _key_lock(f"qwen_reranker:{canonical}"):
        return _qwen_reranker_cached(canonical)


# -------------------------------------------------- Qwen3-Embedding ----

# Same offline-before-import + flat-local-snapshot discipline as the
# reranker: the embedding model is materialised under
# STORAGE_PATH/models/<basename> and loaded with AutoModel (base
# transformer, no LM head) for last-token-pooled sentence embeddings.


@lru_cache(maxsize=2)
def _qwen_embedding_cached(
    model_dir_str: str,
) -> "tuple[PreTrainedTokenizerBase, PreTrainedModel]":
    """Inner cached loader. Returns (tokenizer, base model) on cuda fp16.

    Qwen3-Embedding is a decoder; the sentence vector is the last
    non-pad token's hidden state. Left padding (set here) puts that
    token at index ``-1`` for every row so pooling is a single slice.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen3-Embedding requires CUDA; CPU inference is unsupported by "
            "project policy. Use the API embedding backend (EMBEDDING_BACKEND="
            "api) or run on a GPU host."
        )
    if not Path(model_dir_str).is_dir():
        raise FileNotFoundError(
            f"Embedding model dir not found: {model_dir_str}. Run "
            f"`python download_models.py` to fetch Qwen3-Embedding-0.6B."
        )
    logger.debug("_qwen_embedding_cached: loading %s on cuda fp16", model_dir_str)
    tokenizer = AutoTokenizer.from_pretrained(model_dir_str, padding_side="left")
    model = (
        AutoModel.from_pretrained(model_dir_str, torch_dtype=torch.float16)
        .cuda()
        .eval()
    )
    return tokenizer, model


def shared_qwen_embedding(
    model_dir: Path,
) -> "tuple[PreTrainedTokenizerBase, PreTrainedModel]":
    """Return the cached Qwen3-Embedding handle for ``model_dir``.

    First call materialises the model on GPU FP16; the handle is shared
    across threads because the model is in ``eval()`` mode and the
    forward path is read-only — same contract as
    :func:`shared_qwen_reranker`.
    """
    canonical = str(model_dir.expanduser().resolve(strict=False))
    with _key_lock(f"qwen_embedding:{canonical}"):
        return _qwen_embedding_cached(canonical)


# ------------------------------------------------ Qwen3-VL-Embedding ----

# Multimodal (image+text, shared 2048-d space) embedder. The model
# snapshot ships its own ``scripts/qwen3_vl_embedding.py`` defining
# ``Qwen3VLEmbedder`` (last-token pool + L2-normalize built in); we load
# that rather than re-implementing pooling. Same flat-local-snapshot +
# offline discipline as the other local models. Requires
# ``transformers >= 4.57`` and ``qwen-vl-utils``.


@lru_cache(maxsize=1)
def _qwen_vl_embedding_cached(model_dir_str: str) -> "Any":
    """Inner cached loader. Returns a ready ``Qwen3VLEmbedder`` on cuda fp16."""
    import os
    import sys

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Qwen3-VL-Embedding requires CUDA; CPU inference is unsupported "
            "by project policy. Use VISUAL_EMBEDDING_BACKEND=api or run on a "
            "GPU host."
        )
    model_dir = Path(model_dir_str)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Qwen3-VL-Embedding model dir not found: {model_dir_str}. Run "
            f"`python download_models.py` to fetch Qwen3-VL-Embedding-2B."
        )
    # The embedder class is shipped inside the snapshot; importing it
    # needs the snapshot's ``scripts/`` on sys.path.
    scripts_dir = model_dir / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    logger.debug("_qwen_vl_embedding_cached: loading %s on cuda fp16", model_dir_str)
    from qwen3_vl_embedding import Qwen3VLEmbedder

    return Qwen3VLEmbedder(
        model_name_or_path=model_dir_str,
        torch_dtype=torch.float16,
    )


def shared_qwen_vl_embedding(model_dir: Path) -> "Any":
    """Return the cached ``Qwen3VLEmbedder`` for ``model_dir``.

    First call materialises the 2B model on GPU FP16; shared across
    threads (``eval()``/read-only forward) — same contract as the other
    ``shared_qwen_*`` loaders.
    """
    canonical = str(model_dir.expanduser().resolve(strict=False))
    with _key_lock(f"qwen_vl_embedding:{canonical}"):
        return _qwen_vl_embedding_cached(canonical)


# ----------------------------------------------------------- tiktoken ----


@lru_cache(maxsize=8)
def _tiktoken_cached(model_or_encoding: str) -> "tiktoken.Encoding":
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_or_encoding)
    except KeyError:
        return tiktoken.get_encoding(model_or_encoding)


def shared_tiktoken_encoder(model_or_encoding: str) -> "tiktoken.Encoding":
    """Return the cached tiktoken ``Encoding`` for a model or encoding name.

    Many callsites (``LLMClient``, ``BaseAgent``, ``ProofAgent``,
    ``SemanticSearchTool``, ``ReadTool``, ``BM25SearchTool``) need an
    encoder; one instance is ~50-150 MB resident, so we share. Safe
    because ``tiktoken.Encoding`` is read-only at use time.

    We try ``encoding_for_model`` first (so callers can pass either
    ``"gpt-4o"`` or ``"cl100k_base"``); on failure we fall back to
    ``get_encoding``. Per-key lock for concurrent first-call dedup.
    """
    with _key_lock(f"tiktoken:{model_or_encoding}"):
        return _tiktoken_cached(model_or_encoding)


# ----------------------------------------------------- EmbeddingStore ----


# Registry of every (store_dir, namespace) the process has ever cached
# so ``reload_embedding_stores_from_disk()`` can iterate them. Kept
# separate from the lru_cache because lru_cache's keys aren't
# enumerable through the public API.
_STORE_KEY_LOCK = threading.Lock()
_STORE_KEY_REGISTRY: set[tuple[str, str]] = set()


@lru_cache(maxsize=16)
def _embedding_store_cached(store_dir_str: str, namespace: str) -> "EmbeddingStore":
    """Inner cached factory — see :func:`shared_embedding_store` docstring.

    Concurrency is enforced by the public wrapper's per-key lock; this
    inner function is the canonical place that touches the underlying
    ``EmbeddingStore`` instance and registers the key for the
    out-of-band reload helper.

    Mutability contract: ``EmbeddingStore.add()`` mutates the in-memory
    index under its own per-instance lock, then ``save()`` flushes to
    disk under the same lock. Concurrent readers (queries running on
    the lifespan-cached store) see the new vectors in-memory the moment
    ``add()`` releases — desired, since the file-system save is not
    visible to the cache otherwise.
    """
    from storage.embedding_store import EmbeddingStore

    logger.debug(
        "_embedding_store_cached: opening %s ns=%s", store_dir_str, namespace
    )
    with _STORE_KEY_LOCK:
        _STORE_KEY_REGISTRY.add((store_dir_str, namespace))
    return EmbeddingStore(Path(store_dir_str), namespace=namespace)


def shared_embedding_store(store_dir_str: str, namespace: str) -> "EmbeddingStore":
    """Return the cached ``EmbeddingStore`` for ``(store_dir_str, namespace)``.

    Per-key lock makes the underlying ``faiss.read_index`` +
    ``pd.read_parquet`` happen exactly once per unique key even under
    concurrent first-call. Two threads simultaneously asking for the
    same passage store would otherwise each do a 500 MB read; the
    losing thread's allocation is the OOM trigger we're trying to
    avoid.
    """
    with _key_lock(f"embedding_store:{store_dir_str}:{namespace}"):
        return _embedding_store_cached(store_dir_str, namespace)


def shared_embedding_store_for(store_dir: Path, namespace: str) -> "EmbeddingStore":
    """``Path``-friendly thin wrapper around :func:`shared_embedding_store`.

    Most call sites already hold a ``Path`` from ``config.settings``;
    this helper canonicalises with ``resolve(strict=False)`` so even
    paths that don't exist yet (cold start before any ingest) get a
    stable key. A dir-existence check would TOCTOU-race here: one
    caller might see "not exists" and key by relative path while a
    later caller (after ingest creates the dir) keys by absolute path,
    splitting the cache.
    """
    canonical = str(store_dir.expanduser().resolve(strict=False))
    return shared_embedding_store(canonical, namespace)


# ----------------------------------------------------- requests.Session ----

# Session pool — shared by ``profile`` string key. ``profile`` is a
# caller-chosen identifier ("chat-stream-no-read-retry", "tavily-tight",
# ...); same profile reuses the session (urllib3 connection pool + retry
# config), distinct profiles get distinct sessions.
_SESSION_LOCK = threading.Lock()
_SESSION_POOL: dict[str, "requests.Session"] = {}


def shared_session(
    profile: SessionProfile,
    factory: Callable[[], "requests.Session"],
) -> "requests.Session":
    """Return a process-level shared ``requests.Session`` for ``profile``.

    ``profile`` must be one of the literals declared in
    :data:`SessionProfile` — type-checkers will reject a typo at the
    callsite. Adding a new client? Extend the Literal first.

    ``factory`` is invoked only on cache miss to construct the Session;
    subsequent calls with the same profile return the cached one. The
    factory may set up urllib3 ``Retry`` policy, mount custom adapters,
    etc. Different profiles get independent Sessions so retry tuning
    doesn't bleed across clients.

    ``requests.Session`` is documented thread-safe for concurrent use
    via different requests; the underlying urllib3 connection pool
    handles concurrency.
    """
    # Runtime check, not just static — CPython ignores ``Literal``;
    # without this guard a typo silently opens a second pool. We use
    # ``ValueError`` (not ``assert``) because ``python -O`` strips
    # asserts.
    if profile not in _VALID_SESSION_PROFILES:
        raise ValueError(
            f"unknown shared_session profile: {profile!r}. "
            f"Add it to SessionProfile (config/shared.py) first."
        )
    with _SESSION_LOCK:
        existing = _SESSION_POOL.get(profile)
        if existing is not None:
            return existing
        new_session = factory()
        _SESSION_POOL[profile] = new_session
        return new_session


# ----------------------------------------------------- maintenance ----


def clear_caches() -> None:
    """Drop every cached singleton.

    Production code should never call this — the caches are meant to
    live for the process lifetime, and dropping them mid-request would
    leave dangling references in the running ingest / query path.

    Use cases:

    * Test teardown that needs a fresh load (rare; almost all tests
      can share the cached instances safely).
    * Hot-reload on admin config changes that genuinely require new
      heavy resources (e.g. swapping the GLiNER model id); the route
      should call this and restart any in-flight subscribers.

    GPU memory reclamation: ``_gliner_cached.cache_clear()`` only
    releases the Python references; the PyTorch caching allocator
    keeps the freed CUDA blocks in its pool until ``empty_cache()`` is
    called, and Python's reference graph may still pin tensors through
    a transient frame or weakref unless ``gc.collect()`` walks the
    full cycle. We do both here so that swapping a GLiNER model id
    via the admin route actually returns VRAM to the system.
    """
    _gliner_cached.cache_clear()
    _qwen_reranker_cached.cache_clear()
    _qwen_embedding_cached.cache_clear()
    _qwen_vl_embedding_cached.cache_clear()
    # The outer ``get_cached_rerank_client`` wrapper holds a strong
    # reference to the constructed RerankClient (which pins the tokenizer
    # / model handle in its __init__); clear that too so a model-id
    # swap actually returns VRAM.
    try:
        from model_client.rerank import get_cached_rerank_client

        get_cached_rerank_client.cache_clear()
    except Exception:  # noqa: BLE001
        logger.debug("clear_caches: rerank client clear skipped", exc_info=True)
    # Same rationale for the embedding factory: under the local backend
    # its cached client pins the Qwen3-Embedding handle, so a model-id /
    # backend swap only frees VRAM once this wrapper is dropped too.
    try:
        from model_client.text_embedding import get_cached_embedding_client

        get_cached_embedding_client.cache_clear()
    except Exception:  # noqa: BLE001
        logger.debug("clear_caches: embedding client clear skipped", exc_info=True)
    try:
        from model_client.visual_embedding import get_cached_visual_embedding_client

        get_cached_visual_embedding_client.cache_clear()
    except Exception:  # noqa: BLE001
        logger.debug("clear_caches: visual embedding client clear skipped", exc_info=True)
    _tiktoken_cached.cache_clear()
    _embedding_store_cached.cache_clear()
    with _SESSION_LOCK:
        for s in _SESSION_POOL.values():
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        _SESSION_POOL.clear()
    # GPU reclamation: walk cycles first so any tensor held by a
    # weakref / finalizer gets dropped, then ask the PyTorch caching
    # allocator to release the freed blocks back to the driver.
    # Guarded so a CPU-only host (e.g. CI) doesn't blow up on the
    # ``torch.cuda.*`` calls.
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # noqa: BLE001
        # Torch absent or CUDA in an odd state — best-effort cleanup
        # shouldn't crash the caller (often a test teardown).
        logger.debug("clear_caches: torch.cuda cleanup skipped", exc_info=True)


def reload_embedding_stores_from_disk(
    skip_keys: "set[tuple[str, str]] | None" = None,
) -> None:
    """Force every cached ``EmbeddingStore`` to re-read its on-disk artifacts.

    Powers the ``/admin/refresh-indexes`` out-of-band recovery path:
    when an operator hand-edits ``faiss/`` or restores from backup,
    the in-memory cached stores would otherwise stay frozen at the
    pre-edit snapshot. This iterates over the registered cache keys
    and calls ``store.reload_from_disk()`` (which holds the per-store
    lock so a concurrent query never sees a half-loaded mid-state).

    ``skip_keys`` lets the admin caller exclude stores that another
    upstream component (e.g. ``GraphPPRChannel.reload()``) already
    reloaded for the same operator action — without this the graph
    passage / entity / sentence stores would be reloaded twice per
    refresh, an avoidable disk hit on big corpora. Each entry is a
    ``(canonical_dir_str, namespace)`` pair, matching the registry
    key shape.
    """
    skip = skip_keys or set()
    with _STORE_KEY_LOCK:
        keys = list(_STORE_KEY_REGISTRY)
    for store_dir_str, namespace in keys:
        if (store_dir_str, namespace) in skip:
            continue
        try:
            # Going through the public path avoids re-entering the
            # creator under contention; if cache was cleared we'd just
            # rebuild fresh, which is also acceptable.
            store = shared_embedding_store(store_dir_str, namespace)
            store.reload_from_disk()
        except Exception:  # noqa: BLE001
            logger.exception(
                "reload_embedding_stores_from_disk: failed for %s ns=%s",
                store_dir_str,
                namespace,
            )


def canonical_store_key(store_dir: Path, namespace: str) -> "tuple[str, str]":
    """Build a registry key matching :func:`shared_embedding_store_for`.

    Convenience for callers that want to compute a ``skip_keys`` entry
    without duplicating the canonicalisation logic.
    """
    canonical = str(store_dir.expanduser().resolve(strict=False))
    return canonical, namespace


__all__ = [
    "shared_gliner",
    "shared_qwen_reranker",
    "shared_tiktoken_encoder",
    "shared_embedding_store",
    "shared_embedding_store_for",
    "shared_session",
    "SessionProfile",
    "clear_caches",
    "reload_embedding_stores_from_disk",
    "canonical_store_key",
]
