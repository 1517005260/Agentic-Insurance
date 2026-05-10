"""Process-level singleton cache for heavy resources.

The web layer kept hitting OOM on 8 GB hosts because the same heavy
resource (spaCy zh transformer ~2 GB, faiss EmbeddingStore ~500 MB,
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
  thrown away. For 2 GB spaCy loads or 500 MB faiss reads, that
  duplication is exactly the OOM scenario this module exists to
  prevent. The pattern: ``with _LOCKS.get_or_create(key): return
  _cached(key)``.
* Cached objects are shared across threads. spaCy pipelines and faiss
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
from typing import TYPE_CHECKING, Callable, Literal, get_args


# Closed enumeration of session profiles. Adding a new client?
# Extend this Literal — type-checkers (mypy / pyright / IDE) will
# refuse a typo at the callsite, which is the bug class the previous
# bare-string design enabled (e.g. "chat_no_read_retry" vs
# "chat-no-read-retry" silently created two pools).
SessionProfile = Literal[
    "chat-no-read-retry",
    "embedding-default",
    "visual-embedding-default",
    "rerank-default",
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
    import spacy.language
    import tiktoken

    from storage.embedding_store import EmbeddingStore


logger = logging.getLogger(__name__)


# ----------------------------------------------------- per-key lock pool ----

# Per-key lock pool so each unique cache key has its own factory-call
# lock. Without this, a single global lock would serialise *all*
# first-call loads (e.g. spaCy en + faiss passage waiting on each
# other) when in fact they could run concurrently — they touch
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


# ----------------------------------------------------------- spaCy ----


# Tri-state: None = not probed yet; True/False = result of the first
# successful ``spacy.prefer_gpu()`` (process-wide invariant — once a
# pipeline is loaded on GPU, every later call must keep using GPU ops
# or the loaded model parameters end up on a different device than the
# tensors thinc materialises).
_SPACY_GPU_AVAILABLE: bool | None = None


def _activate_spacy_gpu_for_current_thread() -> None:
    """Make sure thinc's GPU ops are the active backend in *this* thread.

    Why this exists: ``thinc.backends.context_ops`` is a ``ContextVar``.
    ``ThreadPoolExecutor.submit`` (and therefore ``loop.run_in_executor``)
    does **not** propagate the parent thread's contextvar values into
    the worker thread — the worker starts with ``context_ops.get()``
    returning the module default ``None``, which makes
    ``get_current_ops()`` silently fall back to ``NumpyOps`` via
    ``require_cpu()``. From that point on, every input tensor thinc
    materialises (e.g. ``input_ids`` inside ``spacy_curated_transformers``)
    lands on CPU, but the transformer's ``nn.Embedding`` weight is still
    on ``cuda:0`` from the main-thread load — yielding the
    ``Expected all tensors to be on the same device`` ``RuntimeError``
    we hit during concurrent reingest.

    The fix is to re-assert the GPU ops on the calling thread every time
    a cached pipeline is handed out. ``spacy.prefer_gpu()`` is idempotent
    and cheap on the hot path (it short-circuits when the requested ops
    type is already current), and crucially it's the same call the cold
    load uses — so the main thread and any worker thread end up with
    the exact same backend selection.

    No-ops once ``_SPACY_GPU_AVAILABLE`` was probed and came back False
    (e.g. CPU-only host); we don't want to retry GPU activation per call
    and pay the cupy-import cost on every hand-out.
    """
    if _SPACY_GPU_AVAILABLE is not True:
        return
    import spacy

    spacy.prefer_gpu()


@lru_cache(maxsize=8)
def _spacy_cached(model_path: str) -> "spacy.language.Language":
    """Inner cached loader; the public wrapper adds a per-key lock.

    GPU activation is self-contained: ``torch`` is imported first so its
    bundled ``libcudart.so.12`` is dlopen'd into the process — without
    this preload, ``cupy`` cannot find the runtime even when installed,
    and ``thinc.has_cupy`` silently returns False (we lose the GPU path
    without any error). After torch is in, ``spacy.prefer_gpu()`` flips
    thinc's backend to cupy if a usable GPU exists, transparently
    falling back to CPU otherwise (idempotent across calls).

    On a 6 GB RTX 3060 with EN+ZH ``_trf`` resident, GPU mode trims
    Python RSS by ~800 MB (transformer weights + activations live in
    VRAM instead of host) and speeds NER 3-4×. Entity output is
    bit-identical to CPU mode, so the algorithm contract is unchanged.

    The boolean returned by ``prefer_gpu`` is captured into the module
    flag ``_SPACY_GPU_AVAILABLE`` so :func:`shared_spacy` can re-assert
    GPU ops in worker threads without re-probing cupy on every call
    (see :func:`_activate_spacy_gpu_for_current_thread`).
    """
    import torch  # noqa: F401  — preload libcudart for cupy
    import spacy

    global _SPACY_GPU_AVAILABLE
    _SPACY_GPU_AVAILABLE = bool(spacy.prefer_gpu())
    logger.debug(
        "_spacy_cached: loading %s (gpu=%s)", model_path, _SPACY_GPU_AVAILABLE
    )
    return spacy.load(model_path)


def shared_spacy(model_path: str) -> "spacy.language.Language":
    """Return the cached spaCy ``Language`` for ``model_path``.

    First call loads from disk (~2-5 s for ``zh_core_web_trf``, ~1-2 GB
    resident); subsequent calls are O(1). The pipeline is shared across
    threads — spaCy ``nlp.pipe`` / ``nlp(...)`` is documented thread-safe
    for read-only inference (the underlying transformer's forward pass
    is read-only).

    The cache is keyed by the **string** path so callers that pass
    ``Path`` objects must call ``str(path)`` first; we don't accept
    ``Path`` because ``lru_cache`` would treat ``Path("a")`` and
    ``Path("./a")`` as different keys even though they load the same
    model. Force the caller to canonicalise.

    Concurrency: the per-key lock guarantees the inner factory body
    (``spacy.load``) is called at most once per ``model_path``. Plain
    ``functools.lru_cache`` would let two concurrent miss callers
    each run ``spacy.load`` and then race for the cache slot, doubling
    GPU/CPU memory peaks during cold-start.

    Per-thread GPU ops re-activation: when the model was loaded on GPU
    in the main thread, every subsequent caller — including worker
    threads spawned by ``loop.run_in_executor`` — must observe ``CupyOps``
    as the current thinc backend, otherwise input tensors get
    materialised on CPU while the model parameters live on cuda:0 (see
    :func:`_activate_spacy_gpu_for_current_thread` for the full
    diagnosis). We therefore re-assert GPU ops on each hand-out; the
    call is cheap on the hot path and the only correct place to do it
    (any caller-side opt-in would be silently forgotten next time we
    add a new ingest entry point).
    """
    with _key_lock(f"spacy:{model_path}"):
        nlp = _spacy_cached(model_path)
    _activate_spacy_gpu_for_current_thread()
    return nlp


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

    Six callsites in the codebase (``LLMClient``, ``BaseAgent``,
    ``ProofAgent``, ``SemanticSearchTool``, ``ReadTool``, ``BM25SearchTool``)
    each used to hold their own encoder, costing ~50-150 MB resident
    memory per instance. Sharing is safe because ``tiktoken.Encoding``
    is read-only at use time.

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
    stable key — the file-existence check we used to do had a TOCTOU
    race where the first caller might see "not exists" and store by
    relative path while a second caller (after ingest created the dir)
    saw "exists" and stored by absolute path, splitting the cache.
    """
    canonical = str(store_dir.expanduser().resolve(strict=False))
    return shared_embedding_store(canonical, namespace)


# ----------------------------------------------------- requests.Session ----

# Session 池——按 ``profile`` 字符串 key 共享。``profile`` 是 caller
# 自己定的标识符（"chat-stream-no-read-retry" / "tavily-tight" 等），
# 同 profile 复用 session（urllib3 连接池 + retry 配置），不同 profile
# 各自一份。
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
      heavy resources (e.g. swapping the spaCy model path); the route
      should call this and restart any in-flight subscribers.
    """
    _spacy_cached.cache_clear()
    _tiktoken_cached.cache_clear()
    _embedding_store_cached.cache_clear()
    with _SESSION_LOCK:
        for s in _SESSION_POOL.values():
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        _SESSION_POOL.clear()


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
    "shared_spacy",
    "shared_tiktoken_encoder",
    "shared_embedding_store",
    "shared_embedding_store_for",
    "shared_session",
    "SessionProfile",
    "clear_caches",
    "reload_embedding_stores_from_disk",
    "canonical_store_key",
]
