"""FastAPI entrypoint.

Run with::

    PYTHONPATH=src uv run uvicorn api.main:app --reload --port 8000

The lifespan handler creates tables on first startup and seeds a single
admin user (``DEFAULT_ADMIN_USERNAME`` / ``DEFAULT_ADMIN_PASSWORD``) if
no users exist. Routes are registered here so a top-level import sees
the full URL surface.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from api.auth import hash_password
from api.db import SessionLocal, init_db
from api.models import User
from api.routes import admin as admin_routes
from api.routes import admin_users as admin_users_routes
from api.routes import audit as audit_routes
from api.routes import auth as auth_routes
from api.routes import chat as chat_routes
from api.routes import files as files_routes
from api.routes import graph as graph_routes
from api.routes import insurance as insurance_routes
from api.routes import trace as trace_routes
from api.routes import search as search_routes
from api.services.files import (
    reconcile_after_restart,
    register_refresh_hook,
    sweep_orphan_uploads,
    unregister_refresh_hook,
)
from api.services.graph_service import GraphService
from agentic.agent.factory import (
    build_default_agent,
    build_graph_agent,
    build_proof_agent,
    build_web_agent,
)
from config.config_store import ConfigStore
from model_client.web_search import TavilyClient
from rag.pipeline import RAGPipeline
from config.settings import (
    ALLOW_INSECURE_JWT,
    CORS_ORIGINS,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_PASSWORD_IS_DEFAULT,
    DEFAULT_ADMIN_USERNAME,
    JWT_SECRET_IS_DEFAULT,
)


logger = logging.getLogger(__name__)


def _validate_jwt_secret() -> None:
    """Refuse to boot when JWT_SECRET is the placeholder unless explicitly opted-in.

    The sentinel + opt-in flag both live in ``config.settings`` so we
    don't restate the placeholder string here.
    """
    if JWT_SECRET_IS_DEFAULT and not ALLOW_INSECURE_JWT:
        raise RuntimeError(
            "JWT_SECRET is the default placeholder. Generate a real one with "
            "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"` "
            "and put it in .env, or export ALLOW_INSECURE_JWT=1 for local dev."
        )


async def _seed_admin() -> None:
    """Create the bootstrap admin if the users table is empty.

    Idempotent — never overwrites or upgrades an existing user.
    Warns when the password is left at the bootstrap default so
    operators rotate it before exposing to a network.
    """
    async with SessionLocal() as db:
        any_user = (await db.execute(select(User).limit(1))).scalar_one_or_none()
        if any_user is not None:
            return
        admin = User(
            username=DEFAULT_ADMIN_USERNAME,
            password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
            role="admin",
            is_active=1,
        )
        db.add(admin)
        await db.commit()
        if DEFAULT_ADMIN_PASSWORD_IS_DEFAULT:
            logger.warning(
                "seeded default admin '%s' with development password — rotate via .env "
                "DEFAULT_ADMIN_PASSWORD before exposing to a network",
                DEFAULT_ADMIN_USERNAME,
            )
        else:
            logger.info("seeded admin user '%s'", DEFAULT_ADMIN_USERNAME)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _validate_jwt_secret()
    await init_db()
    await _seed_admin()
    # Convert any stuck mid-flight rows from the previous process into
    # 'failed' so the operator can retry. Idempotent — clean restart
    # rewrites zero rows.
    counts = await reconcile_after_restart()
    if counts["files"] or counts["jobs"]:
        logger.warning(
            "post-restart reconcile: marked %d files + %d jobs as failed",
            counts["files"],
            counts["jobs"],
        )
    # Catch the OTHER kind of process-crash residue: blobs the upload
    # route wrote to disk before the matching DB row committed.
    # Idempotent; common case removes zero files.
    upload_counts = await sweep_orphan_uploads()
    if upload_counts["removed"]:
        logger.warning(
            "post-restart upload sweep: removed %d orphan upload(s) (scanned=%d, .part skipped=%d)",
            upload_counts["removed"],
            upload_counts["scanned"],
            upload_counts["skipped_part_files"],
        )
    # Materialize the runtime config store. ``from_app_db`` reads every
    # row in ``app_config`` once and merges over schema defaults, so a
    # fresh DB is observably identical to "no overrides at all". The
    # store is shared by the admin routes (read/write) and the runners
    # (read-only) — admin patches mutate the same object and the next
    # request sees the new value.
    _app.state.config = await ConfigStore.from_app_db(SessionLocal)
    logger.info("config store loaded (34 keys)")

    # Three agent singletons, one per chat session ``agent_kind``.
    # Each instance is per-process state-free (``run()`` builds fresh
    # AgentContext / ProofSession internally), so sharing across
    # requests is safe.
    #
    # Heavy resources (PageStore, embedding/visual clients,
    # GraphPPRChannel) are built ONCE here and threaded into the three
    # agent factories AND the RAG pipeline — the per-component defaults
    # would otherwise duplicate those loads (PageStore scans
    # local_storage/page_assets, GraphPPRChannel mmaps faiss + reads
    # graphml). On 8 GB WSL the duplication is what triggered the OOM
    # during the first /rag/stream smoke; sharing keeps memory + warm
    # time linear AND ensures the graph_service drawer visualizes the
    # SAME channel instance the RAG pipeline used.
    from model_client import (
        LLMClient,
        get_cached_embedding_client,
        get_cached_visual_embedding_client,
    )
    from rag.channels.graph_ppr import GraphPPRChannel
    from storage.inventory_store import InventoryStore
    from storage.page_store import PageStore
    from config.settings import page_assets_root

    shared_llm = LLMClient()
    # Use the cached factories so the lifespan-pinned client objects
    # are byte-identical to what ingest builders / RAG channels reach
    # for via their default-arg fallback. Without this, lifespan held
    # a "version A" instance and ingest got "version B" from the
    # cache — same underlying session pool but two Python wrappers,
    # making the singleton story confusing to reason about.
    shared_embedding = get_cached_embedding_client()
    shared_visual = get_cached_visual_embedding_client()
    # Ensure the page_assets dir exists before PageStore touches it —
    # on a fresh deploy with no uploads yet the directory hasn't been
    # created by the ingest pipeline, and PageStore would crash trying
    # to read it. Idempotent.
    page_assets_root().mkdir(parents=True, exist_ok=True)
    shared_page_store = PageStore(page_assets_root())
    shared_inventory = InventoryStore(page_store=shared_page_store)
    # Materialize the LinearRAG config from admin overrides so the
    # query-time PPR gazetteer (built lazily inside GraphPPRChannel)
    # honours the same literal_backfill_* knobs the admin tunes.
    # Note: hot-reload of these specific knobs requires a backend
    # restart since the gazetteer automaton is built once and cached
    # on first use.
    shared_linear_config = _app.state.config.materialize_linear_rag_config()
    shared_graph_channel = GraphPPRChannel(
        embedding_client=shared_embedding,
        linear_config=shared_linear_config,
    )

    # Pre-warm GLiNER at lifespan so the open-set NER weights land in
    # the GPU once and every subsequent ingest / PPR query reuses the
    # same model via ``shared_gliner``. Without prewarm the first PPR
    # query (or the first ingest) pays the HF download + ``from_pretrained``
    # cost (~5-10 s cold cache, ~3 s warm) on top of its own latency.
    #
    # GPU is mandatory by project policy (CPU inference unsupported); the
    # cached factory raises if CUDA is unavailable. FP16 footprint on a
    # 6 GB RTX 3060 Laptop is ~0.6 GB resident / ~0.7 GB peak during
    # inference, shared across every ingest and query in this process.
    shared_graph_channel._ensure_ner()

    # Build the RAG pipeline AFTER shared resources so it shares the
    # same GraphPPRChannel instance (the new ``graph_channel`` kwarg).
    _app.state.rag_pipeline = RAGPipeline(
        llm=shared_llm,
        embedding_client=shared_embedding,
        visual_client=shared_visual,
        page_store=shared_page_store,
        graph_channel=shared_graph_channel,
    )
    logger.info("RAG pipeline singleton constructed (shared graph channel)")

    # graph_explore tool tunables (entity_lookup_min_sim / gradient)
    # are baked at construction; admin changes require a backend restart.
    # See ConfigStore.graph_explore_kwargs() and the entries' descriptions.
    shared_graph_explore_kwargs = _app.state.config.graph_explore_kwargs()
    _app.state.base_agent = build_default_agent(
        llm_client=shared_llm,
        embedding_client=shared_embedding,
        visual_client=shared_visual,
        page_store=shared_page_store,
        inventory=shared_inventory,
        graph_channel=shared_graph_channel,
        graph_explore_kwargs=shared_graph_explore_kwargs,
    )
    _app.state.proof_agent = build_proof_agent(
        llm_client=shared_llm,
        embedding_client=shared_embedding,
        visual_client=shared_visual,
        page_store=shared_page_store,
        inventory=shared_inventory,
        graph_channel=shared_graph_channel,
        graph_explore_kwargs=shared_graph_explore_kwargs,
    )
    _app.state.graph_agent = build_graph_agent(
        llm_client=shared_llm,
        embedding_client=shared_embedding,
        page_store=shared_page_store,
        inventory=shared_inventory,
        graph_channel=shared_graph_channel,
        graph_explore_kwargs=shared_graph_explore_kwargs,
    )

    # Tavily client + web agent. The Tavily client is fail-soft when
    # TAVILY_API_KEY is missing — :meth:`available` lets routes / the
    # tool short-circuit, so the lifespan still boots cleanly without
    # a key (the web-rag / web-agent chat surfaces will then respond
    # "unavailable" envelopes until the key is set).
    shared_tavily = TavilyClient()
    _app.state.tavily_client = shared_tavily
    if not shared_tavily.available():
        logger.warning(
            "TAVILY_API_KEY missing — chat web mode + web agent will surface "
            "'tavily unavailable' until the key is provided in the env"
        )

    _app.state.web_agent = build_web_agent(
        llm_client=shared_llm,
        tavily_client=shared_tavily,
        config_store=_app.state.config,
    )
    logger.info(
        "base / proof / graph / web agent singletons constructed (shared PageStore + GraphPPRChannel + Tavily)"
    )

    # Web-side facade over the same GraphPPRChannel — no extra mmap;
    # the service is read-only and thread-safe (igraph + EmbeddingStore
    # are immutable post-ingest).
    _app.state.graph_service = GraphService(channel=shared_graph_channel)
    logger.info("graph service constructed")

    # Long-lived singletons capture an empty-disk snapshot at boot
    # (lifespan runs before any PDF is uploaded). After every successful
    # ingest / reingest / delete the bg task calls this hook so the
    # in-memory state matches the on-disk artifacts. Without it
    # toc / graph_explore / read_page / GraphService all silently see
    # the empty-boot state for any file uploaded after process start.
    #
    # The hook also doubles as the implementation of
    # ``POST /admin/refresh-indexes`` — operator-initiated rescue when
    # an out-of-band write to local_storage needs to be picked up.
    def refresh_indexes() -> None:
        """Reload all on-disk-backed singletons + invalidate the
        per-instance derived caches that wrap them. Idempotent.

        Order matters:

        1. Reload the data sources (PageStore re-scans page_assets,
           InventoryStore drops section / passage / table_row caches,
           GraphPPRChannel re-mmaps faiss + re-reads GraphML).
        2. Invalidate the wrappers that snapshot derived views from
           those sources (GraphService passage-meta / cluster /
           sample caches, and each agent's GraphExploreTool
           passage-meta / cluster caches). Without step 2 the wrappers
           keep handing out hashes / clusters that no longer exist
           in the underlying stores.
        """
        shared_page_store.reload()
        shared_inventory.reload()
        # Re-materialise the LinearRAGConfig from the (just-reloaded)
        # config-store snapshot so admin-rotated GLiNER knobs reach
        # the query-side NER on the next request.
        fresh_linear_config = _app.state.config.materialize_linear_rag_config()
        shared_graph_channel.reload(linear_config=fresh_linear_config)
        # Force every process-cached EmbeddingStore that wasn't already
        # reloaded by a higher-level component to re-read from disk.
        # ``GraphPPRChannel.reload()`` above already covered the three
        # graph stores (passage / entity / sentence); skip them here
        # to avoid a second 200 MB read each on big corpora. The
        # remaining dense / visual stores have no per-channel reload
        # hook so they need this helper.
        try:
            from config.settings import (
                faiss_graph_entity_dir,
                faiss_graph_passage_dir,
                faiss_graph_sentence_dir,
            )
            from config.shared import (
                canonical_store_key,
                reload_embedding_stores_from_disk,
            )

            already_reloaded = {
                canonical_store_key(faiss_graph_passage_dir(), "passage"),
                canonical_store_key(faiss_graph_entity_dir(), "entity"),
                canonical_store_key(faiss_graph_sentence_dir(), "sentence"),
            }
            reload_embedding_stores_from_disk(skip_keys=already_reloaded)
        except Exception:
            logger.exception(
                "reload_embedding_stores_from_disk raised; "
                "dense/visual stores may be stale until restart"
            )
        # Wrapper invalidations — must come AFTER channel.reload()
        # because they pull fresh state from the channel on next access.
        try:
            _app.state.graph_service.invalidate_caches()
        except Exception:
            logger.exception("graph_service.invalidate_caches raised")
        for agent_attr in ("base_agent", "proof_agent", "graph_agent"):
            agent = getattr(_app.state, agent_attr, None)
            if agent is None:
                continue
            tool = None
            try:
                tool = agent.tools.get("graph_explore")
            except Exception:
                tool = None
            if tool is not None and hasattr(tool, "invalidate_caches"):
                try:
                    tool.invalidate_caches()
                except Exception:
                    logger.exception(
                        "%s graph_explore.invalidate_caches raised", agent_attr
                    )

    _app.state.refresh_indexes = refresh_indexes
    register_refresh_hook(refresh_indexes)
    logger.info(
        "refresh_indexes hook registered (PageStore + InventoryStore + "
        "GraphPPRChannel + GraphService caches + per-agent graph_explore caches)"
    )
    try:
        yield
    finally:
        # Drop the hook before tearing down singletons so a late ingest
        # task can't call into half-finalized state.
        unregister_refresh_hook()
        # Best-effort close of the long-lived ``requests.Session``
        # objects held by the model clients + the web tool. They are
        # thread-safe but leaking connection pools on shutdown is
        # sloppy under repeated reload cycles (uvicorn dev). Each
        # candidate either IS a session or wraps one as ``_session``.
        # Lookups via ``getattr(default=None)`` so test paths that
        # don't build a particular client don't crash teardown.
        rag_pipeline = getattr(_app.state, "rag_pipeline", None)
        web_agent = getattr(_app.state, "web_agent", None)
        web_fetch_tool = None
        if web_agent is not None:
            try:
                web_fetch_tool = web_agent.tools.get("web_fetch")
            except Exception:
                web_fetch_tool = None
        candidates = [
            getattr(_app.state, "tavily_client", None),
            shared_llm,
            shared_embedding,
            shared_visual,
            getattr(rag_pipeline, "rerank_client", None) if rag_pipeline else None,
            web_fetch_tool,
        ]
        for client in candidates:
            if client is None:
                continue
            sess = getattr(client, "_session", None) or client
            close = getattr(sess, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("client session close raised", exc_info=True)


app = FastAPI(
    title="agentic web API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)
app.include_router(files_routes.router)
app.include_router(audit_routes.router)
app.include_router(chat_routes.router)
app.include_router(admin_routes.router)
app.include_router(admin_routes.admin_actions_router)
app.include_router(admin_users_routes.router)
app.include_router(graph_routes.router)
app.include_router(insurance_routes.router)
app.include_router(trace_routes.router)
app.include_router(search_routes.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
