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
from api.services.files import reconcile_after_restart, sweep_orphan_uploads
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
    logger.info("config store loaded (15 keys)")

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
    from model_client import EmbeddingClient, LLMClient, VisualEmbeddingClient
    from rag.channels.graph_ppr import GraphPPRChannel
    from storage.inventory_store import InventoryStore
    from storage.page_store import PageStore
    from config.settings import page_assets_root

    shared_llm = LLMClient()
    shared_embedding = EmbeddingClient()
    shared_visual = VisualEmbeddingClient()
    # Ensure the page_assets dir exists before PageStore touches it —
    # on a fresh deploy with no uploads yet the directory hasn't been
    # created by the ingest pipeline, and PageStore would crash trying
    # to read it. Idempotent.
    page_assets_root().mkdir(parents=True, exist_ok=True)
    shared_page_store = PageStore(page_assets_root())
    shared_inventory = InventoryStore(page_store=shared_page_store)
    shared_graph_channel = GraphPPRChannel(embedding_client=shared_embedding)

    # Pre-warm spaCy NER as part of singleton init. Without this the
    # first /insurance/fraud-ppr/stream (or any agent path that calls
    # graph_explore mode='ppr') triggers ``spacy.load(en_core_web_trf
    # + zh_core_web_trf)`` synchronously inside the request thread —
    # those models pull torch + CUDA libs and add ~2 GB anon heap, which
    # OOM-kills the worker on 8 GB hosts.
    import time
    t0 = time.monotonic()
    try:
        shared_graph_channel._ensure_spacy()
        logger.info(
            "spaCy NER pre-warmed in %.1fs (en + zh transformer pipelines loaded)",
            time.monotonic() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        # Missing model files / disk error → surface a warning but don't
        # abort startup; PPR paths will lazy-load on demand and the
        # warning makes the trade-off visible in the log.
        logger.warning(
            "spaCy NER pre-warm skipped (%s); first PPR call will pay the "
            "cold-load cost (~2 GB heap, ~30s)",
            exc,
        )

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

    _app.state.base_agent = build_default_agent(
        llm_client=shared_llm,
        embedding_client=shared_embedding,
        visual_client=shared_visual,
        page_store=shared_page_store,
        inventory=shared_inventory,
        graph_channel=shared_graph_channel,
    )
    _app.state.proof_agent = build_proof_agent(
        llm_client=shared_llm,
        embedding_client=shared_embedding,
        visual_client=shared_visual,
        page_store=shared_page_store,
        inventory=shared_inventory,
        graph_channel=shared_graph_channel,
    )
    _app.state.graph_agent = build_graph_agent(
        llm_client=shared_llm,
        embedding_client=shared_embedding,
        page_store=shared_page_store,
        inventory=shared_inventory,
        graph_channel=shared_graph_channel,
    )

    # Tavily client + web agent. The Tavily client is fail-soft when
    # TAVILY_API_KEY is missing — :meth:`available` lets routes / the
    # tool short-circuit, so the lifespan still boots cleanly without
    # a key (the regulation / web-rag / web-agent surfaces will then
    # respond 503 / "unavailable" envelopes until the key is set).
    shared_tavily = TavilyClient()
    _app.state.tavily_client = shared_tavily
    if not shared_tavily.available():
        logger.warning(
            "TAVILY_API_KEY missing — web search / regulation surfaces will return "
            "503 until the key is provided in the env"
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
    try:
        yield
    finally:
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
app.include_router(admin_users_routes.router)
app.include_router(graph_routes.router)
app.include_router(insurance_routes.router)
app.include_router(trace_routes.router)
app.include_router(search_routes.router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
