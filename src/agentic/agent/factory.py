"""Wire the default acquisition agent.

Centralizing construction here keeps three properties straight:

1. **Shared resources.** ``PageStore`` and the LinearRAG ``GraphPPRChannel``
   are heavy to load. The factory builds one of each and threads them
   into every tool that needs them, so we don't double-load nor have
   tools accidentally read stale snapshots.
2. **Warm-up correctness.** ``BaseAgent.warm_up()`` walks tools by
   registry order. The graph tool's NER warmer is the slowest, so we
   register it last to maximise overlap once warm-up parallelises.
3. **Single entry point.** Tests, the CLI, and notebook scripts import
   :func:`build_default_agent` instead of constructing the parts
   themselves; the construction order is the contract.
"""

import logging
from pathlib import Path
from typing import Optional

from agentic.agent.base import BaseAgent
from agentic.agent.prompts import GRAPH_SYSTEM_PROMPT, PROOF_SYSTEM_PROMPT, SYSTEM_PROMPT
from agentic.agent.proof_agent import ProofAgent
from agentic.tools.acquisition import (
    Bm25SearchTool,
    CodeRunTool,
    GraphExploreTool,
    ListFilesTool,
    PatternSearchTool,
    ReadTool,
    SemanticSearchTool,
    TocTool,
)
from agentic.closure.inventory import InventoryAdapter
from agentic.tools.registry import ToolRegistry
from config.settings import page_assets_root
from model_client import EmbeddingClient, LLMClient, VisualEmbeddingClient
from rag.channels.graph_ppr import GraphPPRChannel
from storage.inventory_store import InventoryStore
from storage.page_store import PageStore


logger = logging.getLogger(__name__)


def build_default_agent(
    *,
    llm_client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    visual_client: Optional[VisualEmbeddingClient] = None,
    page_store: Optional[PageStore] = None,
    inventory: Optional[InventoryStore] = None,
    graph_channel: Optional[GraphPPRChannel] = None,
    page_assets_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    max_loops: int = 12,
    max_token_budget: int = 128_000,
    verbose: bool = False,
) -> BaseAgent:
    """Build a BaseAgent with the eight acquisition tools pre-registered.

    Any kwarg may be supplied to swap a backend (e.g. point to a
    different ``page_assets_dir`` for tests). Defaults pull from
    ``config.settings``.
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory = inventory or InventoryStore(page_store=page_store)

    embedding_client = embedding_client or EmbeddingClient()
    visual_client = visual_client or VisualEmbeddingClient()
    llm_client = llm_client or LLMClient()

    graph_channel = graph_channel or GraphPPRChannel(
        embedding_client=embedding_client,
    )

    registry = ToolRegistry()
    # Order: navigation -> retrieval -> read -> compute. Mirrors the
    # strategy the system prompt teaches the agent.
    registry.register(ListFilesTool())
    registry.register(TocTool(page_store=page_store, inventory=inventory))
    registry.register(
        SemanticSearchTool(
            page_store=page_store,
            embedding_client=embedding_client,
            visual_client=visual_client,
            inventory=inventory,
        )
    )
    registry.register(Bm25SearchTool(page_store=page_store, inventory=inventory))
    registry.register(PatternSearchTool(page_store=page_store, inventory=inventory))
    registry.register(GraphExploreTool(channel=graph_channel, inventory=inventory))
    registry.register(ReadTool(page_store=page_store, inventory=inventory))
    registry.register(CodeRunTool())

    agent = BaseAgent(
        llm_client=llm_client,
        tools=registry,
        system_prompt=system_prompt or SYSTEM_PROMPT,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )
    return agent


def build_proof_agent(
    *,
    llm_client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    visual_client: Optional[VisualEmbeddingClient] = None,
    page_store: Optional[PageStore] = None,
    inventory: Optional[InventoryStore] = None,
    graph_channel: Optional[GraphPPRChannel] = None,
    page_assets_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    max_loops: int = 16,
    max_token_budget: int = 128_000,
    verbose: bool = False,
) -> ProofAgent:
    """Build a ProofAgent with the eight acquisition tools wired in.

    The proof tools (plan_init, gap_propose, claim_ingest, finalize)
    are registered fresh per ``ProofAgent.run`` call against the
    per-run ``ProofSession``, so they are not in the registry built
    here.
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory_store = inventory or InventoryStore(page_store=page_store)

    embedding_client = embedding_client or EmbeddingClient()
    visual_client = visual_client or VisualEmbeddingClient()
    llm_client = llm_client or LLMClient()

    graph_channel = graph_channel or GraphPPRChannel(
        embedding_client=embedding_client,
    )

    acquisition = ToolRegistry()
    acquisition.register(ListFilesTool())
    acquisition.register(TocTool(page_store=page_store, inventory=inventory_store))
    acquisition.register(
        SemanticSearchTool(
            page_store=page_store,
            embedding_client=embedding_client,
            visual_client=visual_client,
            inventory=inventory_store,
        )
    )
    acquisition.register(Bm25SearchTool(page_store=page_store, inventory=inventory_store))
    acquisition.register(PatternSearchTool(page_store=page_store, inventory=inventory_store))
    acquisition.register(GraphExploreTool(channel=graph_channel, inventory=inventory_store))
    acquisition.register(ReadTool(page_store=page_store, inventory=inventory_store))
    acquisition.register(CodeRunTool())

    return ProofAgent(
        llm_client=llm_client,
        acquisition_tools=acquisition,
        inventory=InventoryAdapter(inventory_store),
        page_store=page_store,
        inventory_store=inventory_store,
        system_prompt=system_prompt or PROOF_SYSTEM_PROMPT,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )


def build_graph_agent(
    *,
    llm_client: Optional[LLMClient] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    page_store: Optional[PageStore] = None,
    inventory: Optional[InventoryStore] = None,
    graph_channel: Optional[GraphPPRChannel] = None,
    page_assets_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    max_loops: int = 8,
    max_token_budget: int = 64_000,
    verbose: bool = False,
) -> BaseAgent:
    """Build a BaseAgent specialised for knowledge-graph navigation.

    Tools registered: ``graph_explore`` (LinearRAG entity / passage
    graph) and ``read_page`` (full-text page reader). The graph tool
    surfaces *which* pages are relevant; the reader pulls the actual
    Markdown so the LLM can quote and cite verbatim — this mirrors
    upstream LinearRAG's "PPR retrieve → reader" split.

    The system prompt (see :data:`GRAPH_SYSTEM_PROMPT`) walks the
    model through the three graph_explore modes (entity_lookup /
    neighbors / ppr) and the standard "navigate → read → cite"
    trajectory.

    Defaults are tighter than the multi-tool agent (``max_loops=8``,
    ``max_token_budget=64k``): graph traversals are cheap but the loop
    wastes budget quickly if it meanders. Override per call site.
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory = inventory or InventoryStore(page_store=page_store)

    embedding_client = embedding_client or EmbeddingClient()
    llm_client = llm_client or LLMClient()

    graph_channel = graph_channel or GraphPPRChannel(
        embedding_client=embedding_client,
    )

    registry = ToolRegistry()
    registry.register(GraphExploreTool(channel=graph_channel, inventory=inventory))
    registry.register(ReadTool(page_store=page_store, inventory=inventory))

    return BaseAgent(
        llm_client=llm_client,
        tools=registry,
        system_prompt=system_prompt or GRAPH_SYSTEM_PROMPT,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )
