"""Wire the default acquisition agent and the Phase-4 proof agent.

Centralizing construction here keeps three properties straight:

1. **Shared resources.** ``PageStore`` and the LinearRAG ``GraphPPRChannel``
   are heavy to load. The factory builds one of each and threads them
   into every tool that needs them, so we don't double-load nor have
   tools accidentally read stale snapshots.
2. **Warm-up correctness.** ``BaseAgent.warm_up()`` walks tools by
   registry order. The graph tool's NER warmer is the slowest, so we
   register it last to maximise overlap if a future change parallelises
   the warm-up walk.
3. **Single entry point.** Tests, the CLI, and notebook scripts all
   import :func:`build_default_agent` (acquisition baseline) or
   :func:`build_proof_agent` (gate-controlled) instead of constructing
   the parts themselves; the construction order is the contract.
"""

import logging
from pathlib import Path
from typing import Optional

from agentic.agent.base import BaseAgent
from agentic.agent.proof_agent import ProofAgent
from agentic.agent.prompts import PROOF_SYSTEM_PROMPT, SYSTEM_PROMPT
from agentic.proof import Plant
from agentic.tools.acquisition import (
    Bm25SearchTool,
    CodeRunTool,
    GraphExploreTool,
    ListFilesTool,
    PatternSearchTool,
    ReadPageTool,
    SemanticSearchTool,
    TocTool,
)
from agentic.tools.proof import (
    AnswerFinalizeTool,
    EvidenceIngestTool,
    ObligationChallengeTool,
    ObligationCreateTool,
    ObligationDecomposeTool,
)
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
    registry.register(ReadPageTool(page_store=page_store))
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
    plant: Optional[Plant] = None,
    page_assets_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    max_loops: int = 24,
    max_token_budget: int = 128_000,
    stall_window: int = 4,
    verbose: bool = False,
) -> ProofAgent:
    """Build a ProofAgent with all 13 tools (8 acquisition + 5 proof).

    Shares the acquisition stack with :func:`build_default_agent` —
    PageStore, InventoryStore, and the GraphPPRChannel are constructed
    once and threaded into both proof tools (via the plant) and
    acquisition tools (via their constructors).
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory = inventory or InventoryStore(page_store=page_store)

    embedding_client = embedding_client or EmbeddingClient()
    visual_client = visual_client or VisualEmbeddingClient()
    llm_client = llm_client or LLMClient()
    graph_channel = graph_channel or GraphPPRChannel(embedding_client=embedding_client)

    plant = plant or Plant(inventory=inventory)

    registry = ToolRegistry()
    # Acquisition tools first — same order as the BaseAgent factory so
    # warm-up walks them in the strategy-prompt sequence.
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
    registry.register(ReadPageTool(page_store=page_store))
    registry.register(CodeRunTool())

    # Proof tools — created with the plant.
    registry.register(ObligationCreateTool(plant=plant))
    registry.register(ObligationDecomposeTool(plant=plant))
    registry.register(ObligationChallengeTool(plant=plant))
    registry.register(EvidenceIngestTool(plant=plant))
    # answer_finalize gets a budget_check closure so it can return
    # ABSTAIN instead of REJECT when budget is exhausted. The agent
    # also abstains on its own when budget runs out, but answering
    # truthfully through the tool keeps the LLM-visible state coherent.
    registry.register(AnswerFinalizeTool(plant=plant))

    agent = ProofAgent(
        llm_client=llm_client,
        tools=registry,
        plant=plant,
        system_prompt=system_prompt or PROOF_SYSTEM_PROMPT,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        stall_window=stall_window,
        verbose=verbose,
    )
    return agent
