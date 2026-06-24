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
from typing import Any, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config.config_store import ConfigStore
    from model_client.web_search import TavilyClient

from agentic.agent.base import BaseAgent
from agentic.agent.prompts import (
    GRAPH_SYSTEM_PROMPT,
    PROOF_SYSTEM_PROMPT,
    REGEX_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WEB_AGENT_SYSTEM_PROMPT,
)
from agentic.agent.proof_agent import ProofAgent
from agentic.tools.acquisition import (
    Bm25SearchTool,
    CodeRunTool,
    EntityInspectTool,
    GraphChainTool,
    GraphPprTool,
    ListFilesTool,
    PatternSearchTool,
    ReadTool,
    SemanticSearchTool,
    TocTool,
    WebFetchTool,
    WebSearchTool,
)
from agentic.closure.inventory import InventoryAdapter
from agentic.tools.registry import ToolRegistry
from config.settings import page_assets_root
from model_client import (
    EmbeddingClient,
    LLMClient,
    VisualEmbeddingClient,
    get_cached_embedding_client,
    get_cached_visual_embedding_client,
)
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
    max_loops: int = 24,
    max_token_budget: int = 128_000,
    verbose: bool = False,
    graph_explore_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseAgent:
    """Build a BaseAgent with the acquisition tools pre-registered.

    Any kwarg may be supplied to swap a backend (e.g. point to a
    different ``page_assets_dir`` for tests). Defaults pull from
    ``config.settings``.

    ``max_loops=24`` / ``max_token_budget=128_000`` are deliberately
    generous so the agent can fully explore before being force-answered.
    Small-context generators (e.g. vLLM-served Qwen3-8B, 40 960 ctx):
    override ``max_token_budget`` down to ~20 000 AND lower
    ``LLMClient(max_tokens=...)`` in lockstep at the call site.
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory = inventory or InventoryStore(page_store=page_store)

    embedding_client = embedding_client or get_cached_embedding_client()
    visual_client = visual_client or get_cached_visual_embedding_client()
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
    for graph_tool in (GraphPprTool, GraphChainTool, EntityInspectTool):
        registry.register(
            graph_tool(
                channel=graph_channel,
                inventory=inventory,
                **(graph_explore_kwargs or {}),
            )
        )
    registry.register(
        ReadTool(page_store=page_store, inventory=inventory, graph_channel=graph_channel)
    )
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
    max_loops: int = 24,
    max_token_budget: int = 128_000,
    verbose: bool = False,
    graph_explore_kwargs: Optional[Dict[str, Any]] = None,
) -> ProofAgent:
    """Build a ProofAgent with the acquisition tools wired in.

    The proof tools (plan_init, gap_propose, claim_ingest, finalize)
    are registered fresh per ``ProofAgent.run`` call against the
    per-run ``ProofSession``, so they are not in the registry built
    here.
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory_store = inventory or InventoryStore(page_store=page_store)

    embedding_client = embedding_client or get_cached_embedding_client()
    visual_client = visual_client or get_cached_visual_embedding_client()
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
    for graph_tool in (GraphPprTool, GraphChainTool, EntityInspectTool):
        acquisition.register(
            graph_tool(
                channel=graph_channel,
                inventory=inventory_store,
                **(graph_explore_kwargs or {}),
            )
        )
    acquisition.register(
        ReadTool(page_store=page_store, inventory=inventory_store, graph_channel=graph_channel)
    )
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
    max_loops: int = 24,
    max_token_budget: int = 128_000,
    verbose: bool = False,
    graph_explore_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseAgent:
    """Build a BaseAgent specialised for knowledge-graph navigation.

    Tools registered: ``graph_ppr`` (associative page retrieval),
    ``graph_chain`` (relational multi-hop), ``entity_inspect`` (entity
    disambiguation / neighborhood) over the LinearRAG entity / passage
    graph, plus ``read`` (full-text page reader). The graph tools surface
    *which* pages are relevant; the reader pulls the actual Markdown so
    the LLM can quote and cite verbatim — this mirrors upstream
    LinearRAG's "PPR retrieve → reader" split.

    ``max_loops=24`` / ``max_token_budget=128_000``: give navigation room
    to fully traverse the graph on deep (3–4-hop) questions instead of
    terminating against the loop / token cap mid-chain. A tight budget
    starved deep-hop reasoning — the loop hit the cap, force-answered on a
    half-built chain, and abstained; widening to a capable reader's
    context restores deep-hop accuracy. Override DOWN per call site for a
    model whose context is smaller
    (e.g. a vLLM-served Qwen3-8B at 40 960 context − 16 384 reserved
    for output = 24 576 effective input — there ``max_token_budget``
    should be lowered to ~20 000 with ~4 k headroom AND
    ``LLMClient(max_tokens=...)`` should be lowered in lockstep, since
    the server's input cap is ``max_model_len − max_tokens``).
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory = inventory or InventoryStore(page_store=page_store)

    embedding_client = embedding_client or get_cached_embedding_client()
    llm_client = llm_client or LLMClient()

    graph_channel = graph_channel or GraphPPRChannel(
        embedding_client=embedding_client,
    )

    registry = ToolRegistry()
    for graph_tool in (GraphPprTool, GraphChainTool, EntityInspectTool):
        registry.register(
            graph_tool(
                channel=graph_channel,
                inventory=inventory,
                **(graph_explore_kwargs or {}),
            )
        )
    registry.register(
        ReadTool(page_store=page_store, inventory=inventory, graph_channel=graph_channel)
    )

    return BaseAgent(
        llm_client=llm_client,
        tools=registry,
        system_prompt=system_prompt or GRAPH_SYSTEM_PROMPT,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )


def build_regex_agent(
    *,
    llm_client: Optional[LLMClient] = None,
    page_store: Optional[PageStore] = None,
    inventory: Optional[InventoryStore] = None,
    page_assets_dir: Optional[Path] = None,
    system_prompt: Optional[str] = None,
    max_loops: int = 24,
    max_token_budget: int = 128_000,
    verbose: bool = False,
) -> BaseAgent:
    """Build a BaseAgent that locates evidence by regex alone.

    Tools registered: ``pattern_search`` (exhaustive regex scan over
    pages / passages / table_rows) and ``read`` (full-text page reader).
    No embedding retrieval, no graph navigation — the agent's only
    locator is the regex it writes, so prompt-engineering quality of
    those regexes is the dominant determinant of recall.

    Defaults mirror :func:`build_graph_agent` (``max_loops=24``,
    ``max_token_budget=128k``) so the comparison against the graph
    baseline is at iso-budget. Small-context generators (e.g. Qwen3-8B,
    40 960 ctx): override ``max_token_budget`` down to ~20 000 and lower
    ``LLMClient(max_tokens=...)`` in lockstep — same as the graph agent.
    """
    page_store = page_store or PageStore(page_assets_dir or page_assets_root())
    inventory = inventory or InventoryStore(page_store=page_store)
    llm_client = llm_client or LLMClient()

    registry = ToolRegistry()
    registry.register(PatternSearchTool(page_store=page_store, inventory=inventory))
    # No graph channel on the regex path — pillar-2 baseline contract
    # is "regex + read only"; surfacing graph-derived entities/neighbour
    # pages would confound the comparison.
    registry.register(ReadTool(page_store=page_store, inventory=inventory))

    return BaseAgent(
        llm_client=llm_client,
        tools=registry,
        system_prompt=system_prompt or REGEX_SYSTEM_PROMPT,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )


def build_web_agent(
    *,
    llm_client: Optional[LLMClient] = None,
    tavily_client: Optional["TavilyClient"] = None,
    config_store: Optional["ConfigStore"] = None,
    system_prompt: Optional[str] = None,
    max_loops: int = 12,
    max_token_budget: int = 96_000,
    verbose: bool = False,
) -> BaseAgent:
    """Build a BaseAgent specialised for public-web research.

    Toolset is intentionally narrow: ``web_search`` (Tavily) and
    ``web_fetch`` (HTML stripper). No local-corpus or graph access —
    the web agent must answer purely from public sources, cite URLs,
    and abstain when the web doesn't cover the question.

    Why a separate agent (not a ``base_agent`` with extra tools): the
    cite contract differs (URL vs file_id+page_id), the abstain rule
    differs (no local fallback), and the system prompt needs to keep
    those rules visible in every turn. Splitting at factory level
    rather than at runner level keeps each agent's contract local.

    Caps kept modest (12 loops / 96 k) on purpose: each loop hits the
    Tavily web API, and a search + a single fetch answers most questions
    in 2-4 turns — large caps would only risk runaway external-API cost
    with no accuracy benefit (unlike the corpus agents, which need the
    room to traverse the graph). Small-context generators (e.g. Qwen3-8B,
    40 960 ctx): override ``max_token_budget`` down to ~20 000 and lower
    ``LLMClient(max_tokens=...)`` in lockstep at the call site.
    """
    from model_client.web_search import TavilyClient

    llm_client = llm_client or LLMClient()
    tavily_client = tavily_client or TavilyClient()

    registry = ToolRegistry()
    registry.register(WebSearchTool(tavily_client=tavily_client, config_store=config_store))
    registry.register(WebFetchTool())

    return BaseAgent(
        llm_client=llm_client,
        tools=registry,
        system_prompt=system_prompt or WEB_AGENT_SYSTEM_PROMPT,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )
