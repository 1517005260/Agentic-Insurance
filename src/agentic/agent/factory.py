"""Wire the two EvidenceFS agents.

Both register a single ``ShellTool`` over the compiled Tri-Graph
filesystem plus the shared web tools; the only capability difference is
what the sandbox binds. :func:`build_base_agent` sees only the documents
tree (the faithful raw-corpus DCI baseline); :func:`build_graph_agent`
roots at the whole ``evidence_fs/`` and mounts the shell ABI on PATH.
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.config_store import ConfigStore
    from model_client.web_search import TavilyClient

from agentic.agent.base import BaseAgent
from agentic.agent.prompts import (
    BASE_SYSTEM_PROMPT,
    EVIDENCE_FS_SYSTEM_PROMPT,
    EVIDENCE_FS_SEMANTIC_SUFFIX,
    EVIDENCE_FS_MULTIMODAL_SUFFIX,
)
from agentic.tools.acquisition.shell import ShellTool
from agentic.tools.acquisition import (
    ViewPageTool,
    WebFetchTool,
    WebSearchTool,
)
from agentic.tools.registry import ToolRegistry
from config.settings import evidence_fs_root, evfs_sidecar_socket
from model_client import LLMClient


logger = logging.getLogger(__name__)


# Chat models that accept image input. Used only to DEFAULT the ``multimodal``
# switch — an unknown model is treated as text-only so a non-vision relay is
# never sent an image (it 400s). Callers override with ``multimodal=...``.
_VLM_MARKERS = (
    "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5", "o3", "o4",
    "claude", "gemini", "-vl", "vl-", "llava", "pixtral", "vision",
)


def _is_vlm(model: Optional[str]) -> bool:
    m = (model or "").lower()
    return any(k in m for k in _VLM_MARKERS)


# ===================================================================
# EvidenceFS agents (the only two that survive cleanup): a hermetic
# shell over the compiled Tri-Graph filesystem + web. ``base`` sees only
# the documents (faithful DCI baseline); ``graph`` (= our method) sees
# the whole FS plus the shell ABI on PATH. Web stays a shared registered
# tool — it needs the network + an API key, which the no-network shell
# sandbox deliberately cannot provide.
# ===================================================================

# The EvidenceFS shell ABI lives in the repo, version-controlled and
# corpus-agnostic, so a corpus rebuild never overwrites the agent's tooling.
# src/agentic/agent/factory.py -> parents[1] == src/agentic. Scripts are split
# by capability tier so the graph agent can be ablated:
#   ``lexical``  — string/graph-file ops (stdlib; hermetic sandbox).
#   ``semantic`` — ranked retrieval reusing the real channels (PPR / dense /
#                  bridging); composed ON TOP of lexical. The channels load once
#                  in a host-side sidecar served over a unix socket, so the
#                  sandbox stays hermetic.
_SRC_DIR = Path(__file__).resolve().parents[2]            # <repo>/src
_AGENT_SCRIPTS_DIR = _SRC_DIR / "agentic" / "tools" / "agent_scripts"
_SCRIPT_TIERS = {
    "lexical": [_AGENT_SCRIPTS_DIR / "lexical"],
    "semantic": [_AGENT_SCRIPTS_DIR / "lexical", _AGENT_SCRIPTS_DIR / "semantic"],
}


def _web_tools(tavily_client=None, config_store: Optional["ConfigStore"] = None):
    """The shared web affordance (search + fetch), built once per agent."""
    from model_client.web_search import TavilyClient

    tavily_client = tavily_client or TavilyClient()
    return [
        WebSearchTool(tavily_client=tavily_client, config_store=config_store),
        WebFetchTool(),
    ]


def build_base_agent(
    *,
    llm_client: Optional[LLMClient] = None,
    corpus_root: Optional[Path] = None,
    tavily_client: Optional["TavilyClient"] = None,
    config_store: Optional["ConfigStore"] = None,
    system_prompt: Optional[str] = None,
    multimodal: Optional[bool] = None,
    max_loops: int = 24,
    max_token_budget: int = 128_000,
    verbose: bool = False,
) -> BaseAgent:
    """Build the EvidenceFS **base** agent: a hermetic DCI shell over the
    documents tree plus web — the faithful raw-corpus baseline for the v1 A/B
    (shell + web, NO graph files, NO scripts).

    ``corpus_root`` defaults to ``evidence_fs/documents`` so the base and graph
    agents read byte-identical source text (only the graph affordances differ).

    ``multimodal`` adds ``view_page`` so a vision model can read a page's
    rendered image (defaults on for a VLM chat model, off otherwise). It is a
    shared capability — kept identical on base and graph — so the A/B isolates
    the graph affordances, not who can see images.
    """
    corpus_root = corpus_root or (evidence_fs_root() / "documents")
    llm_client = llm_client or LLMClient()

    registry = ToolRegistry()
    registry.register(ShellTool(corpus_root=corpus_root))
    for tool in _web_tools(tavily_client, config_store):
        registry.register(tool)

    prompt = system_prompt or BASE_SYSTEM_PROMPT
    if multimodal if multimodal is not None else _is_vlm(llm_client.model):
        registry.register(ViewPageTool(corpus_root=corpus_root))
        if system_prompt is None:
            prompt += EVIDENCE_FS_MULTIMODAL_SUFFIX

    return BaseAgent(
        llm_client=llm_client,
        tools=registry,
        system_prompt=prompt,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )


def build_graph_agent(
    *,
    llm_client: Optional[LLMClient] = None,
    corpus_root: Optional[Path] = None,
    tier: str = "lexical",
    scripts_dirs: Optional[list] = None,
    tavily_client: Optional["TavilyClient"] = None,
    config_store: Optional["ConfigStore"] = None,
    system_prompt: Optional[str] = None,
    multimodal: Optional[bool] = None,
    max_loops: int = 24,
    max_token_budget: int = 128_000,
    verbose: bool = False,
) -> BaseAgent:
    """Build the EvidenceFS **graph** agent (our method): the same shell + web
    as the base agent, but rooted at the whole ``evidence_fs/`` Tri-Graph
    filesystem (documents + nodes + edges + views) with the shell ABI bound on
    PATH at ``/scripts/<tier>``.

    ``tier`` selects which capability tiers of the ABI are exposed — the
    ablation axis:
      * ``"lexical"`` — string/graph-file ops only (find_surface /
        expand_surface / bridge_surfaces / show_passage / show_sentence /
        grep_passages), hermetic stdlib sandbox.
      * ``"semantic"`` — lexical PLUS the ranked-retrieval programs that reuse
        the real channels (genuine PPR / dense / bridging).
    Pass ``scripts_dirs`` to override the tier→dirs mapping explicitly.

    The only delta vs the base agent is graph access — so the A/B isolates
    "does a shell-operable Tri-Graph help vs raw-DCI" with everything else held
    fixed (same shell, same web, same budget, same model).
    """
    corpus_root = corpus_root or evidence_fs_root()
    if scripts_dirs is None:
        scripts_dirs = _SCRIPT_TIERS.get(tier, _SCRIPT_TIERS["lexical"])
    llm_client = llm_client or LLMClient()

    # The semantic tier's scripts reuse the real channels, served by a host-side
    # sidecar that loads them once. Start it (idempotent; blocks until warm) and
    # bind its socket into the otherwise-hermetic sandbox. Lexical-only needs
    # neither, so it stays a pure hermetic shell with no socket.
    needs_sidecar = any(Path(d).name == "semantic" for d in scripts_dirs)
    socket_path = None
    if needs_sidecar:
        # Imported lazily so running the sidecar as ``python -m
        # agentic.tools.evfs_sidecar`` doesn't double-import via the factory.
        from agentic.tools import evfs_sidecar

        socket_path = evfs_sidecar_socket()
        evfs_sidecar.ensure_running(socket_path)

    registry = ToolRegistry()
    registry.register(ShellTool(
        corpus_root=corpus_root,
        scripts_dirs=scripts_dirs,
        socket_path=socket_path,
        # The first semantic call may wait on the sidecar's one-time channel
        # load, so the semantic tier gets a long ceiling; warm calls are fast.
        timeout_s=240 if needs_sidecar else 30,
    ))
    for tool in _web_tools(tavily_client, config_store):
        registry.register(tool)

    # Teach the agent the semantic ABI only when that tier is actually mounted.
    default_prompt = EVIDENCE_FS_SYSTEM_PROMPT
    if needs_sidecar:
        default_prompt += EVIDENCE_FS_SEMANTIC_SUFFIX

    # view_page is a shared capability (same as on the base agent), so the A/B
    # stays about the graph affordances, not who can see page images.
    if multimodal if multimodal is not None else _is_vlm(llm_client.model):
        registry.register(ViewPageTool(corpus_root=corpus_root))
        default_prompt += EVIDENCE_FS_MULTIMODAL_SUFFIX

    return BaseAgent(
        llm_client=llm_client,
        tools=registry,
        system_prompt=system_prompt or default_prompt,
        max_loops=max_loops,
        max_token_budget=max_token_budget,
        verbose=verbose,
    )
