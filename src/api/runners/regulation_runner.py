"""Non-streaming regulation-search runner.

Resolves the jurisdiction → include_domains mapping from the admin
config (``tavily.include_domains_hk`` / ``..._cn``) and invokes the
single-call web RAG service. Returns a structured response the
route serializes as JSON — no SSE here, the call is fast enough
(~3-5 s) that a normal POST is fine and avoids streaming complexity
for a workbench that just wants a card.
"""
import logging
from typing import List, Optional

from api.schemas.insurance import (
    JurisdictionLiteral,
    RegulationSearchResponse,
    RegulationSource,
)
from api.services import web_rag as web_rag_svc
from config.config_store import ConfigStore
from model_client import LLMClient
from model_client.web_search import TavilyClient


logger = logging.getLogger(__name__)


def _resolve_domains(
    config: ConfigStore, jurisdiction: JurisdictionLiteral
) -> Optional[List[str]]:
    """Translate ``hk`` / ``cn`` / ``both`` into a Tavily include_domains list."""
    hk_csv = str(config.get("tavily.include_domains_hk") or "").strip()
    cn_csv = str(config.get("tavily.include_domains_cn") or "").strip()
    hk = [d.strip() for d in hk_csv.split(",") if d.strip()]
    cn = [d.strip() for d in cn_csv.split(",") if d.strip()]
    if jurisdiction == "hk":
        return hk or None
    if jurisdiction == "cn":
        return cn or None
    # ``both`` = union; if either side is empty, fall back to the
    # other rather than returning nothing (the alternative would be
    # ``None`` which means "no whitelist", not what the user asked).
    merged = sorted(set(hk + cn))
    return merged or None


def run_regulation_search(
    *,
    query: str,
    jurisdiction: JurisdictionLiteral,
    llm: LLMClient,
    tavily: TavilyClient,
    config: ConfigStore,
    max_results: Optional[int] = None,
    days: Optional[int] = None,
) -> RegulationSearchResponse:
    """Synchronous regulation search → summary card.

    Caller (the route) should wrap this in
    :func:`fastapi.concurrency.run_in_threadpool` because both
    Tavily and the LLM block.
    """
    include_domains = _resolve_domains(config, jurisdiction)
    effective_max = (
        int(max_results)
        if max_results is not None
        else int(config.get("tavily.max_results"))
    )
    search_depth = str(config.get("tavily.search_depth") or "basic")
    system_prompt = str(config.get("prompt.regulation"))
    answer_max_tokens = int(config.get("rag.answer_max_tokens"))

    # ``days`` is only honored by Tavily for ``topic="news"`` — keep
    # general topic but pass days through; Tavily will silently ignore
    # if the topic doesn't accept it. We don't switch to ``news`` here
    # because regulation pages typically aren't news-classified.
    rr = web_rag_svc.run_summarized(
        llm=llm,
        tavily=tavily,
        query=query,
        max_results=effective_max,
        search_depth=search_depth,
        include_domains=include_domains,
        system_prompt=system_prompt,
        max_tokens=answer_max_tokens,
        days=days,
    )

    cited = web_rag_svc._extract_cited_sources(rr.answer, rr.sources)

    return RegulationSearchResponse(
        answer=rr.answer,
        summary_chars=len(rr.answer),
        sources=[RegulationSource(**s.to_public_dict()) for s in rr.sources],
        n_results=rr.n_results,
        n_cited=len(cited),
        jurisdiction=jurisdiction,
        used_include_domains=include_domains,
        search_query=rr.search_query,
    )
