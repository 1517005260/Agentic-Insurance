"""Pre-issuance risk-prediction runner — GraphAgent + PPR-anchored Sankey side-channel.

Wraps :func:`api.runners.agent_runner.stream_agent` (kind=``graph``) so the
GraphAgent drives its existing PPR → neighbors → read pipeline behind a
workbench-specific system prompt, then the wrapper augments the agent's
``final`` SSE event with a ``risk_subgraph`` payload computed from one
authoritative PPR call against the same ``(customer, scenario, file_id)``
triple. The frontend renders the markdown answer + a 3-layer Sankey
(customer fields → risk factors → triggered clauses) in the same tab.

Why a wrapper instead of a fork of the agent runner:

* GraphAgent already has the right tool budget (graph_explore + read);
  duplicating its loop just to inject a different system prompt would
  force every loop / token-budget tweak in two places.
* The Sankey data model is independent of agent loop telemetry — it is
  whatever the user's profile picks out of the policy's PPR neighborhood.
  Computing it once up-front (deterministic) and stitching it onto the
  agent's ``final`` keeps the agent free to wander without changing the
  visualization contract.
* Byte-level SSE pass-through preserves every other frame the agent emits
  (token, tool_call, tool_result, graph_subgraph, citations) so the
  workbench scaffold's existing renderers — including the live force-
  directed canvas that subscribes to ``graph_subgraph`` — keep working
  unchanged.

Flavor name ``risk_predict`` is reserved for future tracer plumbing if
a route handler decides to persist runs under
``${STORAGE_PATH}/risk_predict/...``; today the route fires fire-and-
forget like the other insurance endpoints.
"""
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from agentic.agent.base import BaseAgent
from api.runners.agent_runner import stream_agent
from api.schemas.insurance import CustomerProfile
from api.services.graph_service import GraphService, GraphServiceUnavailable
from api.sse import format_event
from config.config_store import ConfigStore


logger = logging.getLogger(__name__)


_FLAVOR = "risk_predict"

# How many risk factors / clauses to surface in the Sankey. Beyond ~8
# the diagram becomes a hairball; below ~3 it looks empty.
_MAX_RISK_FACTORS = 8
_MAX_TRIGGERED_CLAUSES = 8


# ---------------------------------------------------------------- entry point


async def stream_risk_predict(
    *,
    file_id: str,
    customer: CustomerProfile,
    scenario: Optional[str],
    agent: BaseAgent,
    graph_service: GraphService,
    config: ConfigStore,
) -> AsyncIterator[bytes]:
    """Stream one risk-prediction run as SSE bytes.

    Pipeline:
      1. Build the PPR query string from the customer profile +
         optional scenario hint and call :meth:`GraphService.ppr_subgraph`
         once to obtain authoritative entity / passage scores scoped to
         the candidate ``file_id``.
      2. Project the PPR result into the Sankey ``risk_subgraph`` shape.
      3. Drive :func:`stream_agent` with ``system_prompt_override`` set
         to ``prompt.risk_predict`` and a structured query carrying the
         file_id + profile so the GraphAgent's first ``graph_explore``
         call has enough context to mirror the same neighborhood.
      4. Pass every SSE frame through unchanged except the final
         ``final`` event, whose data dict is augmented with
         ``{risk_subgraph, flavor}`` before re-serialisation.
    """
    # --- 1. Build PPR query -------------------------------------------------
    ppr_query = _build_ppr_query(customer, scenario)

    # --- 2. Authoritative PPR call ------------------------------------------
    # The GraphAgent will probably issue its own PPR call as loop 1, but
    # the Sankey needs deterministic scores tied to the visualisation
    # contract — re-running PPR locally guarantees the picture matches
    # the prompt the agent saw. ``ppr_subgraph`` is sync and can take
    # 300-500ms on warm channel / multiple seconds cold; ``to_thread``
    # keeps the event loop free so heartbeats and other concurrent
    # requests don't stall behind it.
    risk_subgraph: Dict[str, Any]
    try:
        ppr_result = await asyncio.to_thread(
            graph_service.ppr_subgraph, ppr_query, file_ids=[file_id]
        )
        risk_subgraph = _build_risk_subgraph(customer, ppr_result)
    except GraphServiceUnavailable as exc:
        # Surface the runtime problem as an early error frame and abort.
        # Better than letting the agent run and emit a final with an
        # empty risk_subgraph that the user has no way to interpret.
        yield format_event(
            "error",
            {"message": str(exc), "type": "GraphServiceUnavailable"},
        )
        yield format_event("done", {})
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("risk_predict: PPR pre-pass failed")
        yield format_event(
            "error",
            {"message": str(exc), "type": type(exc).__name__},
        )
        yield format_event("done", {})
        return

    # --- 3. + 4. Drive agent + intercept SSE --------------------------------
    user_query = _build_agent_query(file_id, customer, scenario)
    system_prompt = str(config.get("prompt.risk_predict"))

    # Buffer carries any partial frame between async iterations of the
    # underlying stream — SSE frames are delimited by ``\n\n`` and a
    # single TCP read may bisect a frame, especially long ``final``
    # payloads. Keeps state out of the inner loop.
    buffer = bytearray()
    final_seen = False

    async for chunk in stream_agent(
        query=user_query,
        kind="graph",
        agent=agent,
        config=config,
        system_prompt_override=system_prompt,
    ):
        buffer.extend(chunk)
        # Drain every complete frame the buffer now contains. ``\n\n``
        # is the SSE record separator, used both for event frames and
        # heartbeat comments — both shapes survive the inspect-or-pass
        # branching below.
        while True:
            sep_idx = buffer.find(b"\n\n")
            if sep_idx < 0:
                break
            frame = bytes(buffer[: sep_idx + 2])
            del buffer[: sep_idx + 2]
            transformed, was_final = _maybe_augment_final(frame, risk_subgraph)
            if was_final:
                final_seen = True
            yield transformed

    # Flush any trailing bytes (well-formed SSE never leaves a partial
    # frame, but guard anyway so the connection terminates cleanly).
    if buffer:
        yield bytes(buffer)

    if not final_seen:
        # Agent crashed before emitting ``final``; the agent runner
        # already pushed an ``error`` + ``done``. Nothing to add — the
        # frontend gets the error-state final via stream_agent's own
        # error path (which still drains through the same SSE stream).
        logger.debug("risk_predict: stream ended without observing final frame")


# ---------------------------------------------------------------- frame interceptor


def _maybe_augment_final(
    frame: bytes, risk_subgraph: Dict[str, Any]
) -> Tuple[bytes, bool]:
    """Return ``(bytes_to_yield, was_augmented_final)``.

    SSE frames have shape ``event: <name>\\n data: <json>\\n\\n`` (or a
    leading ``: keepalive`` heartbeat with no event/data). Anything we
    cannot positively identify as a JSON ``final`` frame passes through
    unchanged — this keeps cost bounded and avoids breaking unknown
    future event types.
    """
    # Heartbeats start with ``:`` and carry no event field.
    if frame.startswith(b":"):
        return frame, False

    parsed = _split_event_frame(frame)
    if parsed is None:
        return frame, False
    event_name, data_json = parsed
    if event_name != "final":
        return frame, False

    try:
        data = json.loads(data_json)
    except (json.JSONDecodeError, TypeError):
        # Pass through; never break the SSE stream over our own
        # augmentation logic.
        return frame, False
    if not isinstance(data, dict):
        return frame, False

    data["risk_subgraph"] = risk_subgraph
    data["flavor"] = _FLAVOR
    return format_event("final", data), True


def _split_event_frame(frame: bytes) -> Optional[Tuple[str, bytes]]:
    """Parse a single SSE frame into ``(event_name, raw_data_bytes)``.

    The encoder in :func:`api.sse.format_event` always emits the two
    fields in this order with single-line JSON, so a strict line-prefix
    parse is enough — no need to handle the SSE multi-line ``data:``
    continuation form here. Returns ``None`` if either field is missing
    or malformed (the caller treats that as "passthrough unchanged").
    """
    # Trim the trailing ``\n\n`` so the line-split below doesn't yield
    # two empty trailing segments.
    body = frame.rstrip(b"\n")
    lines = body.split(b"\n")
    event_name: Optional[str] = None
    data_bytes: Optional[bytes] = None
    for line in lines:
        if line.startswith(b"event: "):
            try:
                event_name = line[len(b"event: "):].decode("ascii")
            except UnicodeDecodeError:
                return None
        elif line.startswith(b"data: "):
            data_bytes = line[len(b"data: "):]
    if event_name is None or data_bytes is None:
        return None
    return event_name, data_bytes


# ---------------------------------------------------------------- prompt builders


def _format_profile(p: CustomerProfile) -> str:
    """Pretty-print the customer profile as a labelled block.

    Mirrors ``exclusion_runner._format_profile`` so the two workbenches'
    user prompts present customer data the same way; copied rather than
    imported because keeping each runner's prompt-build path independent
    is the established convention here.
    """
    parts: List[str] = [
        f"- 年龄: {p.age}",
        f"- 性别: {p.gender}",
        f"- 职业: {p.occupation}",
    ]
    if p.occupation_risk:
        parts.append(f"- 职业风险: {p.occupation_risk}")
    if p.health_history:
        parts.append(f"- 病史: {', '.join(p.health_history)}")
    if p.family_history:
        parts.append(f"- 家族史: {', '.join(p.family_history)}")
    if p.budget_annual is not None:
        parts.append(f"- 年预算: {p.budget_annual}")
    if p.goal:
        parts.append(f"- 主诉求: {p.goal}")
    if p.notes:
        parts.append(f"- 补充说明: {p.notes}")
    return "\n".join(parts)


def _build_ppr_query(customer: CustomerProfile, scenario: Optional[str]) -> str:
    """Compose the free-text query the deterministic PPR call sees.

    Concatenates the highest-signal customer fields with the optional
    scenario hint. PPR cares about *which entities the surface text
    matches*, not sentence structure, so a comma-joined list is good
    enough — ``GraphPPRChannel`` runs GLiNER NER over this string.
    """
    tokens: List[str] = [
        f"{customer.age}岁",
        customer.gender,
        customer.occupation,
    ]
    if customer.occupation_risk:
        tokens.append(f"职业风险{customer.occupation_risk}")
    tokens.extend(customer.health_history)
    tokens.extend(customer.family_history)
    if customer.goal:
        tokens.append(customer.goal)
    if scenario:
        tokens.append(scenario)
    if customer.notes:
        tokens.append(customer.notes)
    return "，".join(t for t in tokens if t)


def _build_agent_query(
    file_id: str, customer: CustomerProfile, scenario: Optional[str]
) -> str:
    """User prompt the GraphAgent receives.

    First line carries the file_id so the GraphAgent's read tool has a
    concrete scope (GraphAgent has no list_files tool — without an
    explicit file_id it cannot ground its analysis on a specific
    policy). The customer profile + scenario follow as labelled blocks
    so the workbench prompt's ``mode=ppr`` step has fields to translate
    into the PPR query.
    """
    blocks = [
        f"## 候选保单\nfile_id=`{file_id}`",
        f"## 客户档案\n{_format_profile(customer)}",
    ]
    if scenario:
        blocks.append(f"## 假设场景\n{scenario.strip()}")
    blocks.append(
        "## 任务\n按系统提示固定的 PPR → neighbors → read 流水线执行，"
        "最后输出风险预测报告。"
    )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- risk subgraph


def _build_risk_subgraph(
    customer: CustomerProfile, ppr_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Project the PPR subgraph into the 3-layer Sankey shape.

    Output shape (frontend ``RiskSankeyCanvas`` consumes this verbatim
    via the augmented ``final.risk_subgraph`` field)::

        {
          "customer_fields":   [{"id", "label"}, ...],
          "risk_factors":      [{"id", "label", "ppr_score"}, ...],
          "triggered_clauses": [{"id", "sup", "file_id", "page_id"}, ...],
          "edges":             [{"source", "target", "weight"}, ...],
          "mode":              "ppr" | "no_seeds" | "no_graph",
        }

    Layer 1→2 edges are explicitly **uniform priors** — the customer
    fields are signals that *could* trigger any factor; we do NOT claim
    to have learnt per-pair weights. (The paper must be honest about
    this; reviewers will ask.) Layer 2→3 edges are taken **only from
    real entity↔passage edges in the PPR-induced subgraph** —
    ``ppr_result.edges`` is the authoritative adjacency, and the link
    weight is the passage's own PPR score so high-ranked clauses
    dominate visually. If the PPR subgraph has no entity↔passage edges
    in scope, the L2→L3 strip is empty rather than fabricated.
    """
    mode = ppr_result.get("mode", "ppr")
    customer_fields = _customer_fields(customer)

    if mode != "ppr":
        return {
            "customer_fields": customer_fields,
            "risk_factors": [],
            "triggered_clauses": [],
            "edges": [],
            "mode": mode,
        }

    # Index passage hash_id → (file_id, page_id) and pre-filter by
    # candidate file_id. ``triggered_clauses`` will reuse this so the
    # clause node id space lines up with the L2→L3 edge endpoints.
    passages_raw = ppr_result.get("passages") or []
    passage_by_hash: Dict[str, Dict[str, Any]] = {}
    for p in passages_raw:
        h = p.get("hash_id")
        fid = p.get("file_id")
        pid = p.get("page_id")
        if not (isinstance(h, str) and fid and pid):
            continue
        passage_by_hash[h] = {
            "file_id": str(fid),
            "page_id": str(pid),
            "ppr_score": round(float(p.get("score") or 0.0), 4),
        }

    # Real entity↔passage adjacency from the PPR subgraph. Maps entity
    # hash_id → list of (passage hash_id, edge weight). Only edges where
    # both endpoints exist in the kept subgraph survive — fabricated
    # uniform fan-outs are explicitly avoided here.
    ent_to_passages: Dict[str, List[Tuple[str, float]]] = {}
    for e in (ppr_result.get("edges") or []):
        src = e.get("source")
        dst = e.get("target")
        if not (isinstance(src, str) and isinstance(dst, str)):
            continue
        # Either direction may carry the entity→passage relation.
        if src in passage_by_hash and dst not in passage_by_hash:
            ent, pas = dst, src
        elif dst in passage_by_hash and src not in passage_by_hash:
            ent, pas = src, dst
        else:
            continue
        weight = float(e.get("weight") or 1.0)
        ent_to_passages.setdefault(ent, []).append((pas, weight))

    # Risk factors = top-N actived entities by PPR score, restricted to
    # those that actually connect to at least one in-scope passage. An
    # actived entity with zero adjacency to the candidate file's pages
    # is noise for this view (it activated globally but not for the
    # selected policy).
    actived_sorted = sorted(
        ppr_result.get("actived_entities") or [],
        key=lambda e: float(e.get("score") or 0.0),
        reverse=True,
    )
    risk_factors: List[Dict[str, Any]] = []
    kept_entity_ids: List[str] = []
    for e in actived_sorted:
        ent_id = e.get("id")
        if not (isinstance(ent_id, str) and ent_id in ent_to_passages):
            continue
        risk_factors.append(
            {
                "id": f"rf_{ent_id}",
                "label": str(e.get("surface") or ent_id),
                "ppr_score": round(float(e.get("score") or 0.0), 4),
            }
        )
        kept_entity_ids.append(ent_id)
        if len(risk_factors) >= _MAX_RISK_FACTORS:
            break

    # Triggered clauses = passages reachable from at least one kept risk
    # factor via real PPR edges. Order by PPR score desc, cap to budget.
    # Intentionally NOT carrying a ``sup`` field: PPR rank order !=
    # agent's read order, and the frontend's CitationDrawer keys on
    # agent-side ``citations[].sup``. Reusing the same name would cross-
    # link clicks to wrong passages. Frontend resolves clicks by
    # ``(file_id, page_id)`` match instead.
    reachable_passage_hashes: set[str] = set()
    for ent_id in kept_entity_ids:
        for pas_hash, _w in ent_to_passages.get(ent_id, []):
            reachable_passage_hashes.add(pas_hash)
    reachable_passages = sorted(
        (passage_by_hash[h] | {"hash_id": h} for h in reachable_passage_hashes),
        key=lambda p: p["ppr_score"],
        reverse=True,
    )[:_MAX_TRIGGERED_CLAUSES]
    hash_to_clause_id: Dict[str, str] = {}
    triggered_clauses: List[Dict[str, Any]] = []
    for rank, p in enumerate(reachable_passages, start=1):
        clause_id = f"clause_{rank}"
        hash_to_clause_id[p["hash_id"]] = clause_id
        triggered_clauses.append(
            {
                "id": clause_id,
                "file_id": p["file_id"],
                "page_id": p["page_id"],
                "ppr_score": p["ppr_score"],
            }
        )

    # --- Edges --------------------------------------------------------------
    edges: List[Dict[str, Any]] = []
    # L1 → L2: uniform prior. Honest framing: "客户档案任一字段都可能与
    # 任一风险因子关联，权重未学习" — never claim a learnt edge weight
    # in this layer. Per-field outflow = 1/N so column 2 inflow is
    # balanced and Sankey doesn't squish.
    if customer_fields and risk_factors:
        l12_weight = 1.0 / len(customer_fields)
        for cf in customer_fields:
            for rf in risk_factors:
                edges.append(
                    {
                        "source": cf["id"],
                        "target": rf["id"],
                        "weight": l12_weight,
                    }
                )

    # L2 → L3: take only the real entity↔passage edges in the PPR
    # subgraph. Weight = passage PPR score (deterministic measurement
    # of how strongly the clause activated against the customer-derived
    # seeds). No fabricated complete bipartite — if the PPR adjacency
    # is sparse, the diagram is sparse, faithful to the data.
    for ent_id in kept_entity_ids:
        rf_id = f"rf_{ent_id}"
        for pas_hash, _w in ent_to_passages.get(ent_id, []):
            clause_id = hash_to_clause_id.get(pas_hash)
            if clause_id is None:
                continue
            score = passage_by_hash[pas_hash]["ppr_score"]
            edges.append(
                {
                    "source": rf_id,
                    "target": clause_id,
                    "weight": round(float(score), 4),
                }
            )

    return {
        "customer_fields": customer_fields,
        "risk_factors": risk_factors,
        "triggered_clauses": triggered_clauses,
        "edges": edges,
        "mode": "ppr",
    }


def _customer_fields(p: CustomerProfile) -> List[Dict[str, str]]:
    """Customer profile → Sankey column-1 nodes.

    Skips fields the user did not supply so the Sankey doesn't show
    empty inputs. Each id is namespaced ``cf_<name>`` so it cannot
    collide with risk-factor / clause ids on the L2 / L3 columns.
    """
    fields: List[Dict[str, str]] = [
        {"id": "cf_age", "label": f"年龄 {p.age}"},
        {"id": "cf_gender", "label": f"性别 {p.gender}"},
        {"id": "cf_occupation", "label": f"职业 {p.occupation}"},
    ]
    if p.occupation_risk:
        fields.append({"id": "cf_occupation_risk", "label": f"职业风险 {p.occupation_risk}"})
    for i, h in enumerate(p.health_history[:5]):
        fields.append({"id": f"cf_health_{i}", "label": f"病史 {h}"})
    for i, h in enumerate(p.family_history[:3]):
        fields.append({"id": f"cf_family_{i}", "label": f"家族史 {h}"})
    if p.goal:
        fields.append({"id": "cf_goal", "label": f"主诉求 {p.goal[:20]}"})
    return fields
