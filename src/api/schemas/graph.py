"""Pydantic shapes for the /graph endpoints.

The node/edge shape mirrors what G6 v5 expects (``id`` + ``label`` on
nodes; ``source`` + ``target`` on edges) so the frontend can call
``setData()`` directly. ``hash_id`` deliberately leaks for graph data
(needed to build the edge list) but NOT for the node-detail hover card
— per the project preference, raw IDs in the UI are ugly.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Hash-id length cap. Real ids are ``<namespace>-<md5>`` ≈ 40 chars;
# we accept up to 128 to leave headroom for future namespacing.
_HASH_ID_MAX = 128


# ---------------------------------------------------------- shared shapes


class GraphNode(BaseModel):
    """One node in the G6 setData payload."""

    id: str
    label: str
    vertex_type: str             # "entity" | "passage" | "sentence"
    hop: int = 0                 # BFS distance from the seed (0 for the seed)
    score: float = 0.0           # max edge weight on the shortest path here


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float
    type: str                    # "entity_passage" | "adjacent_passage" | "alias" | ...


class ClusterBrief(BaseModel):
    canonical: str
    members: List[str]


# ---------------------------------------------------------- /overview


class NodeBrief(BaseModel):
    id: str
    label: str
    vertex_type: str
    degree: int


class GraphOverviewResponse(BaseModel):
    counts: Dict[str, int] = Field(
        ...,
        description=(
            "Always carries ``nodes`` + ``edges`` plus per-vertex_type "
            "tallies (e.g. ``entity``, ``passage``, ``sentence``)."
        ),
    )
    top_central_entities: List[NodeBrief] = Field(
        ...,
        description="Top-10 entity vertices by degree (canvas first paint).",
    )


# ---------------------------------------------------------- /seed


class SeedHit(BaseModel):
    hash_id: str
    surface: str
    similarity: float
    logical_cluster: Optional[ClusterBrief] = None


# ---------------------------------------------------------- /expand


class GraphSubgraphResponse(BaseModel):
    """Generic subgraph shape — used by /expand AND /nodes/{id}/expand."""

    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ---------------------------------------------------------- /nodes/{id}


class NeighborFile(BaseModel):
    """One entry of ``neighboring_files`` — file_id + readable name."""
    file_id: str
    display_name: str


class NodeDetailResponse(BaseModel):
    """Hover-card payload — no hash_id field (the URL already names it).

    For passages, ``file_id`` + ``page_number`` are surfaced so the
    card can offer a "open in PDF" link without a second round trip.
    ``display_name`` accompanies any file_id so the UI can render the
    human-readable label instead of a sha-prefixed slug.
    """

    surface: str
    vertex_type: str
    degree: int
    logical_cluster: Optional[ClusterBrief] = None
    mention_count: Optional[int] = None
    neighboring_files: Optional[List[NeighborFile]] = None
    file_id: Optional[str] = None
    display_name: Optional[str] = None
    page_number: Optional[int] = None


# ---------------------------------------------------------- /ppr-subgraph


class PPRSubgraphRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    file_ids: Optional[List[str]] = Field(
        None,
        max_length=50,
        description="Optional: prune passages to these file_ids only (max 50).",
    )


class PPRSeed(BaseModel):
    id: str
    surface: str
    similarity: float


class PPRActivedEntity(BaseModel):
    id: str
    surface: str
    score: float
    iteration_tier: int


class PPRPassage(BaseModel):
    hash_id: Optional[str]
    file_id: str
    page_id: str
    score: float


class PPRSubgraphResponse(BaseModel):
    """``mode`` distinguishes:
    * ``ppr``       — happy path, lists/edges populated
    * ``no_seeds``  — query had no entity match (UI: "no graph hit")
    * ``no_graph``  — graphml not built yet (UI: "graph not ready")
    """

    mode: Literal["ppr", "no_seeds", "no_graph"]
    seeds: List[PPRSeed]
    actived_entities: List[PPRActivedEntity]
    passages: List[PPRPassage]
    edges: List[GraphEdge]


__all__ = [
    "GraphNode",
    "GraphEdge",
    "ClusterBrief",
    "NodeBrief",
    "GraphOverviewResponse",
    "SeedHit",
    "GraphSubgraphResponse",
    "NodeDetailResponse",
    "PPRSubgraphRequest",
    "PPRSeed",
    "PPRActivedEntity",
    "PPRPassage",
    "PPRSubgraphResponse",
]
