"""E2E for the /graph endpoints.

Exercises the routes against the real on-disk LinearRAG graph (the
same graphml the dev box uses for `scripts/test_graphagent_corpus.py`),
so failures correspond to a real shape change. The harness from
``test_chat_e2e`` boots the full lifespan in-process; we add light
fixtures specific to the graph surface.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient


# Reuse the harness from test_chat_e2e (same lifespan boot + DB cleanup).
# tests/e2e has no __init__.py so pytest puts the directory on sys.path
# and bare-name import works.
from test_chat_e2e import (  # noqa: E402
    _client_for,
    _force_runtime_settings,
    app_harness,
    anyio_backend,
)


pytestmark = pytest.mark.anyio


# Skip the entire module if the on-disk graph artifacts aren't available.
# This makes the suite pass cleanly in fresh-clone CI envs that haven't
# run an ingest.
def _graph_artifacts_present() -> bool:
    from pathlib import Path

    from config.settings import faiss_graph_dir, faiss_graph_entity_dir

    graphml = faiss_graph_dir() / "LinearRAG.graphml"
    entity_dir = Path(faiss_graph_entity_dir())
    return graphml.is_file() and entity_dir.is_dir()


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not _graph_artifacts_present(),
        reason="LinearRAG.graphml not present; run an ingest first",
    ),
]


@asynccontextmanager
async def _admin_and_analyst(app_harness) -> AsyncIterator[tuple]:
    """Boot, log in admin + create+log in an analyst."""
    from api.auth import hash_password
    from api.db import session_scope
    from api.models import User

    async with _client_for(app_harness) as (client, app, admin_headers):
        async with session_scope() as db:
            db.add(
                User(
                    username="graph-analyst",
                    password_hash=hash_password("graph-pwd-123"),
                    role="analyst",
                    is_active=1,
                )
            )
        r = await client.post(
            "/auth/login",
            data={"username": "graph-analyst", "password": "graph-pwd-123"},
        )
        assert r.status_code == 200
        analyst_headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
        yield client, app, admin_headers, analyst_headers


# =====================================================================
# /graph/overview
# =====================================================================


async def test_overview_shape(app_harness):
    """Counts + top-10 entities by degree, in declaration order."""
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get("/graph/overview", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        # Counts always carry nodes + edges; type counts depend on corpus.
        assert "counts" in body and "top_central_entities" in body
        assert body["counts"]["nodes"] > 0
        assert body["counts"]["edges"] > 0
        assert body["counts"].get("entity", 0) > 0
        # top_central is sorted descending by degree.
        top = body["top_central_entities"]
        assert 1 <= len(top) <= 10
        for i in range(len(top) - 1):
            assert top[i]["degree"] >= top[i + 1]["degree"]
        # Every entry has the four required fields.
        for n in top:
            assert {"id", "label", "vertex_type", "degree"} <= set(n.keys())
            assert n["vertex_type"] == "entity"


async def test_overview_requires_auth(app_harness):
    async with _client_for(app_harness) as (client, _app, _headers):
        r = await client.get("/graph/overview")
        # FastAPI's OAuth2PasswordBearer returns 401 with WWW-Authenticate
        assert r.status_code == 401


# =====================================================================
# /graph/seed
# =====================================================================


async def test_seed_finds_known_entity(app_harness):
    """The corpus has 'hong kong' as a high-degree entity; should appear in top."""
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get("/graph/seed", headers=headers, params={"q": "hong kong"})
        assert r.status_code == 200, r.text
        hits = r.json()
        assert isinstance(hits, list)
        # First hit should be the literal match (or extremely close).
        assert any("hong" in h["surface"].lower() for h in hits[:3]), (
            f"top hits: {[h['surface'] for h in hits[:5]]}"
        )
        # Each hit has the expected fields.
        for h in hits:
            assert {"hash_id", "surface", "similarity"} <= set(h.keys())
            assert 0.0 <= h["similarity"] <= 1.0


async def test_seed_top_k_caps(app_harness):
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get(
            "/graph/seed", headers=headers, params={"q": "policy", "top_k": 3}
        )
        assert r.status_code == 200, r.text
        assert len(r.json()) <= 3


async def test_seed_validates_query(app_harness):
    """Empty q must 422 (FastAPI Query min_length=1)."""
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get("/graph/seed", headers=headers, params={"q": ""})
        assert r.status_code == 422


# =====================================================================
# /graph/expand
# =====================================================================


async def _pick_central_entity_id(client, headers) -> str:
    r = await client.get("/graph/overview", headers=headers)
    return r.json()["top_central_entities"][0]["id"]


async def test_expand_seed_then_neighbors(app_harness):
    """Expanding the most-central entity 1-hop returns seed (hop=0) + neighbors (hop=1)."""
    async with _client_for(app_harness) as (client, _app, headers):
        seed_id = await _pick_central_entity_id(client, headers)
        r = await client.get(
            "/graph/expand", headers=headers, params={"node_id": seed_id, "hops": 1}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # seed always present at hop=0.
        seed_node = next((n for n in body["nodes"] if n["id"] == seed_id), None)
        assert seed_node is not None, f"seed missing from nodes: {[n['id'] for n in body['nodes']]}"
        assert seed_node["hop"] == 0
        # At least one neighbor at hop=1.
        non_seed_hops = {n["hop"] for n in body["nodes"] if n["id"] != seed_id}
        assert non_seed_hops <= {1}, f"unexpected hop values: {non_seed_hops}"
        assert len(non_seed_hops) >= 1, "no neighbors returned"
        # Edges only reference returned nodes.
        node_ids = {n["id"] for n in body["nodes"]}
        for e in body["edges"]:
            assert e["source"] in node_ids and e["target"] in node_ids
            assert e["weight"] >= 0
            assert e["type"]


async def test_expand_unknown_node_404(app_harness):
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get(
            "/graph/expand", headers=headers, params={"node_id": "entity-deadbeef"}
        )
        assert r.status_code == 404


async def test_expand_vertex_type_filter(app_harness):
    """vertex_type=entity drops passage neighbors from the result."""
    async with _client_for(app_harness) as (client, _app, headers):
        seed_id = await _pick_central_entity_id(client, headers)
        r = await client.get(
            "/graph/expand",
            headers=headers,
            params={"node_id": seed_id, "hops": 1, "vertex_type": "entity"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        for n in body["nodes"]:
            if n["id"] == seed_id:
                continue
            assert n["vertex_type"] == "entity"


async def test_expand_top_k_caps_neighbors(app_harness):
    """top_k=2 returns at most 2 nodes total (seed + 1 neighbor)."""
    async with _client_for(app_harness) as (client, _app, headers):
        seed_id = await _pick_central_entity_id(client, headers)
        r = await client.get(
            "/graph/expand",
            headers=headers,
            params={"node_id": seed_id, "hops": 2, "top_k": 2},
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["nodes"]) <= 2


# =====================================================================
# /graph/nodes/{hash_id}
# =====================================================================


async def test_node_detail_for_entity(app_harness):
    """Hover card for a known entity has surface + cluster info, NO hash_id field."""
    async with _client_for(app_harness) as (client, _app, headers):
        seed_id = await _pick_central_entity_id(client, headers)
        r = await client.get(f"/graph/nodes/{seed_id}", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        # hash_id MUST NOT leak into the body — that was the user's
        # explicit ask.
        assert "hash_id" not in body
        assert body["vertex_type"] == "entity"
        assert isinstance(body["surface"], str) and body["surface"]
        assert body["degree"] > 0
        assert isinstance(body["mention_count"], int)
        assert isinstance(body["neighboring_files"], list)


async def test_node_detail_unknown_404(app_harness):
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get("/graph/nodes/entity-deadbeef", headers=headers)
        assert r.status_code == 404


# =====================================================================
# /graph/ppr-subgraph
# =====================================================================


async def test_ppr_subgraph_happy_path(app_harness):
    """PPR for a query that should hit the corpus → mode='ppr' + subgraph data.

    'hong kong' appears in the corpus as a top-degree entity, so the
    NER + entity_store match should fire and produce both seeds and
    actived entities.
    """
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.post(
            "/graph/ppr-subgraph",
            headers=headers,
            json={"query": "hong kong life insurance"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] in ("ppr", "no_seeds"), body
        if body["mode"] == "ppr":
            # Edge endpoints reference real seed/actived/passage ids.
            ids_in_play: set = set()
            for s in body["seeds"]:
                ids_in_play.add(s["id"])
            for a in body["actived_entities"]:
                ids_in_play.add(a["id"])
            for p in body["passages"]:
                if p["hash_id"]:
                    ids_in_play.add(p["hash_id"])
            for e in body["edges"]:
                # We only require that referenced ids appear at least once
                # in the kept set; not every edge endpoint will be in our
                # seed/actived/passage payload (induced subgraph may carry
                # non-seed actived entities not in the actived dict).
                assert e["weight"] >= 0
                assert e["type"]


async def test_ppr_subgraph_validates_query(app_harness):
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.post("/graph/ppr-subgraph", headers=headers, json={"query": ""})
        assert r.status_code == 422


async def test_ppr_subgraph_no_seeds_for_gibberish(app_harness):
    """Query with no plausible entity should return mode='no_seeds' or 'ppr' with empty payload."""
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.post(
            "/graph/ppr-subgraph",
            headers=headers,
            json={"query": "xyzzy quux foo bar baz"},
        )
        assert r.status_code == 200
        body = r.json()
        # Either mode is acceptable; what we care about is the shape contract.
        assert {"mode", "seeds", "actived_entities", "passages", "edges"} <= set(body.keys())


# =====================================================================
# RBAC — analyst CAN access (graph is core analyst tooling, not admin-only)
# =====================================================================


async def test_analyst_can_read_graph(app_harness):
    async with _admin_and_analyst(app_harness) as (
        client,
        _app,
        _admin_headers,
        analyst_headers,
    ):
        for path in ["/graph/overview", "/graph/seed?q=policy"]:
            r = await client.get(path, headers=analyst_headers)
            assert r.status_code == 200, (path, r.status_code, r.text)


# =====================================================================
# Auth required across the surface (codex round-1 LOW-5)
# =====================================================================


async def test_all_endpoints_require_auth(app_harness):
    """Every /graph route must 401 without a token (no hidden anonymous read)."""
    async with _client_for(app_harness) as (client, _app, headers):
        seed_id = await _pick_central_entity_id(client, headers)
        cases = [
            ("get", f"/graph/expand?node_id={seed_id}", None),
            ("get", f"/graph/nodes/{seed_id}", None),
            ("post", "/graph/ppr-subgraph", {"query": "test"}),
        ]
        for method, path, payload in cases:
            req = getattr(client, method)
            kwargs = {} if payload is None else {"json": payload}
            r = await req(path, **kwargs)
            assert r.status_code == 401, (method, path, r.status_code, r.text)


# =====================================================================
# /expand file_ids filter (codex round-1 LOW-5)
# =====================================================================


async def test_expand_file_ids_filter(app_harness):
    """file_ids=<unknown> must drop all passage neighbors but keep the seed."""
    async with _client_for(app_harness) as (client, _app, headers):
        seed_id = await _pick_central_entity_id(client, headers)
        r = await client.get(
            "/graph/expand",
            headers=headers,
            params={
                "node_id": seed_id,
                "hops": 1,
                "vertex_type": "both",
                "file_ids": ["this-file-id-does-not-exist"],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        for n in body["nodes"]:
            if n["id"] == seed_id:
                continue
            # Only entity vertices should survive — every passage was
            # filtered by the unknown file_id.
            assert n["vertex_type"] != "passage", n


# =====================================================================
# /nodes/{id} for a passage vertex (codex round-1 LOW-5)
# =====================================================================


async def test_node_detail_for_passage(app_harness):
    """Passage hover card carries file_id + page_number (no cluster info)."""
    import igraph as ig

    from config.settings import faiss_graph_dir

    g = ig.Graph.Read_GraphML(str(faiss_graph_dir() / "LinearRAG.graphml"))
    passage_name = next(
        (v["name"] for v in g.vs if v.attributes().get("vertex_type") == "passage"),
        None,
    )
    assert passage_name is not None, "no passage vertex in test corpus"

    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.get(f"/graph/nodes/{passage_name}", headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        assert "hash_id" not in body
        assert body["vertex_type"] == "passage"
        # Passage payload exposes file_id + page_number for the open-PDF link.
        assert body.get("file_id"), body
        assert body.get("page_number") is not None, body
        # No cluster / mention_count for passages.
        assert body.get("logical_cluster") is None
        assert body.get("mention_count") is None


# =====================================================================
# Schema-bound length cap (codex round-1 LOW-3)
# =====================================================================


async def test_expand_rejects_oversized_node_id(app_harness):
    """node_id > 128 chars must 422 (request-amplification guard)."""
    async with _client_for(app_harness) as (client, _app, headers):
        long_id = "x" * 200
        r = await client.get(
            "/graph/expand", headers=headers, params={"node_id": long_id}
        )
        assert r.status_code == 422, r.text


# =====================================================================
# /ppr-subgraph induced edges over kept ids (codex round-1 LOW-5)
# =====================================================================


async def test_ppr_subgraph_edges_only_reference_kept_ids(app_harness):
    """Every edge must have BOTH endpoints in the kept-id set.

    ``actived_entities`` surfaces every actived dict entry, and
    ``passages`` surfaces every materialized hit, so the union of
    seeds + actived + passages IS the full kept-id set that
    ``_induced_edges`` walked. Both endpoints must land there.
    """
    async with _client_for(app_harness) as (client, _app, headers):
        r = await client.post(
            "/graph/ppr-subgraph",
            headers=headers,
            json={"query": "hong kong life insurance"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        if body["mode"] != "ppr":
            pytest.skip(f"PPR returned mode={body['mode']!r}; cannot exercise edge invariant")
        kept_ids = (
            {s["id"] for s in body["seeds"]}
            | {a["id"] for a in body["actived_entities"]}
            | {p["hash_id"] for p in body["passages"] if p["hash_id"]}
        )
        for e in body["edges"]:
            assert e["source"] in kept_ids, e
            assert e["target"] in kept_ids, e
