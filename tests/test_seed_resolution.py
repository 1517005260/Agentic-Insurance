"""Unit tests for hybrid seed resolution (embedding recall + IDF precision).

Covers the pure ``_hybrid_select`` helper without a channel: bare embedding
top-1 surfaces high-confidence WRONG matches ("Warner Bros. Records" →
"hollywood records" at 0.901); the IDF lexical-overlap gate rescues precision,
and the synonym escape keeps true same-surface exact matches. The helper is
module-level and channel-free so it is testable on synthetic candidates.
"""
from ingestion.index.linear_rag.disambig import build_surface_idf
from agentic.tools.acquisition.graph_explore.base import _hybrid_select


_IDF = build_surface_idf(
    ["hollywood records", "warner records", "sony records", "alum", "afl", "united states"]
)
_OVERLAP_MIN = 0.15
_SYN_SIM = 0.90


def test_hybrid_select_lexical_overrides_higher_embedding():
    # Embedding ranks "hollywood records" (0.901) above "warner records"
    # (0.899), but lexical overlap on the distinctive head token "warner"
    # picks the correct entity.
    candidates = [
        ("h1", "hollywood records", 0.901),
        ("h2", "warner records", 0.899),
        ("h3", "sony records", 0.878),
    ]
    picked = _hybrid_select(
        "warner bros. records", candidates, _IDF, _OVERLAP_MIN, _SYN_SIM
    )
    assert picked is not None
    assert picked["hash_id"] == "h2"
    assert picked["resolved_by"] == "lexical"


def test_hybrid_select_abstains_on_no_overlap_below_synonym():
    # "aluf" shares no IDF-bearing token with "alum"/"afl" and the top NN
    # (0.856) is below the synonym floor → abstain.
    candidates = [("h1", "alum", 0.856), ("h2", "afl", 0.83)]
    picked = _hybrid_select(
        "aluf", candidates, _IDF, _OVERLAP_MIN, _SYN_SIM
    )
    assert picked is None


def test_hybrid_select_exact_match():
    candidates = [("h1", "united states", 0.97)]
    picked = _hybrid_select(
        "united states", candidates, _IDF, _OVERLAP_MIN, _SYN_SIM
    )
    assert picked is not None
    assert picked["hash_id"] == "h1"
    assert picked["overlap"] == 1.0
