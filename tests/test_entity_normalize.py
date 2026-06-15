"""Regression tests for the entity normalisation + chain-split pipeline.

Bracket-repair and chain-split handle OCR fragments like
``"万通危疾加护保("`` and chained spans like
``"… (PHPS)、… (PHPJ)、… (BIS)"``. These tests pin the expected
behaviour so a tweak to the regex set or normalisation order
can't silently regress the entity universe.
"""
from ingestion.index.linear_rag.disambig import (
    is_composite_surface,
    surface_quality_score,
)
from ingestion.index.linear_rag.ner import split_catalog_mentions
from ingestion.index.linear_rag.normalize import (
    cleanup,
    is_junk,
    normalize_for_hash,
)


# ============================================================ cleanup


class TestCleanup:
    def test_strips_dangling_open_bracket(self):
        # spaCy splits "万通危疾加护保(优越版)" mid-token sometimes; the
        # left half lands as an entity surface ending with a lone "(".
        assert cleanup("万通危疾加护保(") == "万通危疾加护保"
        assert cleanup("vip 环球医疗保(") == "vip 环球医疗保"
        # Full-width ( as well.
        assert cleanup("丰裕医疗保（") == "丰裕医疗保"

    def test_keeps_balanced_brackets_intact(self):
        # Properly bracketed product codes are part of the canonical
        # surface and must NOT be stripped.
        assert cleanup("万通多元教育储蓄计划 (mfs)") == "万通多元教育储蓄计划 (mfs)"
        assert cleanup("万通危疾爱护保(phpj)") == "万通危疾爱护保(phpj)"

    def test_strips_trailing_sentence_punct(self):
        assert cleanup("万通危疾加护保 (优越版) 保单。") == "万通危疾加护保 (优越版) 保单"
        assert cleanup("中国平安,") == "中国平安"

    def test_handles_punct_then_open_bracket(self):
        # Two-pass trim: ``保单。(`` strips the period first, then
        # finds and strips the dangling open bracket.
        assert cleanup("万通危疾加护保 保单。(") == "万通危疾加护保 保单"

    def test_idempotent_on_clean_surfaces(self):
        for s in ["香港", "万通保险", "vip 环球医疗保 (vwm)"]:
            assert cleanup(s) == s


# ============================================================ chain split


class TestSplitCatalogMentions:
    def test_chinese_list_separator_dun(self):
        # 、 is the canonical Chinese list separator — split.
        pieces = split_catalog_mentions(
            "万通危疾加护保(优越版)(phps)、万通危疾爱护保(phpj)、富饶万家储蓄保险计划(bis)"
        )
        assert pieces == [
            "万通危疾加护保(优越版)(phps)",
            "万通危疾爱护保(phpj)",
            "富饶万家储蓄保险计划(bis)",
        ]

    def test_full_width_comma_splits(self):
        pieces = split_catalog_mentions("美元，港元，人民币")
        assert pieces == ["美元", "港元", "人民币"]

    def test_bullet_separator_splits(self):
        pieces = split_catalog_mentions("万通危疾加护保(优越版) • 万通危疾爱护保")
        assert pieces == ["万通危疾加护保(优越版)", "万通危疾爱护保"]

    def test_half_width_comma_does_NOT_split(self):
        # ASCII comma is risky to split on (it appears inside English
        # company names like "ABC, Inc."), so we leave it alone. The
        # downstream cleanup may still trim trailing junk per piece.
        pieces = split_catalog_mentions("ABC, Inc.")
        assert pieces == ["ABC, Inc."]

    # ---- regression: conjunction words MUST NOT split ----

    def test_conjunction_huo_does_NOT_split(self):
        # ``或`` is intentionally excluded from the split set:
        # conjunctions are real words that can appear inside legitimate
        # organisation names. The composite surface stays whole and is
        # later excluded from alias edges via ``is_composite_surface``
        # instead.
        pieces = split_catalog_mentions("万通多元终身年金(mfa)或万通多元教育储蓄计划")
        assert pieces == ["万通多元终身年金(mfa)或万通多元教育储蓄计划"]

    def test_conjunction_ji_does_NOT_split_real_org_names(self):
        # Realistic counter-example: 保险及再保险公司 / 联通及电信合作社
        # — valid composite organisation names that must survive intact.
        for legit in ("保险及再保险公司", "联通及电信合作社", "中国及香港销售部"):
            assert split_catalog_mentions(legit) == [legit]

    def test_conjunction_yu_does_NOT_split(self):
        assert split_catalog_mentions("研究与开发部") == ["研究与开发部"]

    # ---- empties ----

    def test_no_separator_yields_original(self):
        pieces = split_catalog_mentions("万通保险")
        assert pieces == ["万通保险"]

    def test_empty_input_yields_empty_list(self):
        assert split_catalog_mentions("") == []
        assert split_catalog_mentions("   ") == []


# ============================================================ end-to-end


class TestNormalizeForHashRegressions:
    """The full cleanup→junk→canonical_form pipeline on real inputs from
    the local PaddleOCR corpus's NER output, asserting they collapse to
    the expected canonical surface (so PPR and graph dedupe see one
    entity instead of three near-duplicates)."""

    def test_dangling_open_collapses_to_canonical(self):
        # These three surface forms must collapse to one canonical key.
        a = normalize_for_hash("万通危疾加护保(")
        b = normalize_for_hash("万通危疾加护保")
        assert a == b == "万通危疾加护保"

    def test_unwanted_junk_still_dropped(self):
        # The junk filter must still reject pure-symbol or too-short
        # surfaces — bracket repair must not accidentally rescue them.
        assert is_junk("(")
        assert is_junk("。")
        assert normalize_for_hash("(") is None

    def test_full_width_punct_canonicalises(self):
        # NFKC keeps full-width brackets as full-width visually, so the
        # canonical key includes them. The point of this test is to
        # confirm we don't crash and that fold to lowercase still hits
        # embedded latin tokens like (PHPJ) → (phpj).
        out = normalize_for_hash("万通危疾爱护保 (PHPJ)")
        assert out == "万通危疾爱护保 (phpj)"


# ============================================================ canonical picker


class TestSurfaceQualityScore:
    """The scoring used by ``compute_clusters`` to pick a cluster's
    canonical surface. Higher = cleaner. The actual numbers don't
    matter; what matters is the **ordering** between common shapes."""

    def test_clean_short_beats_chained_long(self):
        # A multi-product chain must not outrank the clean family name
        # as the canonical surface.
        clean = "万通危疾加护保"
        chain = "万通危疾加护保(优越版)(phps)、万通危疾爱护保(phpj)、富饶万家储蓄保险计划(bis)"
        assert surface_quality_score(clean) > surface_quality_score(chain)

    def test_clean_short_beats_dangling_bracket(self):
        clean = "vip 环球医疗保"
        bad = "vip 环球医疗保("
        assert surface_quality_score(clean) > surface_quality_score(bad)

    def test_clean_short_beats_sentence_fragment(self):
        clean = "万通危疾加护保"
        sent = "万通危疾加护保 (优越版) (php5) 或万通危疾爱护保 (php5) 保单。"
        assert surface_quality_score(clean) > surface_quality_score(sent)

    def test_balanced_brackets_score_OK(self):
        # The SKU-specific surface should be slightly worse than the
        # family-level one (length penalty), but better than chained
        # or dangling — i.e. it's a fine "second-best" canonical when
        # the cluster only has SKU-level members.
        family = "万通危疾加护保"
        sku = "万通危疾加护保(优越版)"
        chain = "万通危疾加护保(优越版)、万通危疾爱护保(phpj)"
        assert surface_quality_score(family) > surface_quality_score(sku)
        assert surface_quality_score(sku) > surface_quality_score(chain)

    def test_empty_surface_floor(self):
        # Defensive: empty / None should be the worst possible score.
        assert surface_quality_score("") < -1e5
        assert surface_quality_score(None) < -1e5  # type: ignore[arg-type]


def _alias_cluster_with_members(members):
    """Build a 1-cluster igraph for testing compute_clusters' picker.

    Each member becomes one vertex; alias edges connect them in a
    star so they end up in the same connected component. ``content``
    attribute = the surface (the picker reads it via vertex attrs)."""
    import igraph as ig

    from ingestion.index.linear_rag.disambig import ALIAS_EDGE_TYPE

    g = ig.Graph(directed=False)
    g.add_vertices(len(members))
    for i, surf in enumerate(members):
        g.vs[i]["name"] = f"entity-{i:02d}"
        g.vs[i]["content"] = surf
    edges = [(0, j) for j in range(1, len(members))]
    g.add_edges(edges)
    g.es["edge_type"] = [ALIAS_EDGE_TYPE] * len(edges)
    return g


class TestComputeClustersCanonical:
    def test_picks_cleanest_surface_over_longest(self):
        from ingestion.index.linear_rag.disambig import compute_clusters

        # The clean family name should win over the long composite chain.
        g = _alias_cluster_with_members([
            "万通危疾加护保",
            "万通危疾加护保(优越版)(phps)、万通危疾爱护保(phpj)、富饶万家储蓄保险计划(bis)",
            "万通危疾加护保(",
        ])
        clusters = compute_clusters(g)
        assert len(clusters) == 1
        assert clusters[0]["canonical"] == "万通危疾加护保"

    def test_picks_balanced_sku_when_no_family_member(self):
        from ingestion.index.linear_rag.disambig import compute_clusters

        # If the cluster contains only SKU-tagged surfaces, the cleanest
        # SKU one wins (here the bracket-balanced single-bracket form).
        # Use the post-cleanup form of the sentence fragment (``保单``,
        # not ``保单。``) — that's what's actually in the live graph
        # after the ingest pipeline runs cleanup() on each surface.
        # If we tested the un-cleanup'd form we'd hide the case where a
        # sentence fragment ending in plain text could beat a balanced
        # SKU just on length.
        g = _alias_cluster_with_members([
            "万通危疾加护保(优越版)",
            "万通危疾加护保(",
            "万通危疾加护保 (优越版) (php5) 保单",
        ])
        clusters = compute_clusters(g)
        assert clusters[0]["canonical"] == "万通危疾加护保(优越版)"

    def test_balanced_sku_beats_post_cleanup_sentence_fragment(self):
        """A balanced SKU must outrank a post-cleanup sentence fragment:
        ``_TRAILING_JUNK_RE`` must not include ``)``, which would
        unfairly penalise a balanced SKU."""
        from ingestion.index.linear_rag.disambig import surface_quality_score

        sku = "万通危疾加护保(优越版)"
        cleaned_fragment = "万通危疾加护保 (优越版) (php5) 保单"
        assert surface_quality_score(sku) > surface_quality_score(cleaned_fragment), (
            f"SKU score {surface_quality_score(sku)} must beat "
            f"cleaned fragment {surface_quality_score(cleaned_fragment)}"
        )


# ============================================================ composite gate


class TestIsCompositeSurface:
    """Admission gate that excludes multi-mention spans from alias-edge
    generation. Critical for preventing garbage-bucket cluster
    pollution."""

    # ---- positives: surfaces that MUST be flagged composite ----

    def test_interior_list_separator_is_composite(self):
        # Defensive: split_catalog_mentions should have caught this,
        # but if a `、` survives into the alias path we want to refuse.
        assert is_composite_surface("万通危疾加护保、万通危疾爱护保")

    def test_conjunction_plus_two_brackets_is_composite(self):
        # The exact pattern that motivated this rule —
        # ``A(code1)或B(code2)`` survived split_catalog_mentions
        # (we don't split on 或 anymore) and would otherwise pollute.
        assert is_composite_surface("万通多元终身年金(mfa)或万通多元教育储蓄计划(mfs)")
        assert is_composite_surface(
            "万通危疾加护保(优越版)(phps)或万通危疾爱护保(phpj)"
        )

    def test_half_width_comma_plus_two_brackets_is_composite(self):
        assert is_composite_surface(
            "万通危疾爱护保(phpj),并选择一(phps)"
        )

    def test_three_or_more_open_brackets_is_composite(self):
        # Even without any separator we can still detect chains by
        # the sheer bracket count.
        assert is_composite_surface(
            "万通危疾加护保(优越版)(phps) 万通危疾爱护保(phpj)"
        )

    # ---- negatives: legitimate surfaces that MUST NOT be flagged ----

    def test_clean_family_name_is_not_composite(self):
        for s in ["万通危疾加护保", "vip 环球医疗保", "美国万通"]:
            assert not is_composite_surface(s)

    def test_single_bracket_pair_is_not_composite(self):
        # Standard SKU surface — one product + one bracketed code.
        assert not is_composite_surface("万通危疾加护保(优越版)")
        assert not is_composite_surface("万通危疾爱护保 (phpj)")

    def test_two_brackets_NO_composite_signals_allowed(self):
        # (优越版)(phps) is 2 brackets but represents ONE SKU with
        # a variant + code — there is no conjunction or comma to
        # corroborate the "two mentions" hypothesis, so we don't flag.
        assert not is_composite_surface("万通危疾加护保(优越版)(phps)")

    def test_conjunction_inside_real_org_name_not_composite(self):
        # 保险及再保险公司 has 或/及 but ZERO brackets — the rule
        # requires brackets to corroborate, so these survive.
        for s in ["保险及再保险公司", "联通及电信合作社", "研究与开发部"]:
            assert not is_composite_surface(s)

    def test_empty_or_none_safe(self):
        assert not is_composite_surface("")
        assert not is_composite_surface(None)  # type: ignore[arg-type]


# ============================================================ cluster cache versioning


class TestClusterCacheVersioning:
    """When the canonical-picker algorithm changes, cache files written
    under an older ``CLUSTERS_CACHE_VERSION`` MUST be silently
    invalidated so the upgraded binary doesn't keep serving stale
    longest-surface canonicals."""

    def test_v1_cache_is_invalidated_and_recomputed(self, tmp_path):
        import json
        import igraph as ig
        from ingestion.index.linear_rag.disambig import (
            ALIAS_EDGE_TYPE,
            CLUSTERS_CACHE_VERSION,
            get_clusters,
        )

        # Pretend an older binary wrote a v1 cache with the
        # longest-surface canonical (the bug we're trying to fix).
        cache_path = tmp_path / "clusters.json"
        cache_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "alias_edge_count": 1,
                    "clusters": [
                        {
                            "id": "c_0000",
                            "members": ["entity-a", "entity-b"],
                            "canonical": "万通危疾加护保(优越版)(phps)、…(bis)",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        # Build a tiny live graph with the same two entities aliased.
        g = ig.Graph(directed=False)
        g.add_vertices(2)
        g.vs[0]["name"] = "entity-a"
        g.vs[0]["content"] = "万通危疾加护保(优越版)(phps)、万通危疾爱护保(phpj)、富饶万家(bis)"
        g.vs[1]["name"] = "entity-b"
        g.vs[1]["content"] = "万通危疾加护保"
        g.add_edges([(0, 1)])
        g.es["edge_type"] = [ALIAS_EDGE_TYPE]

        clusters = get_clusters(g, cache_path)
        # Recomputed under v2 → should pick the clean family name as
        # canonical, not the long composite chain from the v1 cache.
        assert clusters and clusters[0]["canonical"] == "万通危疾加护保"

        # And the cache file is now at v2 so subsequent reads hit cache.
        new_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        assert new_payload["version"] == CLUSTERS_CACHE_VERSION

    def test_current_version_cache_is_trusted(self, tmp_path):
        import json
        from ingestion.index.linear_rag.disambig import (
            CLUSTERS_CACHE_VERSION,
            get_clusters,
        )

        cache_path = tmp_path / "clusters.json"
        cached_clusters = [
            {"id": "c_0000", "members": ["x", "y"], "canonical": "X"}
        ]
        cache_path.write_text(
            json.dumps(
                {
                    "version": CLUSTERS_CACHE_VERSION,
                    "alias_edge_count": 1,
                    "clusters": cached_clusters,
                }
            ),
            encoding="utf-8",
        )

        # Pass a deliberately empty graph — if get_clusters trusted the
        # cache it returns the cached list, otherwise it returns [].
        import igraph as ig
        g = ig.Graph(directed=False)
        result = get_clusters(g, cache_path)
        assert result == cached_clusters
