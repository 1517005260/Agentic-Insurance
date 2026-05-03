"""Semantic validation of LLM-proposed claim candidates.

Takes a raw payload + the source observation and returns either a
fully-populated :class:`Claim` (with a fresh id slot of ``""`` to be
assigned by the evidence store) or an :class:`ErrorEnvelope` dict.

Validation responsibilities, in order:

1. Shape (claim_type, scope, unit_type, citations).
2. Predicate / score spec resolution.
3. Citation snapshot existence + span match.
4. Universe membership (positive / negative units in inventory).
5. ScanClaim — narrowing-source guard, then full-coverage partition
   match against observation.
6. WitnessClaim — predicate-on-content evaluation per cited unit,
   value_map score round-trip when score is declared.

This module replaces plant.py's monolithic ``_build_claim_candidate``
so all claim-validation logic lives in one place. Plant retains a
thin wrapper (``_build_claim_candidate``) that calls
:func:`validate_claim_candidate` with the right helpers.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from agentic.proof import predicate as pr
from agentic.proof.score import registry as sr
from agentic.proof.evidence.extractors import _aggregate_page_hits_to_units
from agentic.proof.evidence.observation import normalise as _ob_normalise
from agentic.proof.types import (
    Citation,
    Claim,
    ClaimType,
    Observation,
    ObservationType,
    PredicateSpec,
    ScopeRef,
)

if TYPE_CHECKING:
    from agentic.proof.plant import Plant


_ErrFn = Callable[..., Dict[str, Any]]


# Plant-side scope/predicate/score resolvers are passed in so we don't
# import plant (would be circular). Each resolver returns
# ``(value, error)`` per the existing plant convention.
ScopeResolver = Callable[[Dict[str, Any]], Tuple[Optional[ScopeRef], Optional[Dict[str, Any]]]]
PredicateResolver = Callable[[Dict[str, Any]], Tuple[Optional[PredicateSpec], Optional[Dict[str, Any]]]]
ScoreResolver = Callable[[Optional[Dict[str, Any]]], Tuple[Any, Optional[Dict[str, Any]]]]


def validate_claim_candidate(
    payload: Dict[str, Any],
    observation: Observation,
    *,
    plant: "Plant",
    _err: _ErrFn,
    resolve_scope: ScopeResolver,
    resolve_predicate: PredicateResolver,
    resolve_score: ScoreResolver,
) -> Tuple[Optional[Claim], Optional[Dict[str, Any]]]:
    """Validate ``payload`` against ``observation`` and return the
    materialised :class:`Claim` (with id slot ``""`` for the evidence
    store to mint) or an error envelope."""
    if not isinstance(payload, dict):
        return None, _err(
            "invalid_claim", "claim_candidate must be a dict",
            remediation="Pass `claim_candidate` as a JSON object containing claim_type, scope, unit_type, positive_units, and citations (see evidence_ingest's schema for full shape).",
            valid_example={"claim_type": "WitnessClaim", "scope": {"file_ids": ["<fid>"], "section_ids": None, "sealed": False}, "unit_type": "file", "predicate": {"name": "contains_string", "args": {"pattern": "X"}}, "positive_units": ["<fid>"], "citations": [{"file_id": "<fid>", "page_id": "p_0001", "span": "<verbatim text>"}]},
        )
    claim_type_raw = payload.get("claim_type")
    try:
        claim_type = ClaimType(claim_type_raw)
    except ValueError:
        return None, _err(
            "invalid_claim_type", f"claim_type={claim_type_raw!r} not recognised",
            remediation="Set `claim_type` to 'WitnessClaim' (per-unit citation) or 'ScanClaim' (full-coverage partition from an exhaustive scan).",
            valid_example={"claim_type": "WitnessClaim"},
        )
    if claim_type not in (ClaimType.WITNESS, ClaimType.SCAN):
        return None, _err(
            "non_certifying_claim_type",
            f"v1 LLM ingest only accepts WitnessClaim/ScanClaim; got {claim_type.value}",
            remediation="Re-emit with claim_type='WitnessClaim' or 'ScanClaim'; other claim types (e.g. EvidenceClaim) are not LLM-ingestable in v1.",
        )
    if (
        claim_type == ClaimType.SCAN
        and observation.observation_type is not ObservationType.PAGE_HITS_EXHAUSTIVE
    ):
        return None, _err(
            "scanclaim_requires_exhaustive",
            f"ScanClaim must source from PAGE_HITS_EXHAUSTIVE; got {observation.observation_type.value}",
            remediation="Source ScanClaim from a pattern_search observation (which is exhaustive by construction). For non-exhaustive observations, use WitnessClaim instead.",
        )

    scope, err = resolve_scope(payload.get("scope") or {})
    if err is not None:
        return None, err
    unit_type = payload.get("unit_type")
    if unit_type not in ("file", "section"):
        return None, _err(
            "invalid_unit_type", "unit_type must be 'file' or 'section'",
            remediation="Set unit_type to 'file' or 'section' — match the obligation you intend this claim to bind to (no 'page' value is accepted).",
            valid_example={"unit_type": "file"},
        )
    predicate_payload = payload.get("predicate")
    predicate: Optional[PredicateSpec] = None
    if predicate_payload is not None:
        predicate, err = resolve_predicate(predicate_payload)
        if err is not None:
            return None, err
    score_payload = payload.get("score")
    score, err = resolve_score(score_payload)
    if err is not None:
        return None, err

    positive_units = list(payload.get("positive_units") or [])
    negative_units = list(payload.get("negative_units") or [])
    if len(positive_units) != len(set(positive_units)):
        return None, _err(
            "duplicate_units", "positive_units must not contain duplicates",
            remediation="Deduplicate positive_units before resubmitting — each unit_id must appear at most once.",
        )
    if len(negative_units) != len(set(negative_units)):
        return None, _err(
            "duplicate_units", "negative_units must not contain duplicates",
            remediation="Deduplicate negative_units before resubmitting — each unit_id must appear at most once.",
        )
    value_map = dict(payload.get("value_map") or {})

    # --- citations ---------------------------------------------------
    citations_raw = payload.get("citations") or []
    citations: List[Citation] = []
    for c in citations_raw:
        if not isinstance(c, dict) or "file_id" not in c or "page_id" not in c:
            return None, _err(
                "invalid_citation", "each citation must include file_id and page_id",
                remediation="Re-emit each citation as {file_id, page_id, span?} — both file_id and page_id are required; span is the verbatim text snippet to anchor the witness.",
                valid_example={"file_id": "<file_id>", "page_id": "p_0001", "span": "<verbatim text>"},
            )
        citations.append(Citation(
            file_id=str(c["file_id"]),
            page_id=str(c["page_id"]),
            span=c.get("span"),
        ))
    if not citations:
        return None, _err(
            "missing_citations", "at least one citation required",
            remediation="Add at least one citation entry pointing at a page that the observation actually fetched.",
            valid_example=[{"file_id": "<file_id>", "page_id": "p_0001", "span": "<verbatim text>"}],
        )

    inventory = plant.inventory
    page_store = inventory.page_store
    normalised = plant.normalised(observation)

    from agentic.proof.evidence.observation import citation_in as _ob_citation_in
    for ct in citations:
        if not _ob_citation_in(ct, normalised, page_store):
            gid = f"{ct.file_id}/{ct.page_id}"
            entry = normalised.entries.get(gid)
            referenced = gid in normalised.referenced_units
            ctx: Dict[str, Any] = {
                "citation_file_id": ct.file_id,
                "citation_page_id": ct.page_id,
                "page_referenced_by_observation": referenced,
                "observation_referenced_pages": sorted(normalised.referenced_units)[:10],
            }
            if entry is not None and entry.text:
                # Echo the FIRST 240 chars of the recorded text snapshot
                # so the LLM doesn't have to scroll back through history
                # to find a valid span on this page.
                ctx["observation_excerpt"] = entry.text[:240]
                ctx["observation_excerpt_truncated"] = len(entry.text) > 240
            if referenced and ct.span is not None:
                msg = (
                    f"citation {gid}: span not present on the cited page "
                    f"(observation has the page, but its text doesn't contain your span)"
                )
                rem = "Copy a verbatim substring from `observation_excerpt` as your span — that's the text the gate validates against."
            elif not referenced:
                msg = (
                    f"citation {gid}: page is NOT in this observation's referenced set "
                    f"(see `observation_referenced_pages`)"
                )
                rem = "Re-call read_page on the page you want to cite, then pass that read_page's observation_id (search-tool observations carry only snippets, not the full page text required for citation)."
            else:
                msg = f"citation {gid} not found in observation payload"
                rem = "Re-call read_page on the page_id you want to cite and use the new observation_id."
            return None, _err(
                "citation_mismatch", msg, remediation=rem, **ctx,
            )

    # --- universe checks --------------------------------------------
    domain_files = list(scope.file_ids)  # type: ignore[union-attr]
    units_universe = set(inventory.units(
        unit_type,                       # type: ignore[arg-type]
        file_ids=domain_files,
        section_ids=list(scope.section_ids) if scope.section_ids else None,  # type: ignore[union-attr]
    ))
    for u in positive_units:
        if u not in units_universe:
            return None, _err(
                "unknown_unit", f"positive unit {u!r} not in inventory.units",
                remediation="Use unit_ids returned by inventory tools — for unit_type='file' that's the file_id; for 'section' that's the '<file_id>:sec_NNN' from toc. Drop unknown ids from positive_units.",
                unit_id=u,
            )
    for u in negative_units:
        if u not in units_universe:
            return None, _err(
                "unknown_unit", f"negative unit {u!r} not in inventory.units",
                remediation="Use unit_ids returned by inventory tools — for unit_type='file' that's the file_id; for 'section' that's the '<file_id>:sec_NNN' from toc. Drop unknown ids from negative_units.",
                unit_id=u,
            )

    if claim_type == ClaimType.SCAN:
        err = _validate_scan(
            scope=scope,
            unit_type=unit_type,
            positive_units=positive_units,
            negative_units=negative_units,
            normalised=normalised,
            observation=observation,
            inventory=inventory,
            _err=_err,
        )
        if err is not None:
            return None, err

    if claim_type == ClaimType.WITNESS:
        err = _validate_witness(
            predicate=predicate,
            score=score,
            unit_type=unit_type,
            positive_units=positive_units,
            value_map=value_map,
            citations=citations,
            observation=observation,
            plant=plant,
            _err=_err,
        )
        if err is not None:
            return None, err

    claim = Claim(
        id="",
        observation_id=observation.id,
        claim_type=claim_type,
        scope=scope,                      # type: ignore[arg-type]
        unit_type=unit_type,              # type: ignore[arg-type]
        predicate=predicate,
        score=score,                      # type: ignore[arg-type]
        positive_units=positive_units,
        negative_units=negative_units,
        value_map=value_map,
        citations=citations,
        derivation="llm_proposed",
    )
    return claim, None


def _validate_scan(
    *,
    scope: ScopeRef,
    unit_type: str,
    positive_units: List[str],
    negative_units: List[str],
    normalised,
    observation: Observation,
    inventory,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    pos_set = set(positive_units)
    neg_set = set(negative_units)
    if pos_set & neg_set:
        return _err(
            "scan_overlap", "positive and negative units overlap",
            remediation="Drop your manual ScanClaim — the auto-extracted ScanClaim from pattern_search already has a clean partition; check the observation's auto_extract_claim_ids. If you must build it manually, ensure positive ∩ negative = ∅.",
        )
    scan_meta = normalised.scan_meta
    if scan_meta is not None and scan_meta.narrowing_sources:
        return _err(
            "scanclaim_from_narrowed_scan",
            f"observation narrowed by {sorted(scan_meta.narrowing_sources)}; "
            f"a ScanClaim sourced here cannot certify completeness over a wider universe",
            remediation=(
                "Re-run pattern_search WITHOUT a narrowing axis "
                "(remove page_range / section_ids / result_limit, and ensure "
                "file_ids covers the full corpus you want to certify), then "
                "ingest the new observation's auto-extracted ScanClaim."
            ),
            narrowing_sources=sorted(scan_meta.narrowing_sources),
        )
    obs_pl = observation.payload or {}
    scanned_pages = list(obs_pl.get("scanned_units") or obs_pl.get("scanned_pages") or [])
    positive_pages = list(obs_pl.get("positive_units") or [])
    negative_pages = list(obs_pl.get("negative_units") or [])
    scanned_set = set(scanned_pages)
    if unit_type == "file":
        claimed_files = set(scope.file_ids)
        for fid in claimed_files:
            file_pages = {
                gid for gid in inventory.page_store.ids()
                if gid.startswith(f"{fid}/")
            }
            missing = file_pages - scanned_set
            if missing:
                return _err(
                    "scanclaim_coverage_incomplete",
                    f"observation did not scan every indexed page of {fid!r}; "
                    f"missing {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}",
                    remediation="Re-run pattern_search WITHOUT a page_range/section_ids filter (or widen the scope to cover every page of the file), then ingest the new observation's auto-extracted ScanClaim.",
                    file_id=fid,
                    missing_pages=sorted(missing)[:10],
                )
    else:
        claimed_sections = list(scope.section_ids or [])
        for sid in claimed_sections:
            sec = inventory.get(sid)
            if sec is None:
                return _err(
                    "unknown_section_id", f"section {sid!r} not in inventory",
                    remediation="Call toc(file_id=...) to refresh section ids, then use one of the returned '<file_id>:sec_NNN' ids.",
                    unknown_section_id=sid,
                )
            section_pages = {
                f"{sec.file_id}/p_{p:04d}"
                for p in range(sec.page_start, sec.page_end + 1)
            }
            indexed = {
                gid for gid in section_pages
                if inventory.page_store.get(gid) is not None
            }
            missing = indexed - scanned_set
            if missing:
                return _err(
                    "scanclaim_coverage_incomplete",
                    f"observation did not scan every page of section {sid!r}; "
                    f"missing {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}",
                    remediation="Re-run pattern_search with section_ids covering this whole section (drop any narrowing page_range), then ingest the new observation's auto-extracted ScanClaim.",
                    section_id=sid,
                    missing_pages=sorted(missing)[:10],
                )
    expected = _aggregate_page_hits_to_units(
        positive_pages=positive_pages,
        negative_pages=negative_pages,
        scanned_pages=scanned_pages,
        inventory=inventory,
        target_unit_type=unit_type,
    )
    if expected is None:
        return _err(
            "scanclaim_partition_unavailable",
            f"observation cannot ground a {unit_type}-level scan partition",
            remediation=f"Re-run pattern_search at the appropriate granularity for unit_type={unit_type!r} — for unit_type='file' the scan must cover the whole file; for 'section' the scan must cover the section's full page range.",
        )
    if pos_set != set(expected["positive_units"]) or neg_set != set(expected["negative_units"]):
        return _err(
            "scanclaim_partition_mismatch",
            "ScanClaim positive/negative units must match the observation's actual partition",
            remediation="Drop your manual ScanClaim — the auto-extracted ScanClaim from the same exhaustive pattern_search already binds; check the observation's auto_extract_claim_ids.",
            expected_positive=sorted(expected["positive_units"])[:10],
            expected_negative_count=len(expected["negative_units"]),
        )
    return None


def _validate_witness(
    *,
    predicate,
    score,
    unit_type: str,
    positive_units: List[str],
    value_map: Dict[str, Any],
    citations: List[Citation],
    observation: Observation,
    plant,
    _err: _ErrFn,
) -> Optional[Dict[str, Any]]:
    if predicate is None:
        return _err(
            "witness_predicate_required", "WitnessClaim must declare a predicate",
            remediation="Add a `predicate` field to claim_candidate matching the obligation's predicate (e.g. {'name':'contains_string','args':{'pattern':'X'}}); the plant evaluates it against the cited span.",
            valid_example={"name": "contains_string", "args": {"pattern": "Premium", "case_sensitive": False}},
        )
    cite_index = {(c.file_id, c.page_id): c for c in citations}
    for u in positive_units:
        citation = plant._citation_for_unit(u, unit_type, cite_index)
        if citation is None:
            return _err(
                "witness_unit_uncited",
                f"positive unit {u!r} has no citation that resolves to it",
                remediation=("Add a citation whose file_id equals the unit_id (for unit_type='file') OR "
                             "whose page_id falls inside the section's page range (for unit_type='section'). "
                             "Call toc to find the section's page range, then read_page that page."),
                unit_id=u,
            )
        unit_payload: Dict[str, Any] = {}
        gid = f"{citation.file_id}/{citation.page_id}"
        for key in ("results", "hits", "candidate_pages"):
            for entry in observation.payload.get(key) or []:
                from agentic.proof.plant import _entry_gid
                if _entry_gid(entry) == gid:
                    unit_payload.update(entry)
                    break
            if unit_payload:
                break
        from agentic.proof.plant import _entry_gid
        if _entry_gid(observation.payload) == gid:
            for k, v in observation.payload.items():
                if k not in {"text", "results", "hits", "candidate_pages"}:
                    unit_payload.setdefault(k, v)
        unit_payload["text"] = plant._fetch_span_text(citation, observation)
        if not pr.evaluate(predicate, unit_payload):
            return _err(
                "predicate_false_on_cited_content",
                f"predicate does not hold on cited content for unit {u!r}",
                remediation="Either pick a citation span that actually satisfies the predicate (e.g. read_page a page where the pattern truly appears), or weaken the predicate to one the cited span satisfies. Re-fetch via read_page first.",
                unit_id=u,
            )
    if score is not None and not value_map:
        return _err(
            "missing_value_map", "WitnessClaim with score must populate value_map",
            remediation="Add `value_map: {<unit_id>: <number>}` for every positive_unit you want argmax to consider (the plant verifies each value against the cited span via the score extractor).",
            valid_example={"value_map": {"<unit_id>": 1234}},
        )
    stray = sorted(set(value_map) - set(positive_units))
    if stray:
        return _err(
            "value_map_key_not_witness",
            f"value_map keys must be a subset of positive_units; stray: {stray}",
            remediation="Either drop the stray keys from value_map or add their unit_ids to positive_units (and supply a citation that anchors each to a page).",
            stray_keys=stray,
        )
    for unit_id, claimed_value in value_map.items():
        if score is None:
            return _err(
                "missing_score_ref", "value_map requires score",
                remediation="Add a `score` field to the claim (matching the obligation's score), e.g. {'name':'numeric_amount','args':{}}.",
                valid_example={"name": "numeric_amount", "args": {}},
            )
        citation = plant._citation_for_unit(unit_id, unit_type, cite_index)
        if citation is None:
            return _err(
                "citation_for_unit_missing", f"no citation maps to unit {unit_id!r}",
                remediation="Add a citation whose file_id equals the unit_id (for unit_type='file') or whose page_id is inside the section's page range (for unit_type='section').",
                unit_id=unit_id,
            )
        span_text = plant._fetch_span_text(citation, observation)
        try:
            extracted = sr.extract_value(score, span_text, observation.payload)
        except sr.ScoreExtractionError as exc:
            return _err(
                "score_extraction_failed", f"unit={unit_id!r} extraction failed: {exc}",
                remediation="Pick a citation span whose text actually contains a value the chosen score extractor can parse (e.g. numeric_amount needs a number like 'USD 1,000'); re-call read_page on a different page if needed.",
                unit_id=unit_id,
            )
        if not sr.values_match(extracted, claimed_value):
            return _err(
                "score_value_mismatch",
                f"unit={unit_id!r} extracted={extracted!r} claimed={claimed_value!r}",
                remediation=f"Update value_map[{unit_id!r}] to {extracted!r} (the value the extractor actually parses from your cited span); or pick a different citation whose span contains the value you intend to claim.",
                unit_id=unit_id,
                extracted=extracted,
                claimed=claimed_value,
            )
    return None
