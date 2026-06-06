from matsci_agent.schemas import DiscoveryConstraints, FloatRange, IntRange, MPRetrieverInput, MPFilters, SearchSpaceTarget
from matsci_agent.tools.mp_retriever import MPRetriever, MPRetrieverConfig


def test_mock_fallback_provenance_when_live_disabled():
    retriever = MPRetriever(MPRetrieverConfig(use_live_if_available=False))
    payload = MPRetrieverInput(
        research_goal="find alloys",
        constraints=DiscoveryConstraints(top_k=2),
    )

    out = retriever.retrieve(payload)

    assert out.provenance.output_summary["source"] == "mock_fallback"
    assert out.provenance.output_summary["fallback_used"] is True
    assert len(out.candidates) <= 8


def test_mock_filter_banned_and_required_elements():
    retriever = MPRetriever(MPRetrieverConfig(use_live_if_available=False))
    payload = MPRetrieverInput(
        research_goal="find Al materials without Co",
        constraints=DiscoveryConstraints(
            banned_elements=["Co"],
            required_elements=["Al"],
            top_k=10,
        ),
    )

    out = retriever.retrieve(payload)

    assert out.candidates
    for candidate in out.candidates:
        elements = retriever._extract_elements(candidate.formula)
        assert "co" not in elements
        assert "al" in elements


def test_formula_element_parser_distinguishes_c_and_co():
    assert "co" in MPRetriever._extract_elements("CoTi")
    assert "co" not in MPRetriever._extract_elements("SiC")


def test_mock_pool_still_enforces_explicit_numeric_and_banned_element_constraints():
    retriever = MPRetriever(MPRetrieverConfig(use_live_if_available=False))
    payload = MPRetrieverInput(
        research_goal="Find semiconductor materials with no silicon and band gap higher than 1 eV",
        constraints=DiscoveryConstraints(banned_elements=["Si"], min_band_gap_ev=1.0, top_k=10),
    )
    out = retriever.retrieve(payload)
    formulas = {c.formula for c in out.candidates}
    assert "SiC" not in formulas
    for candidate in out.candidates:
        elements = retriever._extract_elements(candidate.formula)
        assert "si" not in elements
        band_gap = candidate.features.get("mp_band_gap_ev")
        if band_gap is not None:
            assert float(band_gap) >= 1.0


def test_mock_candidates_include_energy_above_hull_feature():
    retriever = MPRetriever(MPRetrieverConfig(use_live_if_available=False))
    payload = MPRetrieverInput(
        research_goal="Find semiconductors",
        constraints=DiscoveryConstraints(top_k=3),
    )
    out = retriever.retrieve(payload)
    assert out.candidates
    assert "mp_energy_above_hull" in out.candidates[0].features


def test_mock_retriever_honors_excluded_material_ids_and_limit_override():
    retriever = MPRetriever(MPRetrieverConfig(use_live_if_available=False))
    payload = MPRetrieverInput(
        research_goal="Find semiconductors",
        constraints=DiscoveryConstraints(top_k=5),
        exclude_material_ids=["mp-mock-003", "mp-mock-004"],
        limit_override=2,
    )
    out = retriever.retrieve(payload)
    ids = [candidate.material_id for candidate in out.candidates]
    assert len(ids) == 2
    assert "mp-mock-003" not in ids
    assert "mp-mock-004" not in ids


def test_mock_retriever_stays_within_search_space_targets():
    retriever = MPRetriever(MPRetrieverConfig(use_live_if_available=False))
    payload = MPRetrieverInput(
        research_goal="Find bounded target formulas",
        constraints=DiscoveryConstraints(top_k=5),
        search_space_targets=[
            SearchSpaceTarget(
                formula="SiC",
                normalized_formula="SiC",
                chemsys="C-Si",
                elements=["C", "Si"],
                confidence=0.9,
            )
        ],
    )

    out = retriever.retrieve(payload)

    assert {candidate.formula for candidate in out.candidates} == {"SiC"}


def test_client_side_filters_enforce_required_elements_subset():
    effective = MPFilters(
        elements=["Si", "C"],
        band_gap=FloatRange(min=2.0),
        num_elements=IntRange(min=2, max=20),
        is_metal=False,
    )

    assert MPRetriever._passes_client_side_filters(
        effective=effective,
        elements={"si", "c"},
        mp_band_gap_ev=2.4,
        mp_energy_above_hull=0.02,
        nsites=8,
        is_metal=False,
        theoretical=False,
        deprecated=False,
    ) is True

    assert MPRetriever._passes_client_side_filters(
        effective=effective,
        elements={"ac", "f"},
        mp_band_gap_ev=7.4,
        mp_energy_above_hull=0.0,
        nsites=16,
        is_metal=False,
        theoretical=False,
        deprecated=False,
    ) is False
