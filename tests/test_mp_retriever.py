from matsci_agent.schemas import DiscoveryConstraints, FloatRange, IntRange, MPRetrieverInput, MPFilters, SearchSpaceTarget
import sys
import types

import pytest

from matsci_agent.tools.mp_retriever import MPRetriever, MPRetrieverError
from tests.support import FixtureRetriever


def _live_payload(min_gap=None):
    return MPRetrieverInput(
        research_goal="Find AlN semiconductor materials",
        constraints=DiscoveryConstraints(top_k=1, min_band_gap_ev=min_gap),
        search_space_targets=[SearchSpaceTarget(
            formula="AlN", normalized_formula="AlN", chemsys="Al-N", elements=["Al", "N"], confidence=1.0,
        )],
    )


def _fake_mp(monkeypatch, docs=None, error=None):
    monkeypatch.setenv("MP_API_KEY", "test-key")

    class Rest:
        def __init__(self, _key):
            self.materials = types.SimpleNamespace(summary=types.SimpleNamespace(search=self.search))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def search(self, **_kwargs):
            if error:
                raise error
            return docs or []

    module = types.ModuleType("mp_api.client")
    module.MPRester = Rest
    monkeypatch.setitem(sys.modules, "mp_api.client", module)


def test_live_retrieval_keeps_distinct_structures(monkeypatch):
    docs = [
        types.SimpleNamespace(material_id=f"mp-{index}", formula_pretty="AlN", elements=["Al", "N"], band_gap=gap, energy_above_hull=0.01, nsites=8, structure=None, symmetry=None)
        for index, gap in [(1, 4.0), (2, 3.5)]
    ]
    _fake_mp(monkeypatch, docs=docs)
    result = MPRetriever().retrieve(_live_payload(min_gap=2.0))
    assert [candidate.material_id for candidate in result.candidates] == ["mp-1", "mp-2"]
    assert result.provenance.output_summary["source"] == "materials_project"


def test_band_gap_has_props_alias_uses_summary_filter(monkeypatch):
    _fake_mp(monkeypatch)
    payload = _live_payload()
    payload.constraints.mp_filters.has_props = ["band_gap", "dielectric"]
    kwargs = MPRetriever()._build_search_kwargs(payload, limit=4)
    assert kwargs["has_props"] == ["dielectric"]
    assert kwargs["band_gap"] == (0.0, 20.0)


def test_live_retrieval_caps_pages_and_total_documents(monkeypatch):
    calls = []
    monkeypatch.setenv("MP_API_KEY", "test-key")

    class Rest:
        def __init__(self, _key):
            self.materials = types.SimpleNamespace(summary=types.SimpleNamespace(search=self.search))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def search(self, **kwargs):
            calls.append(kwargs)
            return [
                types.SimpleNamespace(
                    material_id=f"mp-{index}", formula_pretty="AlN", elements=["Al", "N"],
                    band_gap=3.0, energy_above_hull=0.01, nsites=8, structure=None, symmetry=None,
                )
                for index in range(130)
            ]

    module = types.ModuleType("mp_api.client")
    module.MPRester = Rest
    monkeypatch.setitem(sys.modules, "mp_api.client", module)
    payload = _live_payload()
    payload.constraints.top_k = 30
    result = MPRetriever().retrieve(payload)
    assert len(result.candidates) == 120
    assert result.provenance.output_summary["raw_document_count"] == 120
    assert len(calls) == 1
    assert calls[0]["num_chunks"] == 1
    assert calls[0]["chunk_size"] == 120


def test_live_retrieval_distinguishes_no_matches_and_constraint_rejection(monkeypatch):
    _fake_mp(monkeypatch, docs=[])
    assert MPRetriever().retrieve(_live_payload()).reason_code == "no_matches"
    docs = [types.SimpleNamespace(material_id="mp-1", formula_pretty="AlN", elements=["Al", "N"], band_gap=None, energy_above_hull=0.01, nsites=8, structure=None, symmetry=None)]
    _fake_mp(monkeypatch, docs=docs)
    assert MPRetriever().retrieve(_live_payload(min_gap=2.0)).reason_code == "constraint_no_pass"


def test_explicit_boolean_filter_rejects_missing_value(monkeypatch):
    doc = types.SimpleNamespace(
        material_id="mp-1", formula_pretty="AlN", elements=["Al", "N"],
        band_gap=3.0, energy_above_hull=0.01, nsites=8,
        structure=None, symmetry=None, has_reconstructed=None,
    )
    _fake_mp(monkeypatch, docs=[doc])
    payload = _live_payload()
    payload.constraints.mp_filters.has_reconstructed = False
    assert MPRetriever().retrieve(payload).reason_code == "constraint_no_pass"


def test_live_retrieval_reports_credentials_and_service_failure(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    with pytest.raises(MPRetrieverError, match="Missing MP_API_KEY") as missing:
        MPRetriever().retrieve(_live_payload())
    assert missing.value.code == "mp_credentials_missing"
    _fake_mp(monkeypatch, error=RuntimeError("server unavailable"))
    with pytest.raises(MPRetrieverError) as failed:
        MPRetriever().retrieve(_live_payload())
    assert failed.value.code == "mp_service_failed"


def test_explicit_test_fixture_provenance_when_live_disabled():
    retriever = FixtureRetriever()
    payload = MPRetrieverInput(
        research_goal="find alloys",
        constraints=DiscoveryConstraints(top_k=2),
    )

    out = retriever.retrieve(payload)

    assert out.provenance.output_summary["source"] == "test_fixture"
    assert out.provenance.output_summary["fallback_used"] is False
    assert len(out.candidates) <= 8


def test_mock_filter_banned_and_required_elements():
    retriever = FixtureRetriever()
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
    retriever = FixtureRetriever()
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
    retriever = FixtureRetriever()
    payload = MPRetrieverInput(
        research_goal="Find semiconductors",
        constraints=DiscoveryConstraints(top_k=3),
    )
    out = retriever.retrieve(payload)
    assert out.candidates
    assert "mp_energy_above_hull" in out.candidates[0].features


def test_effective_filters_only_apply_hull_threshold_when_explicit():
    retriever = FixtureRetriever()

    omitted = retriever._effective_filters(DiscoveryConstraints(), "Find materials")
    explicit = retriever._effective_filters(
        DiscoveryConstraints(max_energy_above_hull=0.05),
        "Find materials with energy above hull below 0.05 eV",
    )

    assert omitted.energy_above_hull is None
    assert explicit.energy_above_hull is not None
    assert explicit.energy_above_hull.max == 0.05


def test_mock_retriever_honors_excluded_material_ids_and_limit_override():
    retriever = FixtureRetriever()
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
    retriever = FixtureRetriever()
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
