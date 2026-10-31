from fastapi.testclient import TestClient

from matsci_agent.api.main import app
from matsci_agent.agents.planner import ChemistryIntentAgent
import matsci_agent.api.main as api_main
from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    DiscoveryPlan,
    DiscoveryRequest,
    FloatRange,
    MPFilters,
    MPRetrieverInput,
    MPRetrieverOutput,
    ParsedDiscoveryIntent,
    PredictedProperties,
    PropertyPredictionRecord,
    PropertyPredictorOutput,
    SearchSpaceExpansionOutput,
    SearchSpaceTarget,
    ToolCallProvenance,
)
from matsci_agent.tools.mp_retriever import MPRetriever
from tests.support import FixtureRetriever
from matsci_agent.tools.policy_filter import PolicyFilter
from matsci_agent.workflow.graph import DiscoveryWorkflow


client = TestClient(app)


class FakeSearchExpander:
    def __init__(self, formulas: list[str] | None = None) -> None:
        self.formulas = formulas or ["AlN", "Fe2VAl", "CoTi", "O2", "AcF3", "SiC"]

    def expand(self, payload) -> SearchSpaceExpansionOutput:
        targets = [
            SearchSpaceTarget(
                formula=formula,
                normalized_formula=formula,
                chemsys="-".join(sorted(MPRetriever._extract_elements(formula))),
                elements=sorted(element.capitalize() for element in MPRetriever._extract_elements(formula)),
                confidence=0.9,
                rationale="test fixture",
            )
            for formula in self.formulas[: payload.target_count]
        ]
        return SearchSpaceExpansionOutput(
            targets=targets,
            provenance=ToolCallProvenance(
                tool_name="search_space_expander",
                input_payload={"test": True},
                output_summary={"target_count": len(targets)},
            ),
        )


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_api_requires_configured_token_and_returns_request_id(monkeypatch):
    monkeypatch.setenv("MATSCI_API_TOKEN", "test-secret")
    payload = {"research_goal": "Calculate lithium diffusivity with molecular dynamics"}
    assert client.post("/discover", json=payload).status_code == 401
    response = client.post("/discover", json=payload, headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_readiness_reports_missing_credentials(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert "MP_API_KEY" in response.json()["missing"]


def test_discover_endpoint_returns_ranked_candidates():
    workflow = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(requested_material_class="semiconductor")
        ),
        policy_filter=PolicyFilter(
            inference_fn=lambda payload: {
                "policy_name": "practical_screening",
                "decisions": [
                    {
                        "material_id": candidate["material_id"],
                        "keep": candidate["formula"] not in {"AcF3", "SiC"},
                        "reasons": (
                            ["radioactive fluoride not practical semiconductor"]
                            if candidate["formula"] == "AcF3"
                            else ["contains banned silicon element"]
                            if candidate["formula"] == "SiC"
                            else []
                        ),
                    }
                    for candidate in payload["candidates"]
                ],
            }
        ),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    api_main.workflow_factory = lambda: workflow
    payload = {
        "research_goal": "Find semiconductor materials without silicon and band gap above 2 eV",
        "constraints": {
            "banned_elements": ["Si"],
            "min_band_gap_ev": 2.0,
            "max_energy_above_hull": 0.08,
            "top_k": 3,
        },
    }
    res = client.post("/discover", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert len(body["candidates"]) <= 3
    if body["candidates"]:
        first = body["candidates"][0]
        assert set(first.keys()) == {
            "material_id",
            "formula",
            "band_gap_ev",
            "mp_band_gap_ev",
            "matgl_band_gap_ev",
            "band_gap_source",
            "matgl_error",
            "gap_difference_ev",
            "source",
            "energy_above_hull",
            "is_stable",
            "stability_source",
            "has_multiple_entries",
            "entry_count",
            "properties",
        }
        formulas = {candidate["formula"] for candidate in body["candidates"]}
        assert "AcF3" not in formulas


def test_workflow_recalculates_matgl_only_for_finalized_shortlist_and_keeps_membership():
    class FourCandidateRetriever(MPRetriever):
        def __init__(self):
            super().__init__()

        def retrieve(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
            candidates = [
                Candidate(
                    material_id="mp-1",
                    formula="A1B1O3",
                    source="mock",
                    features={"mp_band_gap_ev": 5.0, "mp_energy_above_hull": 0.01, "nsites": 5},
                ),
                Candidate(
                    material_id="mp-2",
                    formula="A2B2O3",
                    source="mock",
                    features={"mp_band_gap_ev": 4.0, "mp_energy_above_hull": 0.02, "nsites": 5},
                ),
                Candidate(
                    material_id="mp-3",
                    formula="A3B3O3",
                    source="mock",
                    features={"mp_band_gap_ev": 3.0, "mp_energy_above_hull": 0.03, "nsites": 5},
                ),
                Candidate(
                    material_id="mp-4",
                    formula="A4B4O3",
                    source="mock",
                    features={"mp_band_gap_ev": 2.5, "mp_energy_above_hull": 0.04, "nsites": 5},
                ),
            ]
            return MPRetrieverOutput(
                candidates=candidates,
                provenance=ToolCallProvenance(
                    tool_name="four_candidate_retriever",
                    input_payload=payload.model_dump(mode="json"),
                    output_summary={"candidate_count": len(candidates)},
                ),
            )

    class RecordingPredictor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run(self, payload) -> PropertyPredictorOutput:
            ids = [candidate.material_id for candidate in payload.candidates]
            self.calls.append(
                {
                    "calculate_matgl": payload.calculate_matgl,
                    "candidate_ids": ids,
                }
            )
            predictions = []
            matgl_by_id = {"mp-1": 2.4, "mp-2": 1.1}
            for candidate in payload.candidates:
                cloned = candidate.model_copy(deep=True)
                gap = matgl_by_id[candidate.material_id]
                cloned.features["matgl_band_gap_ev"] = gap
                cloned.features["band_gap_source"] = "matgl"
                predictions.append(
                    PropertyPredictionRecord(
                        candidate=cloned,
                        predicted=PredictedProperties(
                            band_gap_ev=gap,
                            backend="matgl_band_gap:test",
                        ),
                    )
                )
            return PropertyPredictorOutput(
                predictions=predictions,
                provenance=ToolCallProvenance(
                    tool_name="property_predictor",
                    input_payload={"candidate_ids": ids},
                    output_summary={
                        "backend": "hybrid_mp_then_matgl",
                        "prediction_count": len(predictions),
                        "used_matgl_count": len(predictions),
                    },
                ),
            )

    predictor = RecordingPredictor()
    wf = DiscoveryWorkflow(
        retriever=FourCandidateRetriever(),
        predictor=predictor,
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(
                requested_material_class="perovskite",
                constraints=DiscoveryConstraints(calculate_matgl=True, top_k=2, min_band_gap_ev=2.0),
            )
        ),
        policy_filter=PolicyFilter(
            inference_fn=lambda payload: {
                "policy_name": "chemistry_screening",
                "decisions": [
                    {"material_id": candidate["material_id"], "keep": True, "reasons": []}
                    for candidate in payload["candidates"]
                ],
            }
        ),
        search_expander=FakeSearchExpander(["A1B1O3", "A2B2O3", "A3B3O3", "A4B4O3"]),
        enable_policy_filter=True,
    )
    req = DiscoveryRequest(
        research_goal="Find perovskites with band gap above 2 eV",
        constraints=DiscoveryConstraints(calculate_matgl=True, top_k=2, min_band_gap_ev=2.0),
    )

    out = wf.run(req)

    assert predictor.calls == [
        {"calculate_matgl": True, "candidate_ids": ["mp-1", "mp-2"]},
    ]
    assert [candidate.candidate.material_id for candidate in out.candidates] == ["mp-1", "mp-2"]
    assert out.candidates[0].predicted_properties.band_gap_ev == 2.4
    assert out.candidates[1].predicted_properties.band_gap_ev == 1.1


def test_discover_endpoint_exposes_mp_and_matgl_gap_fields(monkeypatch):
    monkeypatch.setenv("MATSCI_ENABLE_MATGL", "1")
    monkeypatch.setattr(api_main, "_matgl_missing", lambda: [])
    workflow = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(
                requested_material_class="semiconductor",
                constraints=DiscoveryConstraints(calculate_matgl=True, top_k=1, min_band_gap_ev=2.0),
            )
        ),
        policy_filter=PolicyFilter(
            inference_fn=lambda payload: {
                "policy_name": "practical_screening",
                "decisions": [
                    {
                        "material_id": candidate["material_id"],
                        "keep": candidate["material_id"] == "mp-mock-003",
                        "reasons": [],
                    }
                    for candidate in payload["candidates"]
                ],
            }
        ),
        search_expander=FakeSearchExpander(["AlN"]),
        enable_policy_filter=True,
    )
    api_main.workflow_factory = lambda: workflow

    res = client.post(
        "/discover",
        json={
            "research_goal": "Find semiconductor materials without silicon and band gap above 2 eV",
            "constraints": {"calculate_matgl": True, "top_k": 1, "min_band_gap_ev": 2.0},
        },
    )

    assert res.status_code == 200
    first = res.json()["candidates"][0]
    assert "mp_band_gap_ev" in first
    assert "matgl_band_gap_ev" in first


def test_workflow_can_still_skip_policy_filter_when_explicitly_disabled():
    wf = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(requested_material_class="semiconductor")
        ),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=False,
    )
    req = DiscoveryRequest(
        research_goal="Find high band gap materials",
        constraints=DiscoveryConstraints(max_energy_above_hull=0.0, top_k=2),
    )
    out = wf.run(req)
    assert out.status == "success"
    assert out.iterations == 1
    assert len(out.candidates) <= 2


def test_workflow_defaults_to_policy_filter_fails_closed_without_llm_credentials(monkeypatch):
    monkeypatch.delenv("MATSCI_LLM_API_KEY", raising=False)
    wf = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
    )
    req = DiscoveryRequest(
        research_goal="Find semiconductor materials without silicon and band gap above 2 eV",
        constraints=DiscoveryConstraints(
            banned_elements=["Si"],
            min_band_gap_ev=2.0,
            max_energy_above_hull=0.08,
            top_k=5,
        ),
    )

    out = wf.run(req)

    assert out.status == "failed"
    assert any("Search-space expansion failed" in message for message in out.messages)


def test_discover_endpoint_refuses_unsupported_diffusivity_request():
    api_main.workflow_factory = DiscoveryWorkflow
    payload = {
        "research_goal": "Estimate diffusivity in bulk materials with long molecular dynamics runs",
    }
    res = client.post("/discover", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "unsupported"
    assert body["candidates"] == []
    assert "unsupported" in body["unsupported_reason"].lower()


def test_discover_full_endpoint_returns_debug_trace():
    workflow = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(requested_material_class="semiconductor")
        ),
        policy_filter=PolicyFilter(
            inference_fn=lambda payload: {
                "policy_name": "practical_screening",
                "decisions": [
                    {
                        "material_id": candidate["material_id"],
                        "keep": candidate["material_id"] in {"mp-mock-003", "mp-mock-005"},
                        "reasons": [] if candidate["material_id"] in {"mp-mock-003", "mp-mock-005"} else ["not selected"],
                    }
                    for candidate in payload["candidates"]
                ],
            }
        ),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    api_main.workflow_factory = lambda: workflow
    payload = {
        "research_goal": "Find semiconductor materials",
        "constraints": {"top_k": 5},
    }

    res = client.post("/discover/full", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert "raw_candidates" in body
    assert "filtered_candidates" in body
    assert "filter_records" in body
    assert "search_space_targets" in body
    assert "candidates" in body
    assert "messages" in body
    assert "provenance" in body
    assert "discovery_plan" in body
    assert "capability_assessment" in body
    assert "report_summary" in body
    assert body["discovery_plan"]["source_universe"] == "materials_project_entries"
    assert body["discovery_plan"]["requested_material_class"] == "semiconductor"
    assert "material_class" not in body["discovery_plan"]
    assert body["raw_candidates"]
    assert body["filtered_candidates"]
    assert body["filter_records"]


def test_discover_endpoint_returns_generic_mp_properties():
    class GenericRetriever(MPRetriever):
        def __init__(self):
            super().__init__()

        def retrieve(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
            candidates = [
                Candidate(
                    material_id="mp-cs-sni3",
                    formula="CsSnI3",
                    source="mock",
                    features={
                        "elements": ["Cs", "Sn", "I"],
                        "formation_energy": -1.2,
                        "mp_energy_above_hull": 0.03,
                        "mp_band_gap_ev": 1.3,
                        "nsites": 5,
                    },
                )
            ]
            return MPRetrieverOutput(
                candidates=candidates,
                provenance=ToolCallProvenance(
                    tool_name="generic_retriever",
                    input_payload=payload.model_dump(mode="json"),
                    output_summary={"candidate_count": len(candidates)},
                ),
            )

    workflow = DiscoveryWorkflow(
        retriever=GenericRetriever(),
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(
                requested_material_class="perovskite",
                constraints=DiscoveryConstraints(),
            )
        ),
        policy_filter=PolicyFilter(
            inference_fn=lambda payload: {
                "policy_name": "chemistry_screening",
                "decisions": [
                    {"material_id": candidate["material_id"], "keep": True, "reasons": []}
                    for candidate in payload["candidates"]
                ],
            }
        ),
        search_expander=FakeSearchExpander(["CsSnI3"]),
        enable_policy_filter=True,
    )
    api_main.workflow_factory = lambda: workflow

    res = client.post(
        "/discover",
        json={"research_goal": "Find lead-free perovskite materials with formation energy below -1 eV."},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["candidates"][0]["properties"]["formation_energy"] == -1.2


def test_discover_full_includes_capability_assessment_for_unsupported_request():
    api_main.workflow_factory = DiscoveryWorkflow
    payload = {
        "research_goal": "Estimate diffusivity in bulk materials with long molecular dynamics runs",
    }

    res = client.post("/discover/full", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "unsupported"
    assert body["capability_assessment"] is not None
    assert body["raw_candidates"] == []
    assert body["filtered_candidates"] == []
    assert body["filter_records"] == []


def test_discover_full_shows_filter_failure_trace():
    workflow = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
        policy_filter=PolicyFilter(inference_fn=lambda _payload: "not-json"),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    api_main.workflow_factory = lambda: workflow
    payload = {
        "research_goal": "Find semiconductor materials",
        "constraints": {"top_k": 3},
    }

    res = client.post("/discover/full", json=payload)

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "failed"
    assert body["filtered_candidates"] == []
    assert body["filter_records"] == []
    assert any("Chemistry filter failed" in message for message in body["messages"])
    assert any(
        provenance["tool_name"] == "policy_filter"
        and provenance["output_summary"].get("failure_code") == "policy_filter_invalid_json"
        for provenance in body["provenance"]
    )





def test_workflow_fails_closed_when_filter_returns_invalid_json():
    wf = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
        policy_filter=PolicyFilter(inference_fn=lambda _payload: "not-json"),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    req = DiscoveryRequest(
        research_goal="Find semiconductor materials",
        constraints=DiscoveryConstraints(top_k=3),
    )

    out = wf.run(req)

    assert out.status == "failed"
    assert out.candidates == []
    assert any("Chemistry filter failed" in message for message in out.messages)
    assert any(
        provenance.output_summary.get("failure_code") == "policy_filter_invalid_json"
        for provenance in out.provenance
        if provenance.tool_name == "policy_filter"
    )


def test_workflow_unknown_stability_does_not_trigger_refine():
    wf = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
        policy_filter=PolicyFilter(
            inference_fn=lambda payload: {
                "policy_name": "exploratory_screening",
                "decisions": [
                    {
                        "material_id": candidate["material_id"],
                        "keep": candidate["material_id"] in {"mp-mock-005", "mp-mock-006"},
                        "reasons": [] if candidate["material_id"] in {"mp-mock-005", "mp-mock-006"} else ["not selected"],
                    }
                    for candidate in payload["candidates"]
                ],
            }
        ),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    req = DiscoveryRequest(
        research_goal="Find semiconductor materials",
        constraints=DiscoveryConstraints(top_k=5),
    )

    out = wf.run(req)

    assert out.iterations == 1
    assert out.status == "success"
    assert out.candidates
    assert any(
        "Stability is unknown" in message
        for message in out.messages
    )


def test_workflow_returns_stability_annotations_in_ranked_candidates():
    wf = DiscoveryWorkflow(
        retriever=FixtureRetriever(),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
        policy_filter=PolicyFilter(
            inference_fn=lambda payload: {
                "policy_name": "exploratory_screening",
                "decisions": [
                    {
                        "material_id": candidate["material_id"],
                        "keep": candidate["material_id"] in {"mp-mock-003", "mp-mock-005"},
                        "reasons": [] if candidate["material_id"] in {"mp-mock-003", "mp-mock-005"} else ["not selected"],
                    }
                    for candidate in payload["candidates"]
                ],
            }
        ),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    req = DiscoveryRequest(
        research_goal="Find semiconductor materials",
        constraints=DiscoveryConstraints(top_k=5),
    )

    out = wf.run(req)

    assert len(out.candidates) >= 2
    by_id = {candidate.candidate.material_id: candidate for candidate in out.candidates}
    assert by_id["mp-mock-003"].stability.is_stable is True
    assert by_id["mp-mock-003"].stability.source == "materials_project"
    assert by_id["mp-mock-005"].stability.is_stable is None
    assert by_id["mp-mock-005"].stability.source == "unknown"


def _structure_workflow(candidates):
    class Retriever(MPRetriever):
        def retrieve(self, payload):
            return MPRetrieverOutput(
                candidates=[candidate.model_copy(deep=True) for candidate in candidates],
                provenance=ToolCallProvenance(
                    tool_name="test_retriever", input_payload={},
                    output_summary={"source": "materials_project"},
                ),
                reason_code="no_matches" if not candidates else None,
            )

    return DiscoveryWorkflow(
        retriever=Retriever(),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
        search_expander=FakeSearchExpander(["AlN", "GaN"]),
        enable_policy_filter=False,
    )


def test_top_k_counts_formulas_and_keeps_all_passing_structures():
    candidates = [
        Candidate(material_id="mp-1", formula="AlN", source="materials_project", features={"mp_band_gap_ev": 4.0, "mp_energy_above_hull": 0.01}),
        Candidate(material_id="mp-2", formula="AlN", source="materials_project", features={"mp_band_gap_ev": 3.5, "mp_energy_above_hull": 0.02}),
        Candidate(material_id="mp-3", formula="GaN", source="materials_project", features={"mp_band_gap_ev": 2.0, "mp_energy_above_hull": 0.01}),
    ]
    result = _structure_workflow(candidates).run(DiscoveryRequest(
        research_goal="Find semiconductors with band gap above 1 eV",
        constraints=DiscoveryConstraints(top_k=1, min_band_gap_ev=1.0),
    ))
    assert [item.candidate.material_id for item in result.candidates] == ["mp-1", "mp-2"]
    assert all(item.candidate.entry_count == 2 for item in result.candidates)


def test_compact_api_exposes_each_structure_and_data_source(monkeypatch):
    candidates = [
        Candidate(material_id="mp-1", formula="AlN", source="materials_project", features={"mp_band_gap_ev": 4.0, "mp_energy_above_hull": 0.01}),
        Candidate(material_id="mp-2", formula="AlN", source="materials_project", features={"mp_band_gap_ev": 3.5, "mp_energy_above_hull": 0.02}),
    ]
    monkeypatch.setattr(api_main, "workflow_factory", lambda: _structure_workflow(candidates))
    response = client.post("/discover", json={"research_goal": "Find semiconductor materials", "constraints": {"top_k": 1}})
    assert response.status_code == 200
    body = response.json()
    assert body["data_source"] == "materials_project"
    assert [item["material_id"] for item in body["candidates"]] == ["mp-1", "mp-2"]


def test_formula_ranking_uses_requested_mp_property():
    candidates = [
        Candidate(material_id="mp-1", formula="Al2O3", source="materials_project", features={"density": 5.2, "mp_energy_above_hull": 0.01}),
        Candidate(material_id="mp-2", formula="Ga2O3", source="materials_project", features={"density": 6.4, "mp_energy_above_hull": 0.01}),
    ]
    result = _structure_workflow(candidates).run(DiscoveryRequest(
        research_goal="Find dense oxide materials",
        constraints=DiscoveryConstraints(top_k=1, mp_filters=MPFilters(density=FloatRange(min=5.0))),
    ))
    assert [item.candidate.formula for item in result.candidates] == ["Ga2O3"]


def test_empty_mp_search_is_success_with_reason():
    result = _structure_workflow([]).run(DiscoveryRequest(
        research_goal="Find semiconductor materials", constraints=DiscoveryConstraints(top_k=1),
    ))
    assert result.status == "success"
    assert result.reason_code == "no_matches"
    assert result.candidates == []


def test_final_constraint_rejection_has_distinct_reason():
    candidate = Candidate(material_id="mp-1", formula="AlN", source="materials_project", features={"mp_band_gap_ev": None})
    result = _structure_workflow([candidate]).run(DiscoveryRequest(
        research_goal="Find semiconductors with band gap above 2 eV",
        constraints=DiscoveryConstraints(top_k=1, min_band_gap_ev=2.0),
    ))
    assert result.status == "success"
    assert result.reason_code == "final_constraint_no_pass"
    assert result.candidates == []


def test_matgl_partial_failure_preserves_structures_and_mp_data(monkeypatch):
    from matsci_agent.tools import property_predictor

    candidates = [
        Candidate(material_id="mp-1", formula="AlN", source="materials_project", features={"mp_band_gap_ev": 2.5, "mp_energy_above_hull": 0.01, "nsites": 8}),
        Candidate(material_id="mp-2", formula="AlN", source="materials_project", features={"mp_band_gap_ev": 2.4, "mp_energy_above_hull": 0.02, "nsites": 8}),
    ]
    values = iter([(1.5, None), (None, "model failed")])
    monkeypatch.setattr(property_predictor, "_calculate_matgl", lambda *_args, **_kwargs: next(values))
    result = _structure_workflow(candidates).run(DiscoveryRequest(
        research_goal="Find semiconductors with MatGL band gap calculation",
        constraints=DiscoveryConstraints(top_k=1, calculate_matgl=True, min_band_gap_ev=2.0),
    ))
    assert result.status == "partial"
    assert result.reason_code == "matgl_calculation_failed"
    assert [item.candidate.material_id for item in result.candidates] == ["mp-1", "mp-2"]
    assert result.candidates[0].candidate.features["gap_difference_ev"] == -1.0
    assert result.candidates[1].predicted_properties.band_gap_ev is None
    assert result.candidates[1].candidate.features["mp_band_gap_ev"] == 2.4
