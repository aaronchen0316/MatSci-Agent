from fastapi.testclient import TestClient

from matsci_agent.api.main import app
from matsci_agent.agents.planner import ChemistryIntentAgent
import matsci_agent.api.main as api_main
from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    DiscoveryPlan,
    DiscoveryRequest,
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
from matsci_agent.tools.mp_retriever import MPRetriever, MPRetrieverConfig
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


def test_discover_endpoint_returns_ranked_candidates():
    workflow = DiscoveryWorkflow(
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
    api_main.workflow = workflow
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
            super().__init__(MPRetrieverConfig(use_live_if_available=False))

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
                            uncertainty=1.0,
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


def test_discover_endpoint_exposes_mp_and_matgl_gap_fields():
    workflow = DiscoveryWorkflow(
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
    api_main.workflow = workflow

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
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
    monkeypatch.delenv("OPENROUTER_API_KEY_RAG", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("MATSCI_LLM_API_KEY_ENV", raising=False)
    wf = DiscoveryWorkflow(
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
    api_main.workflow = DiscoveryWorkflow()
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
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
    api_main.workflow = workflow
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
            super().__init__(MPRetrieverConfig(use_live_if_available=False))

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
    api_main.workflow = workflow

    res = client.post(
        "/discover",
        json={"research_goal": "Find lead-free perovskite materials with formation energy below -1 eV."},
    )

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["candidates"][0]["properties"]["formation_energy"] == -1.2


def test_discover_full_includes_capability_assessment_for_unsupported_request():
    api_main.workflow = DiscoveryWorkflow()
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
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
        policy_filter=PolicyFilter(inference_fn=lambda _payload: "not-json"),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    api_main.workflow = workflow
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


def test_workflow_replenishes_once_when_first_filter_batch_underfills():
    call_count = {"n": 0}

    def _filter(payload):
        call_count["n"] += 1
        if call_count["n"] == 1:
            keep_ids = {"mp-mock-003"}
        else:
            keep_ids = {"mp-mock-007", "mp-mock-008"}
        return {
            "policy_name": "practical_screening",
            "decisions": [
                {
                    "material_id": candidate["material_id"],
                    "keep": candidate["material_id"] in keep_ids,
                    "reasons": [] if candidate["material_id"] in keep_ids else ["not good fit"],
                }
                for candidate in payload["candidates"]
            ],
        }

    wf = DiscoveryWorkflow(
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
        intent_agent=ChemistryIntentAgent(parser_fn=lambda _goal: DiscoveryConstraints()),
        policy_filter=PolicyFilter(inference_fn=_filter),
        search_expander=FakeSearchExpander(),
        enable_policy_filter=True,
    )
    req = DiscoveryRequest(
        research_goal="Find semiconductor materials",
        constraints=DiscoveryConstraints(top_k=3),
    )

    out = wf.run(req)

    assert call_count["n"] == 2
    assert out.status == "success"
    assert len(out.candidates) <= 3


def test_workflow_relaxes_retrieval_limit_when_first_filter_batch_has_zero_passes():
    class WideRetriever(MPRetriever):
        def __init__(self):
            super().__init__(MPRetrieverConfig(use_live_if_available=False))
            self.limit_overrides: list[int | None] = []

        def retrieve(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
            self.limit_overrides.append(payload.limit_override)
            prefix = "raw" if payload.limit_override is None else "relaxed"
            candidates = [
                Candidate(
                    material_id=f"{prefix}-{index}",
                    formula="SiC" if prefix == "relaxed" and index == 0 else f"SiC{index}H",
                    source="mock",
                    features={
                        "elements": ["Si", "C"] if prefix == "relaxed" and index == 0 else ["Si", "C", "H"],
                        "mp_band_gap_ev": 2.5,
                        "mp_energy_above_hull": 0.01,
                        "nsites": 8,
                        "is_metal": False,
                    },
                )
                for index in range(20)
            ]
            return MPRetrieverOutput(
                candidates=candidates,
                provenance=ToolCallProvenance(
                    tool_name="wide_retriever",
                    input_payload=payload.model_dump(mode="json"),
                    output_summary={"candidate_count": len(candidates)},
                ),
            )

    call_count = {"n": 0}

    def _filter(payload):
        call_count["n"] += 1
        return {
            "policy_name": "chemistry_screening",
            "decisions": [
                {
                    "material_id": candidate["material_id"],
                    "keep": candidate["material_id"] == "relaxed-0",
                    "reasons": [] if candidate["material_id"] == "relaxed-0" else ["not requested material concept"],
                }
                for candidate in payload["candidates"]
            ],
        }

    retriever = WideRetriever()
    wf = DiscoveryWorkflow(
        retriever=retriever,
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(requested_material_class="semiconductor")
        ),
        policy_filter=PolicyFilter(inference_fn=_filter),
        search_expander=FakeSearchExpander(["SiC", "SiCH"]),
        enable_policy_filter=True,
    )
    req = DiscoveryRequest(
        research_goal="Find semiconductor materials with silicon and carbon and band gap above 2 eV",
        constraints=DiscoveryConstraints(required_elements=["Si", "C"], min_band_gap_ev=2.0, top_k=1),
    )

    out = wf.run(req)

    assert out.status == "success"
    assert call_count["n"] == 2
    assert retriever.limit_overrides == [None, 100]
    assert out.candidates[0].candidate.material_id == "relaxed-0"


def test_filter_candidate_selection_prioritizes_exact_required_element_set():
    wf = DiscoveryWorkflow(enable_policy_filter=False)
    constraints = DiscoveryConstraints(required_elements=["Si", "C"])
    candidates = [
        Candidate(
            material_id="complex",
            formula="SiH12C2N4O4",
            features={"elements": ["Si", "C", "H", "N", "O"], "mp_band_gap_ev": 7.0},
        ),
        Candidate(
            material_id="binary",
            formula="SiC",
            features={"elements": ["Si", "C"], "mp_band_gap_ev": 2.4},
        ),
    ]

    selected = wf._select_filter_candidates(candidates, limit=2, constraints=constraints)

    assert [candidate.material_id for candidate in selected] == ["binary", "complex"]


def test_workflow_fails_closed_when_filter_returns_invalid_json():
    wf = DiscoveryWorkflow(
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
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
