from fastapi.testclient import TestClient

from matsci_agent.api.main import app
from matsci_agent.agents.planner import ChemistryIntentAgent
import matsci_agent.api.main as api_main
from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    DiscoveryRequest,
    MPRetrieverInput,
    MPRetrieverOutput,
    ParsedDiscoveryIntent,
    ToolCallProvenance,
)
from matsci_agent.tools.mp_retriever import MPRetriever, MPRetrieverConfig
from matsci_agent.tools.policy_filter import PolicyFilter
from matsci_agent.workflow.graph import DiscoveryWorkflow


client = TestClient(app)


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
            "band_gap_source",
            "energy_above_hull",
            "is_stable",
            "stability_source",
            "has_multiple_entries",
            "entry_count",
        }
        formulas = {candidate["formula"] for candidate in body["candidates"]}
        assert "AcF3" not in formulas


def test_workflow_can_still_skip_policy_filter_when_explicitly_disabled():
    wf = DiscoveryWorkflow(
        retriever=MPRetriever(MPRetrieverConfig(use_live_if_available=False)),
        intent_agent=ChemistryIntentAgent(
            parser_fn=lambda _goal: ParsedDiscoveryIntent(requested_material_class="semiconductor")
        ),
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
    assert any("Chemistry filter failed" in message for message in out.messages)


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
