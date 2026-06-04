from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from matsci_agent.agents.planner import ChemistryIntentAgent
from matsci_agent.api.main import app
from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    MPRetrieverInput,
    MPRetrieverOutput,
    ToolCallProvenance,
)
from matsci_agent.workflow.graph import DiscoveryWorkflow
import matsci_agent.api.main as api_main
from matsci_agent.tools.mp_retriever import MPRetriever


client = TestClient(app)


@dataclass(frozen=True)
class PromptCase:
    name: str
    goal: str
    parser_constraints: DiscoveryConstraints = field(default_factory=DiscoveryConstraints)
    expected_status: str = "success"
    expected_task_class: str = "band_gap_screening"
    expected_reason_code: str | None = None
    expected_application_intent: str | None = None
    expected_practicality_mode: str | None = None
    expected_required_elements: tuple[str, ...] = ()
    expected_banned_elements: tuple[str, ...] = ()
    expected_calculate_matgl: bool | None = None
    expected_recalculate_top_n: int | None = None
    expected_top_k: int | None = None
    require_candidates: bool = True
    require_known_and_unknown_stability: bool = False
    require_carrier_element: str | None = None
    compact_reason_substring: str | None = None


class PromptQARetriever:
    """Test-only retriever with a broader offline candidate pool for prompt QA."""

    def __init__(self) -> None:
        self.request_limit_multiplier = 2
        self.pool = [
            Candidate(
                material_id="qa-001",
                formula="AlN",
                source="mock",
                features={"mp_band_gap_ev": 5.9, "mp_energy_above_hull": 0.01, "nsites": 8},
            ),
            Candidate(
                material_id="qa-002",
                formula="GaN",
                source="mock",
                features={"mp_band_gap_ev": 3.4, "mp_energy_above_hull": 0.015, "nsites": 8},
            ),
            Candidate(
                material_id="qa-003",
                formula="Al2O3",
                source="mock",
                features={"mp_band_gap_ev": 8.7, "mp_energy_above_hull": 0.005, "nsites": 10},
            ),
            Candidate(
                material_id="qa-004",
                formula="ZnO",
                source="mock",
                features={"mp_band_gap_ev": 3.3, "mp_energy_above_hull": 0.02, "nsites": 4},
            ),
            Candidate(
                material_id="qa-005",
                formula="SiC",
                source="mock",
                features={"mp_band_gap_ev": 2.4, "mp_energy_above_hull": 0.02, "nsites": 8},
            ),
            Candidate(
                material_id="qa-006",
                formula="Fe2VAl",
                source="mock",
                features={"mp_band_gap_ev": None, "mp_energy_above_hull": None, "nsites": 12},
            ),
            Candidate(
                material_id="qa-007",
                formula="AcF3",
                source="mock",
                features={"mp_band_gap_ev": 6.2, "mp_energy_above_hull": 0.02, "nsites": 16},
            ),
            Candidate(
                material_id="qa-008",
                formula="C3N4",
                source="mock",
                features={"mp_band_gap_ev": 2.8, "mp_energy_above_hull": None, "nsites": 14},
            ),
            Candidate(
                material_id="qa-009",
                formula="CoTi",
                source="mock",
                features={"mp_band_gap_ev": 1.0, "mp_energy_above_hull": 0.03, "nsites": 4},
            ),
            Candidate(
                material_id="qa-010",
                formula="PbS",
                source="mock",
                features={"mp_band_gap_ev": 0.4, "mp_energy_above_hull": 0.01, "nsites": 8},
            ),
        ]

    def retrieve(self, payload: MPRetrieverInput) -> MPRetrieverOutput:
        banned = {el.lower() for el in payload.constraints.banned_elements}
        required = {el.lower() for el in payload.constraints.required_elements}
        excluded_ids = set(payload.exclude_material_ids)
        limit = payload.limit_override or (payload.constraints.top_k * self.request_limit_multiplier)

        filtered = [
            candidate.model_copy(deep=True)
            for candidate in self.pool
            if candidate.material_id not in excluded_ids
            if not (MPRetriever._extract_elements(candidate.formula) & banned)
            if required.issubset(MPRetriever._extract_elements(candidate.formula))
            if MPRetriever._passes_goal_semantics(
                goal=payload.research_goal,
                element_set=MPRetriever._extract_elements(candidate.formula),
                mp_band_gap_ev=candidate.features.get("mp_band_gap_ev"),
                min_band_gap_ev=payload.constraints.min_band_gap_ev,
            )
        ]
        filtered.sort(
            key=lambda candidate: (
                -(float(candidate.features.get("mp_band_gap_ev") or -1.0)),
                candidate.material_id,
            )
        )
        filtered = filtered[:limit]
        provenance = ToolCallProvenance(
            tool_name="mp_retriever",
            input_payload=payload.model_dump(),
            output_summary={
                "candidate_count": len(filtered),
                "source": "prompt_qa_fixture",
                "fallback_used": True,
            },
        )
        return MPRetrieverOutput(candidates=filtered, provenance=provenance)


QA_CASES = [
    PromptCase(
        name="wide_band_gap_nitrides",
        goal="Find wide-band-gap nitride semiconductors without silicon and band gap above 3 eV.",
        parser_constraints=DiscoveryConstraints(
            banned_elements=["Si"],
            required_elements=["N"],
            min_band_gap_ev=3.0,
        ),
        expected_required_elements=("N",),
        expected_banned_elements=("Si",),
    ),
    PromptCase(
        name="oxide_constraints",
        goal="Find oxide semiconductors that contain Al, exclude Co, and have band gap above 2 eV.",
        parser_constraints=DiscoveryConstraints(
            banned_elements=["Co"],
            required_elements=["Al", "O"],
            min_band_gap_ev=2.0,
        ),
        expected_required_elements=("Al", "O"),
        expected_banned_elements=("Co",),
    ),
    PromptCase(
        name="practical_screening",
        goal="Find practical semiconductor materials for UV optoelectronics with no lead.",
        parser_constraints=DiscoveryConstraints(banned_elements=["Pb"]),
        expected_banned_elements=("Pb",),
    ),
    PromptCase(
        name="exploratory_screening",
        goal="Find exploratory semiconductor candidates with large band gap, including unusual chemistries.",
        parser_constraints=DiscoveryConstraints(min_band_gap_ev=4.0),
    ),
    PromptCase(
        name="recalculate_top_two",
        goal="Find semiconductor materials and recalculate the top 2 candidates with MatGL.",
        parser_constraints=DiscoveryConstraints(calculate_matgl=True),
        expected_calculate_matgl=True,
        expected_recalculate_top_n=2,
    ),
    PromptCase(
        name="top_seven_no_recalc",
        goal="Find semiconductors. Show me top 7 results and do not recalculate with MatGL.",
        parser_constraints=DiscoveryConstraints(top_k=7, calculate_matgl=False),
        expected_calculate_matgl=False,
        expected_top_k=7,
    ),
    PromptCase(
        name="stability_annotations",
        goal="Find stable semiconductors without silicon and keep max energy above hull below 0.02 eV/atom.",
        parser_constraints=DiscoveryConstraints(
            banned_elements=["Si"],
            max_energy_above_hull=0.02,
            top_k=10,
        ),
        expected_banned_elements=("Si",),
        expected_top_k=10,
        require_known_and_unknown_stability=True,
    ),
    PromptCase(
        name="carbon_not_cobalt",
        goal="Find carbon-containing semiconductor materials but no cobalt.",
        parser_constraints=DiscoveryConstraints(
            banned_elements=["Co"],
            required_elements=["C"],
        ),
        expected_required_elements=("C",),
        expected_banned_elements=("Co",),
        require_carrier_element="C",
    ),
    PromptCase(
        name="unsupported_relax_only",
        goal="Relax this bulk material structure only and return the relaxed geometry.",
        expected_status="unsupported",
        expected_task_class="bulk_relaxation_only",
        expected_reason_code="unsupported_relaxation_only",
        require_candidates=False,
        compact_reason_substring="unsupported",
    ),
    PromptCase(
        name="unsupported_diffusivity",
        goal="Estimate lithium diffusivity in bulk materials using long molecular dynamics runs.",
        expected_status="unsupported",
        expected_task_class="diffusivity_simulation",
        expected_reason_code="unsupported_diffusivity",
        require_candidates=False,
        compact_reason_substring="Diffusivity",
    ),
    PromptCase(
        name="unsupported_md",
        goal="Run molecular dynamics trajectories for oxide materials and summarize the transport behavior.",
        expected_status="unsupported",
        expected_task_class="molecular_dynamics",
        expected_reason_code="unsupported_molecular_dynamics",
        require_candidates=False,
        compact_reason_substring="Molecular dynamics",
    ),
    PromptCase(
        name="unsupported_transport",
        goal="Estimate transport conductivity for nitride materials.",
        expected_status="unsupported",
        expected_task_class="transport_property_estimation",
        expected_reason_code="unsupported_transport",
        require_candidates=False,
        compact_reason_substring="Transport-property",
    ),
    PromptCase(
        name="unsupported_defect",
        goal="Compute nitrogen-vacancy defect properties in GaN.",
        expected_status="unsupported",
        expected_task_class="defect_property_workflow",
        expected_reason_code="unsupported_defect_workflow",
        require_candidates=False,
        compact_reason_substring="Defect-property",
    ),
    PromptCase(
        name="unsupported_dft",
        goal="Run a general DFT electronic-structure workflow for bulk perovskites.",
        expected_status="unsupported",
        expected_task_class="general_ab_initio_simulation",
        expected_reason_code="unsupported_ab_initio",
        require_candidates=False,
        compact_reason_substring="ab initio",
    ),
    PromptCase(
        name="unknown_task",
        goal="Find interesting materials for my project.",
        expected_status="unsupported",
        expected_task_class="unknown_task",
        expected_reason_code="unknown_task_class",
        require_candidates=False,
        compact_reason_substring="could not be mapped",
    ),
]


def _qa_parser(goal: str) -> DiscoveryConstraints:
    by_goal = {case.goal: case.parser_constraints for case in QA_CASES}
    return by_goal[goal].model_copy(deep=True)


def _qa_workflow() -> DiscoveryWorkflow:
    return DiscoveryWorkflow(
        retriever=PromptQARetriever(),
        intent_agent=ChemistryIntentAgent(parser_fn=_qa_parser),
        enable_policy_filter=False,
    )


def _extract_elements(formula: str) -> set[str]:
    return {element.capitalize() for element in MPRetriever._extract_elements(formula)}


@pytest.mark.parametrize("case", QA_CASES, ids=[case.name for case in QA_CASES])
def test_prompt_qa_full_endpoint(case: PromptCase, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_main, "workflow", _qa_workflow())

    response = client.post("/discover/full", json={"research_goal": case.goal})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == case.expected_status
    assert body["discovery_plan"]["task_class"] == case.expected_task_class

    if case.expected_status == "unsupported":
        capability = body["capability_assessment"]
        assert capability["supported"] is False
        assert capability["reason_code"] == case.expected_reason_code
        assert body["candidates"] == []
        return

    assert body["capability_assessment"]["supported"] is True
    if case.require_candidates:
        assert body["candidates"], case.name

    plan = body["discovery_plan"]
    parsed_constraints = plan["parsed_constraints"]
    if case.expected_application_intent is not None:
        assert plan["application_intent"] == case.expected_application_intent
    if case.expected_practicality_mode is not None:
        assert plan["practicality_mode"] == case.expected_practicality_mode
    if case.expected_calculate_matgl is not None:
        assert plan["execution_policy"]["calculate_matgl"] is case.expected_calculate_matgl
    if case.expected_recalculate_top_n is not None:
        assert plan["execution_policy"]["recalculate_top_n"] == case.expected_recalculate_top_n
    if case.expected_top_k is not None:
        assert parsed_constraints["top_k"] == case.expected_top_k

    for symbol in case.expected_required_elements:
        assert symbol in parsed_constraints["required_elements"]
    for symbol in case.expected_banned_elements:
        assert symbol in parsed_constraints["banned_elements"]

    formulas = [candidate["candidate"]["formula"] for candidate in body["candidates"]]
    for formula in formulas:
        elements = _extract_elements(formula)
        for symbol in case.expected_required_elements:
            assert symbol in elements
        for symbol in case.expected_banned_elements:
            assert symbol not in elements

    if case.require_known_and_unknown_stability:
        stability_sources = {candidate["stability"]["source"] for candidate in body["candidates"]}
        assert "materials_project" in stability_sources
        assert "unknown" in stability_sources

    if case.require_carrier_element is not None:
        assert any(
            case.require_carrier_element in _extract_elements(formula)
            for formula in formulas
        )


@pytest.mark.parametrize("case", QA_CASES, ids=[case.name for case in QA_CASES])
def test_prompt_qa_compact_endpoint(case: PromptCase, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_main, "workflow", _qa_workflow())

    response = client.post("/discover", json={"research_goal": case.goal})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == case.expected_status

    if case.expected_status == "unsupported":
        assert body["candidates"] == []
        assert case.compact_reason_substring is not None
        assert case.compact_reason_substring.lower() in body["unsupported_reason"].lower()
        return

    assert body["unsupported_reason"] is None
    if case.require_candidates:
        assert body["candidates"], case.name
