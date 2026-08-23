from __future__ import annotations

from matsci_agent.multiagent.evaluator import LiveRetrievalEvaluator
from matsci_agent.multiagent.schemas import LiveEvalInput, RetrievalTesterReport
from matsci_agent.schemas import (
    Candidate,
    DiscoveryConstraints,
    DiscoveryFullResponse,
    MPFilters,
    PolicyFilterRecord,
    PredictedProperties,
    RankedCandidate,
    StabilityResult,
    ToolCallProvenance,
)


class FakeWorkflow:
    def __init__(self, response: DiscoveryFullResponse) -> None:
        self.response = response

    def run_full(self, _request):
        return self.response


def _candidate(material_id: str, formula: str, **features) -> Candidate:
    return Candidate(
        material_id=material_id,
        formula=formula,
        source="materials_project",
        features=features,
    )


def _ranked(candidate: Candidate) -> RankedCandidate:
    return RankedCandidate(
        rank=1,
        candidate=candidate,
        predicted_properties=PredictedProperties(
            band_gap_ev=float(candidate.features.get("mp_band_gap_ev") or 0.0),
            uncertainty=0.0,
            backend="mp_summary",
        ),
        stability=StabilityResult(
            energy_above_hull=candidate.features.get("mp_energy_above_hull"),
            is_stable=True,
            method="materials_project_summary_properties",
            source="materials_project",
        ),
        score=1.0,
    )


def _retriever_provenance(
    *,
    source: str,
    candidate_count: int,
    search_kwargs: dict | None = None,
    search_kwargs_sequence: list[dict] | None = None,
    fallback_used: bool = False,
) -> ToolCallProvenance:
    return ToolCallProvenance(
        tool_name="mp_retriever",
        input_payload={"research_goal": "query"},
        output_summary={
            "source": source,
            "candidate_count": candidate_count,
            "fallback_used": fallback_used,
            "search_kwargs": search_kwargs or {},
            "search_kwargs_sequence": search_kwargs_sequence or ([] if search_kwargs is None else [search_kwargs]),
            "search_space_target_count": 2,
        },
    )


def _default_search_kwargs(**overrides) -> dict:
    kwargs = {"energy_above_hull": [0.0, 0.1]}
    kwargs.update(overrides)
    return kwargs


def _response(
    *,
    constraints: DiscoveryConstraints | None = None,
    raw_candidates: list[Candidate] | None = None,
    filtered_candidates: list[Candidate] | None = None,
    ranked_candidates: list[RankedCandidate] | None = None,
    filter_records: list[PolicyFilterRecord] | None = None,
    provenance: list[ToolCallProvenance] | None = None,
    status: str = "success",
    messages: list[str] | None = None,
) -> DiscoveryFullResponse:
    return DiscoveryFullResponse(
        research_goal="query",
        constraints=constraints or DiscoveryConstraints(),
        status=status,
        iterations=1,
        raw_candidates=raw_candidates or [],
        filtered_candidates=filtered_candidates or [],
        filter_records=filter_records or [],
        candidates=ranked_candidates or [],
        provenance=provenance or [],
        messages=messages or [],
        search_space_targets=[],
    )


def test_live_evaluator_returns_blocked_when_live_disabled(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(_response()))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=False))

    assert evidence.status == "blocked"
    assert evidence.blocked_reason == "live MP eval disabled"


def test_live_evaluator_returns_blocked_when_missing_mp_credentials(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(_response()))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=True))

    assert evidence.status == "blocked"
    assert evidence.blocked_reason == "missing MP_API_KEY"


def test_live_evaluator_rejects_mock_fallback_provenance(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    response = _response(
        provenance=[
            _retriever_provenance(
                source="mock_fallback_no_live_results",
                candidate_count=0,
                fallback_used=True,
            )
        ],
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=True))

    assert evidence.status == "blocked"
    assert "mock fallback" in evidence.blocked_reason


def test_live_evaluator_extracts_compiled_filters_and_counts(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    constraints = DiscoveryConstraints(required_elements=["O"], min_band_gap_ev=2.0)
    candidate = _candidate(
        "mp-1",
        "SiO2",
        mp_band_gap_ev=5.5,
        mp_energy_above_hull=0.01,
        nsites=6,
        is_metal=False,
    )
    search_kwargs = _default_search_kwargs(elements=["O"], band_gap=[2.0, 20.0])
    response = _response(
        constraints=constraints,
        raw_candidates=[candidate],
        filtered_candidates=[candidate],
        ranked_candidates=[_ranked(candidate)],
        provenance=[
            _retriever_provenance(
                source="materials_project",
                candidate_count=1,
                search_kwargs=search_kwargs,
            )
        ],
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=True))

    assert evidence.status == "pass"
    assert evidence.real_source_used is True
    assert evidence.compiled_filters.effective_filters["elements"] == ["O"]
    assert evidence.compiled_filters.mp_search_kwargs == search_kwargs
    assert evidence.result_counts.raw_count == 1
    assert evidence.result_counts.filtered_count == 1
    assert evidence.result_counts.ranked_count == 1


def test_live_evaluator_builds_bounded_whitelisted_candidate_snapshots(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    candidates = [
        _candidate(
            f"mp-{index}",
            "TiO2",
            elements=["Ti", "O"],
            mp_band_gap_ev=3.0,
            mp_energy_above_hull=0.01,
            formation_energy=-2.4,
            density=4.2,
            volume=62.0,
            nsites=6,
            crystal_system="tetragonal",
            spacegroup_number=136,
            spacegroup_symbol="P4_2/mnm",
            structure={"should_not": "reach critic"},
            arbitrary_payload="do not copy",
        )
        for index in range(25)
    ]
    response = _response(
        raw_candidates=candidates,
        filtered_candidates=candidates,
        ranked_candidates=[_ranked(candidate) for candidate in candidates[:3]],
        filter_records=[
            PolicyFilterRecord(
                candidate=candidates[0],
                passed=False,
                reasons=["outside requested material family"],
                policy="chemistry_screening",
            )
        ],
        provenance=[
            _retriever_provenance(
                source="materials_project",
                candidate_count=25,
                search_kwargs=_default_search_kwargs(),
            )
        ],
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=True))

    snapshots = evidence.candidate_snapshots
    assert len(snapshots.raw) == 20
    assert len(snapshots.filtered) == 20
    assert len(snapshots.ranked) == 3
    assert snapshots.raw[0].policy_passed is False
    assert snapshots.raw[0].policy_reasons == ["outside requested material family"]
    assert snapshots.raw[0].elements == ["O", "Ti"]
    assert "structure" not in snapshots.raw[0].model_dump()
    assert "arbitrary_payload" not in snapshots.raw[0].model_dump()


def test_live_evaluator_maps_search_space_failure(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    response = _response(
        provenance=[
            ToolCallProvenance(
                tool_name="search_space_expander",
                input_payload={"research_goal": "query"},
                output_summary={"failure_code": "search_space_expansion_empty"},
            )
        ],
        status="failed",
        messages=["Search-space expansion failed."],
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=True))

    assert evidence.status == "fail"
    assert evidence.failed_stage == "search_space_expansion"


def test_live_evaluator_maps_zero_results_from_real_mp(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    search_kwargs = _default_search_kwargs(elements=["O"])
    response = _response(
        provenance=[
            _retriever_provenance(
                source="materials_project",
                candidate_count=0,
                search_kwargs=search_kwargs,
            )
        ],
        status="failed",
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=True))

    assert evidence.status == "fail"
    assert evidence.failed_stage == "mp_zero_results"


def test_live_evaluator_reports_constraint_violations_by_stage(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    constraints = DiscoveryConstraints(required_elements=["O"], min_band_gap_ev=2.0)
    raw_bad = _candidate("mp-raw", "SiC", mp_band_gap_ev=1.0, mp_energy_above_hull=0.01, nsites=2, is_metal=False)
    filtered_bad = _candidate("mp-filtered", "Si", mp_band_gap_ev=4.0, mp_energy_above_hull=0.01, nsites=2, is_metal=False)
    ranked_bad = _candidate("mp-ranked", "Si", mp_band_gap_ev=4.0, mp_energy_above_hull=0.01, nsites=2, is_metal=False)
    response = _response(
        constraints=constraints,
        raw_candidates=[raw_bad],
        filtered_candidates=[filtered_bad],
        ranked_candidates=[_ranked(ranked_bad)],
        provenance=[
            _retriever_provenance(
                source="materials_project",
                candidate_count=1,
                search_kwargs=_default_search_kwargs(elements=["O"], band_gap=[2.0, 20.0]),
            )
        ],
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find oxides", allow_live_mp=True))

    assert evidence.constraint_violations.raw[0].stage == "raw"
    assert "missing_required_elements" in evidence.constraint_violations.filtered[0].violations
    assert evidence.constraint_violations.ranked[0].stage == "ranked"


def test_live_evaluator_passes_explicit_constraints_to_workflow(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    response = _response(
        provenance=[
            _retriever_provenance(
                source="materials_project",
                candidate_count=0,
                search_kwargs=_default_search_kwargs(density=[0.0, 5.0]),
            )
        ],
        status="failed",
    )

    class CapturingWorkflow(FakeWorkflow):
        request = None

        def run_full(self, request):
            self.request = request
            return self.response

    workflow = CapturingWorkflow(response)
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: workflow)
    constraints = DiscoveryConstraints(mp_filters=MPFilters(density={"max": 5.0}))

    evaluator.evaluate(LiveEvalInput(query="find low-density materials", constraints=constraints, allow_live_mp=True))

    assert workflow.request.constraints.mp_filters.density.max == 5.0


def test_live_evaluator_rejects_missing_compiled_has_props_filter(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    constraints = DiscoveryConstraints(mp_filters=MPFilters(has_props=["dielectric"]))
    candidate = _candidate("mp-1", "SiO2", mp_band_gap_ev=5.5, mp_energy_above_hull=0.01, nsites=6)
    response = _response(
        constraints=constraints,
        raw_candidates=[candidate],
        filtered_candidates=[candidate],
        ranked_candidates=[_ranked(candidate)],
        provenance=[
            _retriever_provenance(
                source="materials_project",
                candidate_count=1,
                search_kwargs=_default_search_kwargs(),
            )
        ],
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find dielectric materials", allow_live_mp=True))

    assert evidence.status == "fail"
    assert evidence.failed_stage == "mp_query_compilation"
    assert evidence.compiled_filters.missing_filter_keys == ["has_props"]


def test_live_evaluator_reports_cubic_symmetry_mismatch(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "token")
    constraints = DiscoveryConstraints(mp_filters=MPFilters(crystal_system="cubic"))
    candidate = _candidate(
        "mp-1",
        "SiO2",
        mp_band_gap_ev=5.5,
        mp_energy_above_hull=0.01,
        nsites=6,
        crystal_system="tetragonal",
    )
    response = _response(
        constraints=constraints,
        raw_candidates=[candidate],
        filtered_candidates=[candidate],
        ranked_candidates=[_ranked(candidate)],
        provenance=[
            _retriever_provenance(
                source="materials_project",
                candidate_count=1,
                search_kwargs=_default_search_kwargs(crystal_system="cubic"),
            )
        ],
    )
    evaluator = LiveRetrievalEvaluator(workflow_factory=lambda: FakeWorkflow(response))

    evidence = evaluator.evaluate(LiveEvalInput(query="find cubic materials", allow_live_mp=True))

    assert evidence.status == "fail"
    assert "crystal_system_mismatch" in evidence.constraint_violations.ranked[0].violations


def test_retrieval_tester_report_accepts_blocked_status():
    report = RetrievalTesterReport(status="blocked", summary="live MP unavailable")

    assert report.status == "blocked"
