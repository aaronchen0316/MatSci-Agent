from __future__ import annotations

from rich.console import Console
from typer.testing import CliRunner

from matsci_agent.cli import doctor, render, transport
from matsci_agent.cli.main import app
from matsci_agent.schemas import (
    CapabilityAssessment,
    Candidate,
    DiscoveryConstraints,
    DiscoveryFullResponse,
    DiscoveryPlan,
    DiscoverySummaryResponse,
    PolicyFilterRecord,
    PredictedProperties,
    RankedCandidate,
    ReportSummary,
    StabilityResult,
    ToolCallProvenance,
)

runner = CliRunner()


def test_demo_command_renders_candidates():
    result = runner.invoke(
        app,
        [
            "demo",
            "Find semiconductor materials without silicon and band gap above 2 eV",
            "--top-k",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "MatSci Demo" in result.stdout
    assert "Candidates" in result.stdout


def test_render_demo_shows_split_gap_columns_when_matgl_enabled():
    console = Console(record=True, width=140)

    render.render_demo(
        console,
        request=transport.DiscoveryRequest(
            research_goal="Find semiconductor materials",
            constraints=DiscoveryConstraints(calculate_matgl=True),
        ),
        response=DiscoverySummaryResponse(
            status="success",
            candidates=[
                transport.CandidateBandGapSummary(
                    material_id="mp-mock-003",
                    formula="AlN",
                    band_gap_ev=4.8,
                    mp_band_gap_ev=5.9,
                    matgl_band_gap_ev=4.8,
                    band_gap_source="matgl",
                    energy_above_hull=0.01,
                    is_stable=True,
                    stability_source="materials_project",
                    entry_count=1,
                )
            ],
            messages=[],
        ),
    )

    output = console.export_text()
    assert "MP Gap" in output
    assert "MatGL Gap" in output


def test_demo_command_prompts_for_goal():
    result = runner.invoke(app, ["demo"], input="Find high band gap materials\n")

    assert result.exit_code == 0
    assert "Research goal" in result.stdout
    assert "MatSci Demo" in result.stdout


def test_operator_command_renders_all_sections(monkeypatch):
    monkeypatch.setattr(
        transport,
        "run_full",
        lambda request, api_url=None, enable_policy_filter=None: DiscoveryFullResponse(
            research_goal=request.research_goal,
            constraints=request.constraints,
            status="success",
            iterations=1,
            raw_candidates=[
                Candidate(
                    material_id="mp-mock-003",
                    formula="AlN",
                    source="mock",
                    features={"mp_band_gap_ev": 5.9, "mp_energy_above_hull": 0.01, "nsites": 8},
                )
            ],
            filtered_candidates=[
                Candidate(
                    material_id="mp-mock-003",
                    formula="AlN",
                    source="mock",
                    features={"mp_band_gap_ev": 5.9, "mp_energy_above_hull": 0.01, "nsites": 8},
                )
            ],
            filter_records=[
                PolicyFilterRecord(
                    candidate=Candidate(
                        material_id="mp-mock-003",
                        formula="AlN",
                        source="mock",
                        features={"mp_band_gap_ev": 5.9},
                    ),
                    passed=True,
                    reasons=[],
                    policy="disabled_mvp_pass_through",
                )
            ],
            candidates=[
                RankedCandidate(
                    rank=1,
                    candidate=Candidate(
                        material_id="mp-mock-003",
                        formula="AlN",
                        source="mock",
                        features={
                            "band_gap_source": "matgl",
                            "mp_band_gap_ev": 5.9,
                            "matgl_band_gap_ev": 5.1,
                        },
                    ),
                    predicted_properties=PredictedProperties(
                        band_gap_ev=5.1,
                        uncertainty=0.2,
                        backend="matgl_band_gap:test",
                    ),
                    stability=StabilityResult(
                        energy_above_hull=0.01,
                        is_stable=True,
                        method="mp",
                        source="materials_project",
                    ),
                    score=4.9,
                )
            ],
            provenance=[
                ToolCallProvenance(
                    tool_name="mp_retriever",
                    input_payload={},
                    output_summary={"candidate_count": 1},
                )
            ],
            messages=["ok"],
            discovery_plan=DiscoveryPlan(
                research_goal_raw=request.research_goal,
                task_class="band_gap_screening",
                parsed_constraints=request.constraints,
                execution_policy={"calculate_matgl": True},
            ),
            capability_assessment=CapabilityAssessment(supported=True),
            report_summary=ReportSummary(
                scientific_summary="summary",
                execution_summary="exec",
                caveats=[],
            ),
        ),
    )

    result = runner.invoke(app, ["operator", "Find semiconductor materials"])

    assert result.exit_code == 0
    for section in [
        "Effective Constraints",
        "Discovery Plan",
        "Capability Assessment",
        "Raw Candidates",
        "Filtered Candidates",
        "Filter Records",
        "Ranked Candidates",
        "Provenance",
        "Messages / Caveats",
    ]:
        assert section in result.stdout


def test_render_operator_ranked_candidates_show_split_gap_columns():
    console = Console(record=True, width=160)

    render.render_operator(
        console,
        response=DiscoveryFullResponse(
            research_goal="Find semiconductor materials",
            constraints=DiscoveryConstraints(calculate_matgl=True),
            status="success",
            iterations=1,
            raw_candidates=[],
            filtered_candidates=[],
            filter_records=[],
            candidates=[
                RankedCandidate(
                    rank=1,
                    candidate=Candidate(
                        material_id="mp-mock-003",
                        formula="AlN",
                        source="mock",
                        features={
                            "band_gap_source": "matgl",
                            "mp_band_gap_ev": 5.9,
                            "matgl_band_gap_ev": 5.1,
                        },
                    ),
                    predicted_properties=PredictedProperties(
                        band_gap_ev=5.1,
                        uncertainty=0.2,
                        backend="matgl_band_gap:test",
                    ),
                    stability=StabilityResult(
                        energy_above_hull=0.01,
                        is_stable=True,
                        method="mp",
                        source="materials_project",
                    ),
                    score=4.9,
                )
            ],
            provenance=[],
            messages=[],
            discovery_plan=DiscoveryPlan(
                research_goal_raw="Find semiconductor materials",
                task_class="band_gap_screening",
                parsed_constraints=DiscoveryConstraints(calculate_matgl=True),
                execution_policy={"calculate_matgl": True},
            ),
            capability_assessment=CapabilityAssessment(supported=True),
            report_summary=ReportSummary(scientific_summary="summary", execution_summary="exec", caveats=[]),
        ),
    )

    output = console.export_text()
    assert "MP Gap" in output
    assert "MatGL Gap" in output


def test_doctor_renders_status_rows(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "run_doctor_checks",
        lambda api_url=None: [
            doctor.DoctorCheck("PASS", "Core package", "Import available."),
            doctor.DoctorCheck("WARN", "MP API key", "missing"),
        ],
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "MatSci Doctor" in result.stdout
    assert "PASS" in result.stdout
    assert "WARN" in result.stdout


def test_scenarios_list_shows_built_ins():
    result = runner.invoke(app, ["scenarios", "list"])

    assert result.exit_code == 0
    for name in ["basic_success", "policy_filter", "unsupported_request", "matgl_recalc"]:
        assert name in result.stdout


def test_scenarios_run_unsupported_request_shows_structured_refusal(monkeypatch):
    monkeypatch.setattr(
        transport,
        "run_summary",
        lambda request, api_url=None, enable_policy_filter=None: DiscoverySummaryResponse(
            status="unsupported",
            candidates=[],
            messages=["unsupported request"],
            unsupported_reason="outside current system scope",
        ),
    )

    result = runner.invoke(app, ["scenarios", "run", "unsupported_request"])

    assert result.exit_code == 0
    assert "Unsupported" in result.stdout
    assert "outside current system scope" in result.stdout


def test_transport_probe_health_reports_failure():
    ok, detail = transport.probe_health("http://127.0.0.1:9")

    assert ok is False
    assert detail


def test_doctor_failure_exit(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "run_doctor_checks",
        lambda api_url=None: [doctor.DoctorCheck("FAIL", "Core package", "missing")],
    )

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
