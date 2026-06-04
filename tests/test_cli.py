from __future__ import annotations

from typer.testing import CliRunner

from matsci_agent.cli import doctor, transport
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
                        features={"band_gap_source": "materials_project"},
                    ),
                    predicted_properties=PredictedProperties(
                        band_gap_ev=5.9,
                        uncertainty=0.2,
                        backend="materials_project_band_gap",
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
            discovery_plan=DiscoveryPlan(research_goal_raw=request.research_goal, task_class="band_gap_screening"),
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
