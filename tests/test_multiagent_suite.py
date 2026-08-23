from __future__ import annotations

from pathlib import Path

from multiagent.live_suite import LIVE_EVAL_SCENARIOS
from multiagent.schemas import (
    HarnessRunReport,
    LiveEvalEvidence,
    LiveEvalScenarioResult,
    LiveEvalSuiteReport,
    ModelPreflightReport,
    PullRequestPublication,
    RepairAuditReport,
)
from multiagent.settings import MultiAgentSettings
from multiagent.suite import run_repair_suite


def _settings(tmp_path: Path) -> MultiAgentSettings:
    return MultiAgentSettings(tool_root=tmp_path, target_repo=tmp_path, artifact_root=tmp_path / "artifacts")


def _preflight() -> ModelPreflightReport:
    return ModelPreflightReport(
        status="pass",
        primary_model="gpt-5.4-mini",
        selected_model="gpt-5.4-mini",
        selected_product_model="gpt-5.4-mini",
        attempts=["gpt-5.4-mini"],
        summary="ok",
    )


def _suite(*failed: str) -> LiveEvalSuiteReport:
    results = []
    for scenario in LIVE_EVAL_SCENARIOS:
        is_failed = scenario.name in failed
        results.append(
            LiveEvalScenarioResult(
                scenario=scenario,
                evidence=LiveEvalEvidence(
                    status="fail" if is_failed else "pass",
                    query=scenario.query,
                    failed_stage="llm_policy_filter" if is_failed else None,
                    real_source_used=not is_failed,
                ),
            )
        )
    return LiveEvalSuiteReport(status="fail" if failed else "pass", scenarios=results)


def _harness_report(scenario: str, status: str = "fail") -> HarnessRunReport:
    return HarnessRunReport(
        status=status,
        summary=f"{scenario} {status}",
        next_step="inspect",
        stop_reason="debugger_blocked" if status == "blocked" else "verifier_fail",
        attempt_count=1,
    )


def test_repair_suite_attempts_each_initial_failure_once_and_retests_after_merge(tmp_path: Path):
    settings = _settings(tmp_path)
    evaluations = [_suite("volume", "formation_energy"), _suite("formation_energy"), _suite("formation_energy")]
    repaired: list[str] = []
    refreshes: list[str] = []

    def evaluator(_settings):
        return evaluations.pop(0)

    def repair(_settings, scenario):
        repaired.append(scenario.name)
        if scenario.name == "volume":
            return _harness_report(scenario.name), PullRequestPublication(
                status="merged",
                branch_name="fix/volume",
                base_branch="main",
                summary="merged",
            )
        return _harness_report(scenario.name, "blocked"), None

    report = run_repair_suite(
        settings,
        _preflight(),
        audit_runner=lambda _settings: RepairAuditReport(status="fail", target_base_branch="main"),
        evaluator=evaluator,
        repair_runner=repair,
        base_refresher=lambda _settings: refreshes.append("main") or None,
    )

    assert repaired == ["formation_energy", "volume"]
    assert refreshes == ["main"]
    assert report.status == "fail"
    assert [attempt.scenario_name for attempt in report.attempts] == repaired
    artifact = Path(report.artifact_dir or "")
    assert (artifact / "baseline_live_suite.json").is_file()
    assert (artifact / "retests/2_live_suite.json").is_file()
    assert (artifact / "final_live_suite.json").is_file()


def test_repair_suite_fails_when_merged_base_cannot_refresh(tmp_path: Path):
    settings = _settings(tmp_path)
    evaluations = [_suite("volume"), _suite("volume")]

    report = run_repair_suite(
        settings,
        _preflight(),
        audit_runner=lambda _settings: RepairAuditReport(status="pass", target_base_branch="main"),
        evaluator=lambda _settings: evaluations.pop(0),
        repair_runner=lambda _settings, scenario: (
            _harness_report(scenario.name),
            PullRequestPublication(status="merged", branch_name="fix/volume", base_branch="main", summary="merged"),
        ),
        base_refresher=lambda _settings: "unable to refresh origin/main after merge",
    )

    assert report.status == "fail"
    assert report.summary == "unable to refresh origin/main after merge"
