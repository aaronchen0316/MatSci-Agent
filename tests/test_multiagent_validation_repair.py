from __future__ import annotations

from pathlib import Path

import multiagent.validation_repair as validation_repair
from multiagent.live_suite import LIVE_EVAL_SCENARIOS
from multiagent.schemas import (
    AdoptedBranchAttempt,
    CodexDebuggerReport,
    HarnessRunReport,
    LiveEvalEvidence,
    LiveEvalScenarioResult,
    LiveEvalSuiteReport,
    ModelPreflightReport,
    PullRequestPublication,
)
from multiagent.settings import MultiAgentSettings
from multiagent.validation_repair import prepare_control_baseline, run_validation_repair


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


def test_validation_repair_attempts_each_failure_once_and_retests_after_merge(tmp_path: Path):
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

    report = run_validation_repair(
        settings,
        _preflight(),
        evaluator=evaluator,
        repair_runner=repair,
        base_refresher=lambda _settings: refreshes.append("main") or None,
    )

    assert repaired == ["formation_energy", "volume"]
    assert refreshes == ["main"]
    assert report.status == "fail"
    assert [attempt.scenario_name for attempt in report.attempts] == repaired
    artifact = Path(report.artifact_dir or "")
    assert (artifact / "baseline_validation.json").is_file()
    assert (artifact / "retests/2_validation.json").is_file()
    assert (artifact / "final_validation.json").is_file()


def test_validation_repair_blocks_when_merged_control_baseline_cannot_prepare(tmp_path: Path):
    settings = _settings(tmp_path)
    evaluations = [_suite("volume"), _suite("volume")]

    report = run_validation_repair(
        settings,
        _preflight(),
        evaluator=lambda _settings: evaluations.pop(0),
        repair_runner=lambda _settings, scenario: (
            _harness_report(scenario.name),
            PullRequestPublication(status="merged", branch_name="fix/volume", base_branch="main", summary="merged"),
        ),
        base_refresher=lambda _settings: "unable to refresh origin/main after merge",
    )

    assert report.status == "blocked"
    assert report.summary == "unable to refresh origin/main after merge"


def test_control_baseline_requires_synced_product_paths(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)

    def run(args, _cwd):
        if args[:2] == ["git", "diff"]:
            return __import__("subprocess").CompletedProcess(args, 1, "", "")
        if args[:3] == ["git", "branch", "--show-current"]:
            return __import__("subprocess").CompletedProcess(args, 0, "multi-agent\n", "")
        return __import__("subprocess").CompletedProcess(args, 0, "main-sha\n" if "rev-parse" in args else "", "")

    monkeypatch.setattr(validation_repair, "_run", run)

    assert prepare_control_baseline(settings) == "forward-port product changes from multi-agent to origin/main before validation"


def test_control_baseline_accepts_harness_only_divergence(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)

    def run(args, _cwd):
        if args[:3] == ["git", "branch", "--show-current"]:
            return __import__("subprocess").CompletedProcess(args, 0, "multi-agent\n", "")
        return __import__("subprocess").CompletedProcess(args, 0, "main-sha\n" if "rev-parse" in args else "", "")

    monkeypatch.setattr(validation_repair, "_run", run)

    assert prepare_control_baseline(settings) is None


def test_validation_repair_processes_explicit_adoptions_before_fresh_repairs(tmp_path: Path):
    settings = _settings(tmp_path)
    evaluations = [_suite("formation_energy", "volume"), _suite("volume"), _suite("volume")]
    adopted: list[str] = []
    repaired: list[str] = []

    def adopt(_settings, branch_name):
        adopted.append(branch_name)
        return AdoptedBranchAttempt(
            branch_name=branch_name,
            scenario_name="formation_energy",
            status="merged",
            summary="merged",
        )

    report = run_validation_repair(
        settings,
        _preflight(),
        evaluator=lambda _settings: evaluations.pop(0),
        adoption_runner=adopt,
        repair_runner=lambda _settings, scenario: (
            repaired.append(scenario.name) or _harness_report(scenario.name, "blocked"),
            None,
        ),
        base_refresher=lambda _settings: None,
        adopt_branches=["fix/formation"],
    )

    assert adopted == ["fix/formation"]
    assert repaired == ["volume"]
    assert report.adopted_branches[0].status == "merged"
    artifact = Path(report.artifact_dir or "")
    assert (artifact / "adopted/1/fix_formation.json").is_file()
    assert (artifact / "adopted_retests/1_validation.json").is_file()


def test_adoption_deletes_explicit_noop_branch(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(validation_repair, "_branch_head", lambda *_args: "head-sha")
    monkeypatch.setattr(validation_repair, "_has_product_diff", lambda *_args: (False, None))

    def run(args, _cwd):
        commands.append(args)
        return __import__("subprocess").CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(validation_repair, "_run", run)

    result = validation_repair.run_adopted_branch_repair(settings, "fix/noop")

    assert result.status == "deleted"
    assert any(command[:3] == ["git", "branch", "-D"] for command in commands)


def test_adoption_uses_existing_patch_evidence_without_new_debugger(monkeypatch, tmp_path: Path):
    settings = _settings(tmp_path)
    worktree = tmp_path / "adopted"
    worktree.mkdir()
    scenario = LIVE_EVAL_SCENARIOS[0]
    debugger = CodexDebuggerReport(
        status="patched",
        branch_name="fix/oxide",
        worktree_path=None,
        commit_sha="head-sha",
        files_touched=["src/matsci_agent/tools/mp_retriever.py", "tests/test_mp_retriever.py"],
        test_files=["tests/test_mp_retriever.py"],
        test_targets=["tests/test_mp_retriever.py"],
        change_summary="existing patch",
    )
    context = validation_repair._AdoptionContext(
        branch_name="fix/oxide",
        head_sha="head-sha",
        scenario=scenario,
        debugger_report=debugger,
        artifact_dir=tmp_path / "old-artifact",
    )
    report = _harness_report(scenario.name, "pass").model_copy(
        update={
            "branch_name": "fix/oxide",
            "artifact_dir": str(tmp_path / "artifact"),
            "stop_reason": "dual_review_pass",
        }
    )
    Path(report.artifact_dir or "").mkdir()
    calls: dict[str, object] = {}

    class FakeHarness:
        async def repair_scenario(self, received_scenario, **kwargs):
            calls["scenario"] = received_scenario
            calls.update(kwargs)
            return report

    monkeypatch.setattr(validation_repair, "_branch_head", lambda *_args: "head-sha")
    monkeypatch.setattr(validation_repair, "_has_product_diff", lambda *_args: (True, None))
    monkeypatch.setattr(validation_repair, "_product_only_diff", lambda *_args: None)
    monkeypatch.setattr(validation_repair, "_matching_adoption_context", lambda *_args: context)
    monkeypatch.setattr(validation_repair, "_rebase_adopted_branch", lambda *_args: ("head-sha", None))
    monkeypatch.setattr(validation_repair, "_create_adoption_worktree", lambda *_args: (worktree, None))
    monkeypatch.setattr(validation_repair, "cleanup_worktree", lambda *_args: {"status": "removed"})
    result = validation_repair.run_adopted_branch_repair(
        settings,
        "fix/oxide",
        harness_builder=lambda _settings: FakeHarness(),
        publisher=lambda *_args, **_kwargs: PullRequestPublication(
            status="merged", branch_name="fix/oxide", base_branch="main", summary="merged"
        ),
    )

    assert result.status == "merged"
    assert calls["existing_branch_name"] == "fix/oxide"
    adopted = calls["adopted_debugger_report"]
    assert isinstance(adopted, CodexDebuggerReport)
    assert adopted.commit_sha == "head-sha"
    assert adopted.worktree_path == str(worktree)
