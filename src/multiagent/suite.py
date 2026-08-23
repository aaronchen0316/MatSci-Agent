from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from multiagent.artifacts import HarnessArtifactStore
from multiagent.audit import audit_repair_branches
from multiagent.live_suite import LIVE_EVAL_SCENARIOS, run_live_suite
from multiagent.orchestrator import MultiAgentHarness
from multiagent.publisher import publish_and_merge_repair
from multiagent.schemas import (
    HarnessRunReport,
    LiveEvalScenario,
    LiveEvalSuiteReport,
    ModelPreflightReport,
    PullRequestPublication,
    RepairAuditReport,
    RepairSuiteAttempt,
    RepairSuiteReport,
)
from multiagent.settings import MultiAgentSettings
from multiagent.tools import cleanup_worktree, create_target_base_worktree


def repair_prerequisite_error(settings: MultiAgentSettings) -> str | None:
    if not settings.enable_live_mp:
        return "MULTIAGENT_ENABLE_LIVE_MP=1 is required"
    if not settings.enable_git_write:
        return "MULTIAGENT_ENABLE_GIT_WRITE=1 is required"
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(settings.resolved_tool_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        return "tooling checkout must be clean before live repair"
    fetched = subprocess.run(
        ["git", "fetch", "origin", settings.target_base_branch],
        cwd=str(settings.resolved_target_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if fetched.returncode != 0:
        return f"unable to refresh {settings.target_base_ref} before live repair"
    remote = subprocess.run(
        ["git", "rev-parse", "--verify", settings.target_base_ref],
        cwd=str(settings.resolved_target_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if remote.returncode != 0 or not remote.stdout.strip():
        return f"base branch does not exist: {settings.target_base_ref}"
    return None


def refresh_target_base(settings: MultiAgentSettings) -> str | None:
    """Advance the remote-tracking product baseline after an automated merge."""

    result = subprocess.run(
        ["git", "fetch", "origin", settings.target_base_branch],
        cwd=str(settings.resolved_target_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"unable to refresh {settings.target_base_ref} after merge"
    return None


def run_from_target_base(settings: MultiAgentSettings, action):
    created = create_target_base_worktree(settings)
    if created["status"] != "created":
        raise RuntimeError(f"unable to create target base worktree: {created.get('reason', 'unknown error')}")
    worktree_path = created["worktree_path"]
    try:
        return action(replace(settings, active_target_root=Path(worktree_path)))
    finally:
        cleanup_worktree(settings, worktree_path)


def evaluate_live_suite_from_target(settings: MultiAgentSettings) -> LiveEvalSuiteReport:
    return run_from_target_base(settings, run_live_suite)


def run_single_repair(
    settings: MultiAgentSettings,
    scenario: LiveEvalScenario,
    *,
    harness_builder: Callable[[MultiAgentSettings], MultiAgentHarness] = MultiAgentHarness.build,
    publisher: Callable[..., PullRequestPublication] = publish_and_merge_repair,
) -> tuple[HarnessRunReport, PullRequestPublication | None]:
    report = run_from_target_base(
        settings,
        lambda scoped: asyncio.run(harness_builder(scoped).run(scenario.query, scenario=scenario)),
    )
    if report.status != "pass" or report.branch_name is None or report.artifact_dir is None:
        return report, None
    publication = publisher(settings, branch_name=report.branch_name, artifact_dir=Path(report.artifact_dir))
    Path(report.artifact_dir, "pull_request_publication.json").write_text(publication.model_dump_json(indent=2) + "\n")
    return report, publication


def _failed_scenarios(report: LiveEvalSuiteReport) -> list[LiveEvalScenario]:
    return [
        result.scenario
        for result in report.scenarios
        if result.evidence.status != "pass" or result.assertion_failures
    ]


def _suite_status(final: LiveEvalSuiteReport) -> str:
    return "blocked" if final.status == "blocked" else "pass" if final.status == "pass" else "fail"


def run_repair_suite(
    settings: MultiAgentSettings,
    model_preflight: ModelPreflightReport,
    *,
    audit_runner: Callable[[MultiAgentSettings], RepairAuditReport] = audit_repair_branches,
    evaluator: Callable[[MultiAgentSettings], LiveEvalSuiteReport] = evaluate_live_suite_from_target,
    repair_runner: Callable[[MultiAgentSettings, LiveEvalScenario], tuple[HarnessRunReport, PullRequestPublication | None]] = run_single_repair,
    base_refresher: Callable[[MultiAgentSettings], str | None] = refresh_target_base,
) -> RepairSuiteReport:
    """Repair every currently failing live scenario once, retesting all eight after each merge."""

    store = HarnessArtifactStore.create(settings, "multiagent_repair_suite")
    audit = audit_runner(settings)
    store.write_model("repair_audit_report.json", audit)
    store.write_model("model_preflight.json", model_preflight)

    baseline = evaluator(settings)
    store.write_model("baseline_live_suite.json", baseline)
    attempts: list[RepairSuiteAttempt] = []
    current = baseline
    attempted: set[str] = set()
    refresh_error: str | None = None

    while True:
        pending = [scenario for scenario in _failed_scenarios(current) if scenario.name not in attempted]
        if not pending:
            break
        scenario = pending[0]
        attempted.add(scenario.name)
        harness_report, publication = repair_runner(settings, scenario)
        attempts.append(
            RepairSuiteAttempt(
                scenario_name=scenario.name,
                harness_report=harness_report,
                publication=publication,
                summary=(
                    publication.summary
                    if publication is not None
                    else harness_report.summary
                ),
            )
        )
        store.write_model(f"attempts/{len(attempts)}/{scenario.name}.json", attempts[-1])
        if publication is not None and publication.status == "merged":
            refresh_error = base_refresher(settings)
            if refresh_error is None:
                current = evaluator(settings)
                store.write_model(f"retests/{len(attempts)}_live_suite.json", current)
            else:
                break

    final = evaluator(settings)
    store.write_model("final_live_suite.json", final)
    status = "fail" if refresh_error else _suite_status(final)
    summary = (
        refresh_error
        if refresh_error
        else "all eight live scenarios passed"
        if status == "pass"
        else "one or more live scenarios remain failed or blocked"
    )
    report = RepairSuiteReport(
        status=status,
        summary=summary,
        artifact_dir=str(store.run_dir),
        model_preflight=model_preflight,
        audit_report=audit,
        baseline=baseline,
        attempts=attempts,
        final=final,
    )
    store.write_model("repair_suite_report.json", report)
    return report


def blocked_repair_suite(
    settings: MultiAgentSettings,
    model_preflight: ModelPreflightReport,
    *,
    summary: str,
) -> RepairSuiteReport:
    store = HarnessArtifactStore.create(settings, "multiagent_repair_suite")
    audit = audit_repair_branches(settings)
    report = RepairSuiteReport(
        status="blocked",
        summary=summary,
        artifact_dir=str(store.run_dir),
        model_preflight=model_preflight,
        audit_report=audit,
    )
    store.write_model("model_preflight.json", model_preflight)
    store.write_model("repair_audit_report.json", audit)
    store.write_model("repair_suite_report.json", report)
    return report
