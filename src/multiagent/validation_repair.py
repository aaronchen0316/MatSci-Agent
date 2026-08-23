from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from multiagent.artifacts import HarnessArtifactStore
from multiagent.live_suite import LiveEvalScenario, LiveEvalSuiteReport, run_live_suite
from multiagent.orchestrator import MultiAgentHarness
from multiagent.publisher import publish_and_merge_repair
from multiagent.schemas import (
    HarnessRunReport,
    ModelPreflightReport,
    PullRequestPublication,
    ValidationRepairAttempt,
    ValidationRepairReport,
)
from multiagent.settings import MultiAgentSettings
from multiagent.tools import cleanup_worktree, create_target_base_worktree


def validation_repair_prerequisite_error(settings: MultiAgentSettings) -> str | None:
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
        return "tooling checkout must be clean before validation-repair"
    return refresh_target_base(settings)


def refresh_target_base(settings: MultiAgentSettings) -> str | None:
    """Refresh the remote product baseline used by every isolated evaluation."""

    fetched = subprocess.run(
        ["git", "fetch", "origin", settings.target_base_branch],
        cwd=str(settings.resolved_target_repo),
        capture_output=True,
        text=True,
        check=False,
    )
    if fetched.returncode != 0:
        return f"unable to refresh {settings.target_base_ref}"
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


def run_in_product_worktree(settings: MultiAgentSettings, action):
    created = create_target_base_worktree(settings)
    if created["status"] != "created":
        raise RuntimeError(f"unable to create target base worktree: {created.get('reason', 'unknown error')}")
    worktree_path = created["worktree_path"]
    try:
        return action(replace(settings, active_target_root=Path(worktree_path)))
    finally:
        cleanup_worktree(settings, worktree_path)


def run_live_validation(settings: MultiAgentSettings) -> LiveEvalSuiteReport:
    return run_in_product_worktree(settings, run_live_suite)


def run_scenario_repair(
    settings: MultiAgentSettings,
    scenario: LiveEvalScenario,
    *,
    harness_builder: Callable[[MultiAgentSettings], MultiAgentHarness] = MultiAgentHarness.build,
    publisher: Callable[..., PullRequestPublication] = publish_and_merge_repair,
) -> tuple[HarnessRunReport, PullRequestPublication | None]:
    report = run_in_product_worktree(
        settings,
        lambda scoped: asyncio.run(harness_builder(scoped).repair_scenario(scenario)),
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


def _validation_status(final: LiveEvalSuiteReport) -> str:
    return "blocked" if final.status == "blocked" else "pass" if final.status == "pass" else "fail"


def run_validation_repair(
    settings: MultiAgentSettings,
    model_preflight: ModelPreflightReport,
    *,
    evaluator: Callable[[MultiAgentSettings], LiveEvalSuiteReport] = run_live_validation,
    repair_runner: Callable[[MultiAgentSettings, LiveEvalScenario], tuple[HarnessRunReport, PullRequestPublication | None]] = run_scenario_repair,
    base_refresher: Callable[[MultiAgentSettings], str | None] = refresh_target_base,
) -> ValidationRepairReport:
    """Validate eight live scenarios, repair each failure once, then revalidate."""

    store = HarnessArtifactStore.create(settings, "multiagent_validation_repair")
    store.write_model("model_preflight.json", model_preflight)

    baseline = evaluator(settings)
    store.write_model("baseline_validation.json", baseline)
    attempts: list[ValidationRepairAttempt] = []
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
        attempt = ValidationRepairAttempt(
            scenario_name=scenario.name,
            harness_report=harness_report,
            publication=publication,
            summary=publication.summary if publication is not None else harness_report.summary,
        )
        attempts.append(attempt)
        store.write_model(f"attempts/{len(attempts)}/{scenario.name}.json", attempt)
        if publication is not None and publication.status == "merged":
            refresh_error = base_refresher(settings)
            if refresh_error is not None:
                break
            current = evaluator(settings)
            store.write_model(f"retests/{len(attempts)}_validation.json", current)

    final = evaluator(settings)
    store.write_model("final_validation.json", final)
    status = "fail" if refresh_error else _validation_status(final)
    summary = (
        refresh_error
        if refresh_error
        else "all eight live scenarios passed"
        if status == "pass"
        else "one or more live scenarios remain failed or blocked"
    )
    report = ValidationRepairReport(
        status=status,
        summary=summary,
        artifact_dir=str(store.run_dir),
        model_preflight=model_preflight,
        baseline=baseline,
        attempts=attempts,
        final=final,
    )
    store.write_model("validation_repair_report.json", report)
    return report


def blocked_validation_repair(
    settings: MultiAgentSettings,
    model_preflight: ModelPreflightReport,
    *,
    summary: str,
) -> ValidationRepairReport:
    store = HarnessArtifactStore.create(settings, "multiagent_validation_repair")
    report = ValidationRepairReport(
        status="blocked",
        summary=summary,
        artifact_dir=str(store.run_dir),
        model_preflight=model_preflight,
    )
    store.write_model("model_preflight.json", model_preflight)
    store.write_model("validation_repair_report.json", report)
    return report
