from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from multiagent.artifacts import HarnessArtifactStore
from multiagent.live_suite import LiveEvalScenario, LiveEvalSuiteReport, get_live_scenario, run_live_suite
from multiagent.orchestrator import MultiAgentHarness
from multiagent.publisher import _product_only_diff, _validate_branch, publish_and_merge_repair
from multiagent.schemas import (
    AdoptedBranchAttempt,
    CodexDebuggerReport,
    HarnessRunReport,
    ModelPreflightReport,
    PullRequestPublication,
    ValidationRepairAttempt,
    ValidationRepairReport,
)
from multiagent.settings import MultiAgentSettings
from multiagent.tools import cleanup_worktree, create_target_base_worktree

_CONTROL_BRANCH = "multi-agent"
_PRODUCT_PATHS = (
    "src/matsci_agent",
    "tests",
    "README.md",
    "CONTEXT.md",
    ":(exclude)tests/test_multiagent_*.py",
)


@dataclass(frozen=True)
class _AdoptionContext:
    branch_name: str
    head_sha: str
    scenario: LiveEvalScenario
    debugger_report: CodexDebuggerReport
    artifact_dir: Path


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


def validation_repair_prerequisite_error(settings: MultiAgentSettings) -> str | None:
    if not settings.enable_live_mp:
        return "MULTIAGENT_ENABLE_LIVE_MP=1 is required"
    if not settings.enable_git_write:
        return "MULTIAGENT_ENABLE_GIT_WRITE=1 is required"
    return prepare_control_baseline(settings)


def prepare_control_baseline(settings: MultiAgentSettings) -> str | None:
    """Refresh and verify the synchronized product/control baseline."""

    target_root = settings.resolved_target_repo
    tool_root = settings.resolved_tool_root
    for root in {target_root, tool_root}:
        fetched = _run(["git", "fetch", "origin", settings.target_base_branch], root)
        if fetched.returncode != 0:
            return f"unable to refresh {settings.target_base_ref}"
    target_base = _run(["git", "rev-parse", "--verify", settings.target_base_ref], target_root)
    tool_base = _run(["git", "rev-parse", "--verify", settings.target_base_ref], tool_root)
    if target_base.returncode != 0 or tool_base.returncode != 0 or not target_base.stdout.strip() or not tool_base.stdout.strip():
        return f"base branch does not exist: {settings.target_base_ref}"
    if target_base.stdout.strip() != tool_base.stdout.strip():
        return "tooling checkout target base does not match product checkout"
    if _output(_run(["git", "branch", "--show-current"], tool_root)) != _CONTROL_BRANCH:
        return f"tooling checkout must be on {_CONTROL_BRANCH}"
    if _output(_run(["git", "status", "--porcelain"], tool_root)):
        return "tooling checkout must be clean before validation"
    if _run(["git", "merge-base", "--is-ancestor", settings.target_base_ref, "HEAD"], tool_root).returncode != 0:
        return f"{_CONTROL_BRANCH} must merge {settings.target_base_ref} before validation"
    product_diff = _run(["git", "diff", "--quiet", f"{settings.target_base_ref}...HEAD", "--", *_PRODUCT_PATHS], tool_root)
    if product_diff.returncode == 1:
        return f"forward-port product changes from {_CONTROL_BRANCH} to {settings.target_base_ref} before validation"
    if product_diff.returncode != 0:
        return "unable to compare product changes between control and target branches"
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


def _branch_head(settings: MultiAgentSettings, branch_name: str) -> str | None:
    result = _run(["git", "rev-parse", "--verify", branch_name], settings.resolved_target_repo)
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _has_product_diff(settings: MultiAgentSettings, branch_name: str) -> tuple[bool, str | None]:
    result = _run(["git", "diff", "--quiet", f"{settings.target_base_ref}...{branch_name}"], settings.resolved_target_repo)
    if result.returncode in {0, 1}:
        return result.returncode == 1, None
    return False, f"unable to inspect branch diff: {_output(result)}"


def _matching_adoption_context(settings: MultiAgentSettings, branch_name: str, head_sha: str) -> _AdoptionContext | str:
    matches: list[tuple[Path, HarnessRunReport]] = []
    for report_path in settings.resolved_artifact_root.glob("*/harness_run_report.json"):
        try:
            report = HarnessRunReport.model_validate_json(report_path.read_text())
        except Exception:
            continue
        debugger = report.latest_debugger_report
        if report.branch_name == branch_name and debugger and debugger.status == "patched" and debugger.commit_sha == head_sha:
            matches.append((report_path.parent, report))
    if not matches:
        return "no stored debugger evidence matches the current branch head"
    artifact_dir, report = max(matches, key=lambda item: item[0].name)
    scenario_name = str((report.latest_tester_report.evidence if report.latest_tester_report else {}).get("scenario_name") or "")
    if not scenario_name:
        return "stored debugger evidence does not name a live scenario"
    try:
        scenario = get_live_scenario(scenario_name)
    except ValueError:
        return "stored debugger evidence names an unknown live scenario"
    debugger = report.latest_debugger_report
    assert debugger is not None
    return _AdoptionContext(
        branch_name=branch_name,
        head_sha=head_sha,
        scenario=scenario,
        debugger_report=debugger,
        artifact_dir=artifact_dir,
    )


def _rebase_adopted_branch(settings: MultiAgentSettings, context: _AdoptionContext) -> tuple[str | None, str | None]:
    ancestor = _run(
        ["git", "merge-base", "--is-ancestor", settings.target_base_ref, context.branch_name],
        settings.resolved_target_repo,
    )
    if ancestor.returncode == 0:
        return context.head_sha, None
    root = settings.worktree_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    worktree = root / f".adopt-rebase-{uuid4().hex[:12]}"
    add = _run(["git", "worktree", "add", str(worktree), context.branch_name], settings.resolved_target_repo)
    if add.returncode != 0:
        return None, f"unable to create adoption rebase worktree: {_output(add)}"
    rebase = _run(["git", "rebase", settings.target_base_ref], worktree)
    if rebase.returncode != 0:
        return None, f"rebase failed; retained worktree {worktree}: {_output(rebase)}"
    head = _branch_head(settings, context.branch_name)
    cleanup = cleanup_worktree(settings, str(worktree))
    if cleanup.get("status") != "removed":
        return None, f"rebased branch but could not remove temporary worktree: {cleanup.get('reason', 'unknown error')}"
    return head, None


def _create_adoption_worktree(settings: MultiAgentSettings, branch_name: str) -> tuple[Path | None, str | None]:
    root = settings.worktree_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    worktree = root / f".adopt-{uuid4().hex[:12]}"
    add = _run(["git", "worktree", "add", str(worktree), branch_name], settings.resolved_target_repo)
    if add.returncode != 0:
        return None, f"unable to create adoption worktree: {_output(add)}"
    return worktree.resolve(), None


def run_adopted_branch_repair(
    settings: MultiAgentSettings,
    branch_name: str,
    *,
    harness_builder: Callable[[MultiAgentSettings], MultiAgentHarness] = MultiAgentHarness.build,
    publisher: Callable[..., PullRequestPublication] = publish_and_merge_repair,
) -> AdoptedBranchAttempt:
    """Review or retry one explicit retained product repair branch."""

    try:
        branch_name = _validate_branch(branch_name, require_fix=True)
    except ValueError as exc:
        return AdoptedBranchAttempt(branch_name=branch_name, status="rejected", summary=str(exc))
    head_sha = _branch_head(settings, branch_name)
    if head_sha is None:
        return AdoptedBranchAttempt(branch_name=branch_name, status="rejected", summary="repair branch does not exist")
    has_product_diff, diff_error = _has_product_diff(settings, branch_name)
    if diff_error:
        return AdoptedBranchAttempt(branch_name=branch_name, status="blocked", summary=diff_error)
    if not has_product_diff:
        deleted = _run(["git", "branch", "-D", branch_name], settings.resolved_target_repo)
        if deleted.returncode == 0:
            return AdoptedBranchAttempt(branch_name=branch_name, status="deleted", summary="no product diff from current origin/main")
        return AdoptedBranchAttempt(
            branch_name=branch_name,
            status="blocked",
            summary=f"no product diff, but branch deletion failed: {_output(deleted)}",
        )
    product_error = _product_only_diff(settings, branch_name)
    if product_error:
        return AdoptedBranchAttempt(branch_name=branch_name, status="rejected", summary=product_error)
    context = _matching_adoption_context(settings, branch_name, head_sha)
    if isinstance(context, str):
        return AdoptedBranchAttempt(branch_name=branch_name, status="rejected", summary=context)
    rebased_head, rebase_error = _rebase_adopted_branch(settings, context)
    if rebase_error or rebased_head is None:
        return AdoptedBranchAttempt(
            branch_name=branch_name,
            status="blocked",
            scenario_name=context.scenario.name,
            artifact_dir=str(context.artifact_dir),
            summary=rebase_error or "unable to rebase adopted branch",
        )
    worktree, worktree_error = _create_adoption_worktree(settings, branch_name)
    if worktree_error or worktree is None:
        return AdoptedBranchAttempt(
            branch_name=branch_name,
            status="blocked",
            scenario_name=context.scenario.name,
            artifact_dir=str(context.artifact_dir),
            rebased_from_sha=context.head_sha if rebased_head != context.head_sha else None,
            summary=worktree_error or "unable to create adoption worktree",
        )
    adopted_debugger = context.debugger_report.model_copy(
        update={
            "branch_name": branch_name,
            "worktree_path": str(worktree),
            "commit_sha": rebased_head,
            "status": "patched",
        }
    )
    try:
        scoped = replace(settings, active_target_root=worktree)
        report = asyncio.run(
            harness_builder(scoped).repair_scenario(
                context.scenario,
                existing_branch_name=branch_name,
                existing_worktree_path=str(worktree),
                adopted_debugger_report=adopted_debugger,
            )
        )
    except Exception as exc:
        cleanup_worktree(settings, str(worktree))
        return AdoptedBranchAttempt(
            branch_name=branch_name,
            status="blocked",
            scenario_name=context.scenario.name,
            artifact_dir=str(context.artifact_dir),
            rebased_from_sha=context.head_sha if rebased_head != context.head_sha else None,
            summary=f"adopted branch harness failed: {type(exc).__name__}",
        )
    publication: PullRequestPublication | None = None
    if report.status == "pass" and report.artifact_dir:
        publication = publisher(settings, branch_name=branch_name, artifact_dir=Path(report.artifact_dir))
        Path(report.artifact_dir, "pull_request_publication.json").write_text(publication.model_dump_json(indent=2) + "\n")
    status = "merged" if publication and publication.status == "merged" else "failed" if report.status == "fail" else "blocked" if report.status == "blocked" else "failed"
    summary = publication.summary if publication is not None else report.summary
    return AdoptedBranchAttempt(
        branch_name=branch_name,
        status=status,
        scenario_name=context.scenario.name,
        artifact_dir=report.artifact_dir,
        rebased_from_sha=context.head_sha if rebased_head != context.head_sha else None,
        harness_report=report,
        publication=publication,
        summary=summary,
    )


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
    adoption_runner: Callable[[MultiAgentSettings, str], AdoptedBranchAttempt] = run_adopted_branch_repair,
    base_refresher: Callable[[MultiAgentSettings], str | None] = prepare_control_baseline,
    adopt_branches: list[str] | None = None,
) -> ValidationRepairReport:
    """Validate eight live scenarios, repair each failure once, then revalidate."""

    store = HarnessArtifactStore.create(settings, "multiagent_validation_repair")
    store.write_model("model_preflight.json", model_preflight)

    baseline = evaluator(settings)
    store.write_model("baseline_validation.json", baseline)
    adopted_branches: list[AdoptedBranchAttempt] = []
    attempts: list[ValidationRepairAttempt] = []
    current = baseline
    attempted: set[str] = set()
    baseline_error: str | None = None

    for branch_name in adopt_branches or []:
        adopted = adoption_runner(settings, branch_name)
        adopted_branches.append(adopted)
        store.write_model(f"adopted/{len(adopted_branches)}/{branch_name.replace('/', '_')}.json", adopted)
        if adopted.scenario_name:
            attempted.add(adopted.scenario_name)
        if adopted.status == "merged":
            baseline_error = base_refresher(settings)
            if baseline_error is not None:
                break
            current = evaluator(settings)
            store.write_model(f"adopted_retests/{len(adopted_branches)}_validation.json", current)

    while baseline_error is None:
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
            baseline_error = base_refresher(settings)
            if baseline_error is not None:
                break
            current = evaluator(settings)
            store.write_model(f"retests/{len(attempts)}_validation.json", current)

    final = evaluator(settings)
    store.write_model("final_validation.json", final)
    status = "blocked" if baseline_error else _validation_status(final)
    summary = (
        baseline_error
        if baseline_error
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
        adopted_branches=adopted_branches,
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
