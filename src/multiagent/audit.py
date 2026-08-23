from __future__ import annotations

import subprocess
from pathlib import Path

from multiagent.publisher import _load_production_artifact, _validate_branch
from multiagent.schemas import HarnessRunReport, PullRequestPublication, RepairAuditRecord, RepairAuditReport
from multiagent.settings import MultiAgentSettings


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=False)


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return ((result.stdout or "") + (result.stderr or "")).strip()


def discover_repair_branches(settings: MultiAgentSettings) -> list[str]:
    result = _run(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads"], settings.resolved_target_repo)
    if result.returncode != 0:
        return []
    return sorted(
        branch
        for branch in result.stdout.splitlines()
        if branch.startswith(("fix/", "retrieval-fix", "formation-energy"))
    )


def _matching_artifact(settings: MultiAgentSettings, branch_name: str) -> tuple[Path | None, HarnessRunReport | None]:
    matches: list[tuple[Path, HarnessRunReport]] = []
    for path in settings.resolved_artifact_root.glob("*/harness_run_report.json"):
        try:
            report = HarnessRunReport.model_validate_json(path.read_text())
        except Exception:
            continue
        if report.branch_name == branch_name:
            matches.append((path.parent, report))
    if not matches:
        return None, None
    return max(matches, key=lambda item: item[0].name)


def _artifact_publication(artifact_dir: Path | None) -> PullRequestPublication | None:
    if artifact_dir is None:
        return None
    path = artifact_dir / "pull_request_publication.json"
    if not path.is_file():
        return None
    try:
        return PullRequestPublication.model_validate_json(path.read_text())
    except Exception:
        return None


def _diff_paths(settings: MultiAgentSettings, branch_name: str) -> tuple[list[str], str | None]:
    result = _run(
        ["git", "diff", "--name-only", f"{settings.target_base_ref}...{branch_name}"],
        settings.resolved_target_repo,
    )
    if result.returncode != 0:
        return [], f"unable to inspect product diff: {_output(result)}"
    return [path for path in result.stdout.splitlines() if path], None


def _product_only_diff(settings: MultiAgentSettings, branch_name: str) -> tuple[bool, str | None]:
    paths, error = _diff_paths(settings, branch_name)
    if error:
        return False, error
    disallowed = [
        path
        for path in paths
        if path and not (path.startswith("src/matsci_agent/") or path.startswith("tests/"))
    ]
    if disallowed:
        return False, f"repair diff includes non-product paths: {', '.join(disallowed)}"
    return True, None


def audit_repair_branch(settings: MultiAgentSettings, branch_name: str) -> RepairAuditRecord:
    reasons: list[str] = []
    head = _run(["git", "rev-parse", "--verify", branch_name], settings.resolved_target_repo)
    if head.returncode != 0:
        return RepairAuditRecord(branch_name=branch_name, status="rejected", reasons=["branch does not exist"])
    head_sha = head.stdout.strip()
    try:
        _validate_branch(branch_name, require_fix=True)
        safe_namespace = True
    except ValueError as exc:
        safe_namespace = False
        reasons.append(str(exc))

    ancestor = _run(["git", "merge-base", "--is-ancestor", settings.target_base_ref, branch_name], settings.resolved_target_repo)
    descends = ancestor.returncode == 0
    if not descends:
        reasons.append("branch is not descended from current origin/main")

    diff_paths, _ = _diff_paths(settings, branch_name)
    product_only, product_error = _product_only_diff(settings, branch_name)
    if product_error:
        reasons.append(product_error)

    artifact_dir, report = _matching_artifact(settings, branch_name)
    evidence_sha = report.latest_debugger_report.commit_sha if report and report.latest_debugger_report else None
    evidence_matches = evidence_sha == head_sha
    debugger_committed = bool(report and report.latest_debugger_report and report.latest_debugger_report.status == "patched" and evidence_sha)
    changed_tests_proven = bool(report and report.latest_repair_test_evidence and report.latest_repair_test_evidence.status == "pass")
    verifier_accepted = bool(report and report.latest_verifier_report and report.latest_verifier_report.status == "accepted")
    fresh_live_passed = bool(
        report
        and report.status == "pass"
        and report.stop_reason == "dual_review_pass"
        and report.latest_tester_report
        and report.latest_tester_report.live_evaluation
        and report.latest_tester_report.live_evaluation.status == "pass"
        and report.latest_tester_report.live_evaluation.real_source_used
    )
    publication = _artifact_publication(artifact_dir)
    if report is None or artifact_dir is None:
        reasons.append("no stored harness evidence matches branch")
    else:
        artifact_error = _load_production_artifact(artifact_dir, branch_name, expected_head_sha=head_sha)
        if artifact_error:
            reasons.append(artifact_error)
        if not evidence_matches:
            reasons.append("debugger commit SHA does not match branch head")

    return RepairAuditRecord(
        branch_name=branch_name,
        head_sha=head_sha,
        status="eligible" if not reasons else "rejected",
        reasons=reasons,
        artifact_dir=str(artifact_dir) if artifact_dir else None,
        evidence_commit_sha=evidence_sha,
        diff_paths=diff_paths,
        safe_namespace=safe_namespace,
        descends_from_target_main=descends,
        product_only_diff=product_only,
        evidence_matches_head=evidence_matches,
        debugger_committed=debugger_committed,
        changed_tests_proven=changed_tests_proven,
        verifier_accepted=verifier_accepted,
        fresh_live_passed=fresh_live_passed,
        publication_status=publication.status if publication else None,
    )


def audit_repair_branches(settings: MultiAgentSettings, branch_names: list[str] | None = None) -> RepairAuditReport:
    branches = branch_names if branch_names is not None else discover_repair_branches(settings)
    records = [audit_repair_branch(settings, branch_name) for branch_name in sorted(set(branches))]
    base = _run(["git", "rev-parse", settings.target_base_ref], settings.resolved_target_repo)
    return RepairAuditReport(
        status="pass" if all(record.status == "eligible" for record in records) else "fail",
        target_base_branch=settings.target_base_branch,
        target_base_sha=base.stdout.strip() if base.returncode == 0 else None,
        records=records,
    )
